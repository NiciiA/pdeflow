"""Reference→disclosure/observations→reference-free sampling→evaluation orchestration."""
from dataclasses import asdict,replace
from pathlib import Path
import numpy as np
import torch
from pdeflow import local
from pdeflow.runtime import RuntimeSpec,read_json,write_json,file_digest,InferenceConfig
from pdeflow.core import ProblemSpec
from pdeflow.observations import ObservationSet
from pdeflow.training import _read_checkpoint,load_inference
from pdeflow.inference import validate_compatibility,selection_key
from pdeflow.sampling import sample as reconstruct
from pdeflow.evaluation import export_result
from . import create,load
from .observations import ObservationSamplingSpec,sample

def reference_path(selector,workspace=None):
    p=Path(selector)
    if p.exists():return p.parent if p.is_file() else p
    if len(selector)!=32 or any(c not in '0123456789abcdef' for c in selector):raise ValueError('Reference must be an existing path or32-character artifact ID')
    return local.workspace_root(workspace)/'workbench/data'/selector

def checkpoint(selector,problem,inference,workspace=None):
    entries=[{'path':selector}] if selector!='auto' and Path(selector).is_file() else local.model_entries(workspace)
    errors=[];compatible=[]
    for entry in entries:
        path=Path(entry['path'])
        try:
            identifier=file_digest(path)[:32]
            if selector!='auto' and not Path(selector).is_file() and identifier!=selector:continue
            cp,config,_=_read_checkpoint(path);validate_compatibility(problem,config,inference)
            compatible.append((selection_key(identifier,{**entry,'config':config,'checkpoint':cp,'export_kinds':[entry.get('export_kind','linked')]}),path))
        except (ValueError,OSError) as error:errors.append(str(error))
    if compatible:return min(compatible,key=lambda x:x[0])[1]
    raise ValueError('No compatible inference checkpoint. Train the matching preset or use pdeflow models --import PATH. '+('; '.join(errors[:4])))

def experiment(spec,sampling: ObservationSamplingSpec,output,model='auto',workspace=None,device='cpu',inference=None,mode='observation_pde',stop_after=None,existing_reference=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    manifest={'schema_version':1,'kind':'synthetic_inverse_experiment','created_at':local.now(),'state':'preparing','artifacts':{},'reference_in_worker':False,'mode':mode}
    def save(**changes):manifest.update(changes);write_json(output/'experiment.json',manifest)
    try:
        reference=reference_path(existing_reference,workspace) if existing_reference else output/'reference'
        if not existing_reference:create(spec,reference)
        save(state='reference_ready',artifacts={'reference':str(reference.resolve())})
        if stop_after=='reference':return manifest
        sample(reference,sampling,output/'observations')
        save(state='observations_ready',artifacts={**manifest['artifacts'],'observations':'observations','inverse_request':'observations/inverse-request.json'})
        if stop_after=='observations':return manifest
        inverse=read_json(output/'observations/inverse-request.json')
        p=ProblemSpec.from_dict(inverse['problem']);obs=ObservationSet.from_dict(inverse['observations'],p.grid)
        inf=InferenceConfig(**(inference or {'steps':4,'particles':2}))
        path=checkpoint(model,p,inf,workspace)
        runtime=RuntimeSpec(device=device)
        bundle=load_inference(path,runtime)
        save(state='reconstructing',checkpoint_sha256=file_digest(path),inverse_identity=read_json(output/'observations/sampling.json')['inverse_identity'],resolved_inference=asdict(inf),runtime=asdict(runtime))
        # This function and all controller decisions receive no reference artifact.
        result=reconstruct(bundle,p,obs,inf,mode)
        final=ProblemSpec.from_dict(result.provenance.get('final_problem',p.to_dict()))
        final_obs=ObservationSet.from_dict(result.provenance['final_observations'],final.grid) if 'final_observations' in result.provenance else obs
        reference_problem,truth,reference_manifest=load(reference)
        if final!=reference_problem:raise ValueError('Experiment final grid/problem differs: use explicit evaluation comparison policy; no silent array transfer')
        report=export_result(output/'reconstruction',result,final,final_obs,torch.from_numpy(truth))
        from pdeflow.specialists import get_specialist
        baselines=get_specialist(final.family).baseline(final,final_obs,inf)
        for name,baseline in baselines.items():export_result(output/('baseline_'+name),baseline,final,final_obs,torch.from_numpy(truth))
        save(state='completed',reference_id=reference_manifest['id'],finished_at=local.now(),artifacts={**manifest['artifacts'],'reconstruction':'reconstruction/result.npz','metrics':'reconstruction/comparison.json','figure':'reconstruction/reconstruction.png','baselines':['baseline_'+name for name in baselines]})
        return manifest
    except BaseException as error:
        save(state='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=f'{type(error).__name__}: {error}')
        raise
