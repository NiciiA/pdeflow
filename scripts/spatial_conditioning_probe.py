"""Bounded NEW B experiments and deterministic/kinematic controls. Never edits parents.

Not a scientific accuracy study. All variants share the same frozen A1/A2,
training recipe, effective batch, seed and budget. Test split is never accessed.
"""
from pathlib import Path
from dataclasses import replace
import argparse,time
import numpy as np
import torch
from pdeflow.runtime import write_json,read_json,file_digest,atomic_write
from pdeflow.coordinate.training import read_checkpoint,portable_checkpoint
from pdeflow.coordinate.experiments import conditional_experiment,copy_frozen_dependencies
from pdeflow.coordinate.inference import CoordinateBundle
from pdeflow.coordinate.conditioning import prepare
from pdeflow.coordinate.contracts import grid_queries
from pdeflow.coordinate.diagnostic_cases import load_case,sensor_truth
from pdeflow.coordinate.diagnostic_metrics import sensor_metrics
from pdeflow.coordinate.diagnostics import exact_sensors,shuffle_within_quantity
from pdeflow.coordinate.diagnostic_inventory import oracle_model
from pdeflow.coordinate.diagnostic_baseline import DirectLatent,train_baseline
from pdeflow.coordinate.model import Autoencoder
from pdeflow.coordinate.joint import active_mask
from pdeflow.coordinate.physics import get
from pdeflow.core import GridSpec


def evaluate(bundle,case_root,head=None,ae=None):
    """Same immutable cases/measurements and fixed Gaussian noise; no hash rewrite."""
    case,p,solver,context,noise=load_case(case_root);c=bundle.config;dev=bundle.device;rows=case['observations'];true=case['true_parameter_evaluation_only'];held=case['heldout_observations_evaluation_only']
    times=tuple(sorted(set([0.,.01*c.interval[1],.237*c.interval[1],c.interval[1]])));grid=GridSpec((12,12),p.grid.domain,times);q=grid_queries(grid);norms=bundle.cp['normalizations'][p.family]
    def metrics(z,packed):
        fields={}
        with torch.no_grad():
            for j,name in enumerate(get(p.family).state):
                pred=bundle.decode(z,q.to(dev),packed.parameter,packed.known,packed.families,name).cpu().numpy().reshape(len(z),len(times),12,12);truth=solver.evaluate(q.numpy(),true,c.interval,name).reshape(len(times),12,12);mse=np.mean((pred-truth)**2,(2,3));mean=np.mean((pred.mean(0)-truth)**2,(1,2))
                fields[name]={'times':times,'per_sample_mse_per_time':mse.tolist(),'ensemble_mean_mse_per_time':mean.tolist(),'ensemble_mean_rmse_per_time':np.sqrt(mean).tolist(),'normalized_mse_per_time':(mean/norms[j]['scale']**2).tolist(),'spread_rms':float(np.sqrt(pred.var(0).mean()))}
            sensors=exact_sensors(bundle,z,packed.parameter,packed.known,packed.families,rows);unseen=exact_sensors(bundle,z,packed.parameter,packed.known,packed.families,held);parameter=bundle.ae.decode_parameter(z,packed.parameter,packed.known,True,packed.families).cpu().tolist()
        return {'fields':fields,'conditioning':sensor_metrics(sensors,rows,sensor_truth(solver,rows,true,c.interval)),'heldout':sensor_metrics(unseen,held,[r['value'] for r in held]),'parameter_samples':parameter,'parameter_truth_evaluation_only':true,'parameter_absolute_errors':[abs(x-true) for x in parameter]}
    variants={'correct':rows,'shuffled_values':shuffle_within_quantity(rows,np.random.default_rng(194))}
    results={};previous=bundle.ae
    if ae is not None:bundle.ae=ae
    try:
        for label,observations in variants.items():
            count=1 if head else len(noise);packed=prepare(c,bundle.cp['normalizations'],p,observations,count,dev)
            with torch.no_grad():
                graph=bundle.ae.graph_encoder(packed.families);condition=bundle.velocity.condition(packed.parameter,packed.observations,packed.known,packed.families,graph)
                if head:
                    raw=bundle.velocity.observation_encoder.tokenize(packed.observations[:3],packed.observations[3]);sensors=head.sensor_attention.prepare(*raw) if head.sensor_attention else None
                    standardized=head(bundle.velocity.geometry.embedding()[None],condition[0],active_mask(c,packed.known,packed.families),sensors)
                else:
                    standardized=torch.tensor(noise,device=dev)*active_mask(c,packed.known,packed.families)[...,None]
                    for k in range(8):
                        s=torch.full((count,),k/8,device=dev);v=bundle.velocity(standardized,s,conditioning=condition);standardized=standardized+(v+bundle.velocity(standardized+v/8,s+1/8,conditioning=condition))/16
                z=standardized*bundle.scale+bundle.mean
            results[label]=metrics(z,packed)
    finally:bundle.ae=previous
    return {'case_identity':case['case_identity'],'original_case_checkpoint':case['checkpoint_sha256'],'evaluated_checkpoint':bundle.fingerprint,'family':p.family,'known_parameter':bool(p.known_parameters),'sensor_count':len(rows),'sensor_times':[r['time'] for r in rows],'variants':results,'meaning':'Immutable case, exact sensor/heldout coordinates and time; fields12² at four declared times. All four flow samples, no best-of-N. Deterministic head has one output. Incorrect-pairing errors scored against ORIGINAL observations/truth; changes alone are not improvements.'}


def cases_for(parent,root,requested=None):
    if requested is not None:
        if not requested.is_dir():raise ValueError('Requested immutable case directory is missing')
        return requested
    existing=Path('outputs/trained-model-analysis-20261101/cases')
    if existing.is_dir():return existing
    from pdeflow.coordinate.diagnostic_cases import capture
    destination=root/'probe-cases'
    if not destination.exists():
        for name,index in (('validation-cosine',0),('validation-travelling',3),('validation-velocity',7)):
            capture(parent/'flow/coordinate.pt',destination/name,validation_index=index,training_observations=True,particles=4,steps=8,resolution=12)
    return destination


def b_probes(parent,root,device,updates,cases=None):
    from pdeflow.coordinate.engine import train
    from pdeflow.source_bundle import package
    from pdeflow.training_artifacts import build
    baseline,old=read_checkpoint(parent/'flow/coordinate.pt','flow');cases=cases_for(parent,root,cases);results={}
    for name in ('B0','B1','B2','B3'):
        destination=root/name
        if destination.exists():raise FileExistsError('Never overwrite an experiment: '+str(destination))
        destination.mkdir(parents=True);copy_frozen_dependencies(parent,destination)
        cfg=conditional_experiment(replace(old,flow_steps=updates,batch_size=4,checkpoint_every=updates,runtime={'device':device,'dtype':'float32','threads':2},training_options={**old.training_options,'precision':'fp32','microbatch':2,'accumulation':2,'stage_batches':{},'batch_selection':None,'flow_warm_start':str(parent/'flow/coordinate.pt'),'dependency_source_policy':'verified_frozen_v2','validation_every':updates,'validation_subset':2,'profile_components':True,'flow':{'learning_rate':.0003,'schedule':'constant','warmup':0}}),name,{'warmup_updates':0})
        # Capture source before training. Canonical copied parents are immutable.
        meta=package(destination/'source/source.zip');write_json(destination/'source/identity.json',meta)
        before={s:file_digest(destination/s/(s+'.pt')) for s in ('autoencoder','evolution')};started=time.perf_counter();record=train(cfg.to_dict(),destination/'flow','flow',destination/'evolution/evolution.pt');record['probe_wall_seconds']=time.perf_counter()-started
        assert before=={s:file_digest(destination/s/(s+'.pt')) for s in before}
        cp,_=read_checkpoint(destination/'flow/resume.pt');(destination/'export').mkdir();portable=portable_checkpoint(cp);portable['dataset_root']=None;atomic_write(destination/'export/coordinate.pt',lambda f:torch.save(portable,f));write_json(destination/'export/coordinate.json',{'architecture':cp['architecture'],'probe':True});views=build(destination)
        bundle=CoordinateBundle(destination/'export/coordinate.pt',device);evaluations={}
        for label in ('validation-cosine','validation-travelling','validation-velocity'):
            evaluations[label]=evaluate(bundle,cases/label)
        history=cp['history'];measured=history[2:];result={'config':cfg.to_dict(),'warm_start':cp['warm_start'],'frozen_dependency_audit':cp['frozen_dependency_audit'],'successful_updates':cp['progress'],'frozen_hashes_unchanged':before,'history':history,'evaluation':evaluations,'batch':4,'actual_microbatch':2,'precision':'fp32','parameters':record['parameters'],'memory':record['memory'],'setup_seconds':record['setup_seconds'],'wall_seconds_including_export_validation':record['probe_wall_seconds'],'warmup_step_seconds':[r['step_seconds'] for r in history[:2]],'median_measured_step_seconds':float(np.median([r['step_seconds'] for r in measured])) if measured else None,'artifact_views_verified':list(views),'limits':'Tiny NEW warm-start probe, not validated accuracy. Synchronized step_seconds excludes validation/persistence; row end_to_end_seconds and whole call include their respective boundary costs; allocator peaks process-local CUDA only; MPS current is not peak. Not T4 throughput evidence.'}
        write_json(destination/'probe.json',result);results[name]=result;print('FINISHED',name,flush=True);del bundle,cp
    write_json(root/'b-comparison.json',results)


def controls(parent,root,device,steps,cases=None):
    cases=cases_for(parent,root,cases)
    headroot=root/'deterministic-spatial';report=train_baseline(parent/'flow/coordinate.pt',headroot,parent/'evolution/evolution.pt',parent/'autoencoder/autoencoder.pt',device,steps=steps,train_cases=4,validation_cases=4,spatial=True)
    bundle=CoordinateBundle(parent/'flow/coordinate.pt',device);bundle.checkpoint_path=str(parent/'flow/coordinate.pt');ae,_=oracle_model(bundle,parent/'evolution/evolution.pt',parent/'autoencoder/autoencoder.pt');head=DirectLatent(bundle.config,True).to(device);head.load_state_dict(torch.load(headroot/'head.pt',weights_only=True,map_location=device)['head']);head.eval();evaluations={}
    for label in [x for x in ('saved-periodic','validation-cosine','validation-travelling','validation-velocity') if (cases/x).exists()]:
        evaluations[label]={'head':evaluate(bundle,cases/label,head,ae),'baseline_flow':evaluate(bundle,cases/label)}
    write_json(headroot/'same-case-evaluation.json',evaluations)
    print('DETERMINISTIC COMPLETE',flush=True)


def kinematic(root,device):
    from pdeflow.colab import training_preset
    from pdeflow.coordinate.evolution_objective import decoded_kinematic
    torch.manual_seed(415);c=training_preset('smoke',device,'fp32');ae=Autoencoder(c).to(device).eval().requires_grad_(False);ae.evolution.requires_grad_(True);opt=torch.optim.Adam(ae.evolution.parameters(),lr=.001)
    z=torch.randn(1,c.tokens,c.width,device=device);q=torch.rand(1,4,3,device=device);p=torch.tensor([.5],device=device);known=torch.ones(1,dtype=torch.bool,device=device);f=torch.ones(1,dtype=torch.long,device=device);norms={'wave':[{'mean':0.,'scale':1.},{'mean':0.,'scale':1.}]};history=[]
    frozen={n:t.clone() for n,t in ae.state_dict().items() if not n.startswith('evolution.')}
    for i in range(12):
        opt.zero_grad();r=decoded_kinematic(ae,z,q,p,known,f,norms);loss=r.square().mean();loss.backward();grad=torch.nn.utils.clip_grad_norm_(ae.evolution.parameters(),10.,error_if_nonfinite=True);opt.step();history.append({'step':i+1,'kinematic_mse':float(loss.detach()),'gradient_norm':float(grad)})
    assert all(torch.equal(t,ae.state_dict()[n]) for n,t in frozen.items());write_json(root/'kinematic-probe.json',{'history':history,'device':device,'frozen_representation_unchanged':True,'scope':'12-update random tiny T/D fixed-probe derivative learning check. Only kinematic term in this gradient control; production A2 retains supervised u/v, alignment/composition. No wave accuracy result, no trained-parent modification.'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',choices=['cpu','mps','cuda'],default='cpu');p.add_argument('--cases',type=Path,help='Existing immutable case root; otherwise reuse checkout analysis or generate and save shared validation controls');p.add_argument('--mode',choices=['b','controls','kinematic'],required=True);p.add_argument('--updates',type=int,default=8);a=p.parse_args()
    if not 4<=a.updates<=40:raise ValueError('Bounded4..40 updates only')
    torch.set_num_threads(2);a.output.mkdir(parents=True,exist_ok=True)
    if a.mode=='b':b_probes(a.parent.resolve(),a.output.resolve(),a.device,a.updates,a.cases)
    elif a.mode=='controls':controls(a.parent.resolve(),a.output.resolve(),a.device,a.updates,a.cases)
    else:kinematic(a.output.resolve(),a.device)
