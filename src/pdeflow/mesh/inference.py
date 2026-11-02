"""Reference-free mesh inference. One sampled state is reused for every query/time."""
from dataclasses import replace
from pathlib import Path
from functools import lru_cache
import time
import numpy as np
import torch
from pdeflow.runtime import RuntimeSpec,file_digest,read_json,write_json,atomic_write,synchronize
from pdeflow.core import ProblemSpec,GridSpec
from pdeflow.coordinate.contracts import validate_rows,validate_queries,grid_queries
from pdeflow.coordinate.physics import get,invariant_drift
from .model import MeshModel
from .provenance import read_checkpoint
from .state import parameter_normalize,parameter_physical
from .sampling import generate


def compatible(c,problem):
    c.reference_config().compatible(problem)
    return True


def prepare(c,norms,problem,rows,count,device):
    reference=c.reference_config();points,values,sigma=validate_rows(reference,rows,problem);f=torch.full((count,),int(problem.family=='wave'),device=device,dtype=torch.long);name=get(problem.family).parameter;known=torch.full((count,),name in problem.known_parameters,device=device,dtype=torch.bool)
    p=parameter_normalize(c,torch.full((count,),problem.known_parameters[name],device=device),f) if known.all() else torch.zeros(count,device=device);n=max(1,len(rows));q=torch.zeros(count,n,3,device=device);feat=torch.zeros_like(q);valid=torch.zeros(count,n,dtype=torch.bool,device=device);qty=torch.zeros(count,n,dtype=torch.long,device=device)
    if rows:
        q[:,:len(rows)]=points.to(device);valid[:,:len(rows)]=True;ids=torch.tensor([int(r['field']=='v') for r in rows],device=device);qty[:,:len(rows)]=ids
        for j in range(2 if problem.family=='wave' else 1):
            ix=torch.where(ids.eq(j))[0];norm=norms[problem.family][j];feat[:,ix,0]=(values[ix.cpu()].to(device)-norm['mean'])/norm['scale'];feat[:,ix,1]=sigma[ix.cpu()].to(device)/norm['scale'];feat[:,ix,2]=1
    return (q,feat,valid,qty),p,known,f


class MeshBundle:
    def __init__(self,path,device='cpu'):
        self.path=str(Path(path).resolve());self.cp,self.config=read_checkpoint(path)
        if not self.cp.get('inference_ready'):raise ValueError('Checkpoint contains no trained final reconstruction/generation stage; use its diagnostic bundle')
        self.device,_=RuntimeSpec(device=device,threads=2).resolve();self.model=MeshModel(self.config).to(self.device).eval().requires_grad_(False);self.model.load_state_dict(self.cp['model'],strict=True);self.fingerprint=file_digest(path)
    def decode(self,state,points,families,quantity):
        q=validate_queries(self.config.reference_config(),points).to(self.device);ids=torch.full((len(q),),int(quantity=='v'),device=self.device,dtype=torch.long);out=self.model.decode_points(state,q,ids,families);norm=self.cp['normalizations'][('heat','wave')[int(families[0])]][int(quantity=='v')];return out*norm['scale']+norm['mean']
    def sample(self,problem,rows,queries,steps=None,particles=2,seed=91,chunk_size=128,initial_noise=None,solver=None,correction=None,progress=None):
        c=self.config;compatible(c,problem)
        if correction:raise ValueError('Mesh correction is not enabled; use uncorrected outputs or explicit legacy correction')
        if not 1<=particles<=16 or chunk_size<1:raise ValueError('Bounded1..16 particles and positive query chunk required')
        if not rows and c.reference_config().observation_counts[0]>0 and c.reference_config().unconditional_probability==0:raise ValueError('This checkpoint was not trained with zero observations')
        count=particles if c.kind['generative'] else 1
        if self.device.type=='cuda':torch.cuda.reset_peak_memory_stats(self.device)
        points=validate_queries(c.reference_config(),queries)
        if not c.kind['trajectory'] and bool(points[:,-1].ne(0).any()):raise ValueError('M1 is initial-state only. Measurement times can be later; reconstruction queries must be t=0.')
        obs,p,known,f=prepare(c,self.cp['normalizations'],problem,rows,count,self.device);started=time.perf_counter();self.model.decoder.c=replace(c,query_chunk=min(chunk_size,65536))
        with torch.no_grad():
            context=self.model.condition(obs,p,known,f);synchronize(self.device);tick=time.perf_counter();state,sampler=generate(self.model,context,p,known,f,torch.Generator().manual_seed(seed),initial_noise,steps,solver);synchronize(self.device);generation_seconds=time.perf_counter()-tick;tick=time.perf_counter();all_values=self.model.query(state,points.to(self.device),f);norms=self.cp['normalizations'][problem.family];u=all_values[...,0]*norms[0]['scale']+norms[0]['mean'];self.velocity_samples=(all_values[...,1]*norms[1]['scale']+norms[1]['mean']).cpu().numpy() if problem.family=='wave' else None;synchronize(self.device);decode_seconds=time.perf_counter()-tick;scalar=parameter_physical(c,state[:,-1,0],f)
            evaluable=rows if c.kind['trajectory'] else [r for r in rows if r['time']==0]
            sensor=np.empty((count,len(evaluable)))
            for j,name in enumerate(get(problem.family).state):
                indices=[i for i,r in enumerate(evaluable) if r['field']==name]
                if indices:
                    q=torch.tensor([[evaluable[i]['x'],evaluable[i]['y'],evaluable[i]['time']] for i in indices]);sensor[:,indices]=self.decode(state,q,f,name).cpu().numpy()
        samples=u.cpu().numpy()
        if not np.isfinite(samples).all() or self.velocity_samples is not None and not np.isfinite(self.velocity_samples).all():raise FloatingPointError('Nonfinite mesh reconstruction; no success-shaped fallback')
        physical=get(problem.family);invariants=None
        if len(points)==np.prod(problem.grid.shape)*len(problem.grid.times):
            try:invariants=invariant_drift(samples.reshape(count,len(problem.grid.times),*problem.grid.shape),self.velocity_samples.reshape(count,len(problem.grid.times),*problem.grid.shape) if self.velocity_samples is not None else None,problem.grid.times,c.domain['lengths'],scalar.cpu().numpy())
            except ValueError:pass
        consistency=None if not evaluable else {'predictions_per_sample':sensor.tolist(),'rmse_against_noisy_observed_values_per_sample':np.sqrt(np.mean((sensor-np.array([r['value'] for r in evaluable]))**2,1)).tolist(),'sensor_coordinates_and_times':[[r['x'],r['y'],r['time']] for r in evaluable],'quantities':[r['field'] for r in evaluable],'evaluation_policy':'Exact stored physical sensor times and quantities; M1 only evaluates available t=0 sensors'}
        report={'architecture':c.architecture,'candidate':c.kind['name'],'checkpoint_sha256':self.fingerprint,'checkpoint_updates':self.cp['successful_updates'],'training_configuration':{'architecture':c.architecture,'state_contract':self.cp['state_contract'],'observation_time':c.reference_config().observation_time,'stages':self.cp.get('training_provenance',{}),'objective':c.objective,'legacy':False},'device':str(self.device),'dtype':'float32','seconds':time.perf_counter()-started,'generation_seconds':generation_seconds,'physical_evolution_and_query_decode_seconds':decode_seconds,'sampler':sampler,'memory':{'cuda_peak_allocated':torch.cuda.max_memory_allocated(self.device) if self.device.type=='cuda' else None,'cuda_peak_reserved':torch.cuda.max_memory_reserved(self.device) if self.device.type=='cuda' else None,'mps_current_allocated':torch.mps.current_allocated_memory() if self.device.type=='mps' else None,'scope':'CUDA peak; MPS current allocation is not peak; CPU process peak unavailable'},'query_workload':{'points':len(points),'quantities':len(get(problem.family).state),'particles':count,'decoder_chunk':chunk_size,'internal_nodes':c.nodes},'sample_count':count,'prediction_kind':'generative samples' if c.kind['generative'] else 'deterministic prediction','initial_only':not c.kind['trajectory'],'no_reference_input':True,'inferred_parameters':{physical.parameter:scalar.cpu().tolist()},'parameter_status':'known' if bool(known.all()) else 'inferred positive coefficient','observation_count':len(rows),'observation_consistency':consistency,'sensor_diagnostics_rows':evaluable,'unevaluable_later_sensors_for_M1':len(rows)-len(evaluable),'normalizations':self.cp['normalizations'][problem.family],'field_loss_provenance':c.field_loss,'field_normalization_scale':self.cp['normalizations'][problem.family][0]['scale'],'physics':{'invariants':invariants,'meaning':'Grid diagnostic, not exact learned conservation'},'correction':None,'uncertainty':'Empirical small ensemble; not calibrated','physical_state_reuse':'One generated initial physical state and coefficient reused for every coordinate/time; no redraw per query chunk'}
        self.state=state.detach().cpu();self.families=f.cpu();self.parameter=p.cpu();self.known=known.cpu()
        if progress:progress(1,1)
        return samples,report


def reconstruct(checkpoint,problem,rows,output,queries=None,steps=None,particles=2,seed=91,chunk_size=128,device='cpu',solver=None,correction=None,progress=None):
    p=ProblemSpec.from_dict(problem) if isinstance(problem,dict) else problem;bundle=MeshBundle(checkpoint,device);input_problem=p
    if queries is None and not bundle.config.kind['trajectory']:p=replace(p,grid=replace(p.grid,times=(0.,)))
    q=grid_queries(p.grid) if queries is None else torch.as_tensor(queries);samples,report=bundle.sample(input_problem,rows,q,steps,particles,seed,chunk_size,solver=solver,correction=correction,progress=progress);root=Path(output);root.mkdir(parents=True,exist_ok=True);arrays={'samples':samples,'queries':q.numpy()}
    if bundle.velocity_samples is not None:arrays['velocity_samples']=bundle.velocity_samples
    atomic_write(root/'result.npz',lambda f:np.savez_compressed(f,**arrays));atomic_write(root/'state.pt',lambda f:torch.save({'state':bundle.state,'families':bundle.families,'checkpoint_sha256':bundle.fingerprint},f));write_json(root/'problem.json',p.to_dict());write_json(root/'conditioning-problem.json',input_problem.to_dict());write_json(root/'observations.json',rows);report.update(regular_grid=queries is None,grid_shape=list(p.grid.shape) if queries is None else None);write_json(root/'metrics.json',report);return report


@lru_cache(maxsize=2)
def _bundle(path,mtime,size,device):return MeshBundle(path,device)


def decode_saved(directory,checkpoint,time_value,quantity='u'):
    root=Path(directory);p=Path(checkpoint);stat=p.stat();meta=read_json(root/'metrics.json');bundle=_bundle(str(p),stat.st_mtime_ns,stat.st_size,meta['device']);problem=ProblemSpec.from_dict(read_json(root/'problem.json'));saved=torch.load(root/'state.pt',map_location='cpu',weights_only=True)
    if saved['checkpoint_sha256']!=bundle.fingerprint:raise ValueError('Checkpoint changed after generation')
    if quantity not in get(problem.family).state:raise ValueError('Unsupported field quantity')
    q=grid_queries(GridSpec(problem.grid.shape,problem.grid.domain,(float(time_value),)))
    with torch.no_grad():a=bundle.decode(saved['state'].to(bundle.device),q,saved['families'].to(bundle.device),quantity)
    return {'time':float(time_value),'quantity':quantity,'frame':a.mean(0).cpu().reshape(problem.grid.shape).tolist(),'spread':a.std(0,unbiased=False).cpu().reshape(problem.grid.shape).tolist(),'fixed_physical_state':True,'device':str(bundle.device)}


def compute_job(directory,checkpoint):
    directory=Path(directory);request=read_json(directory/'request.json')
    def update(**kw):write_json(directory/'job.json',{**read_json(directory/'job.json'),**kw})
    try:
        update(status='running',stage='mesh_physical_reconstruction',progress=0.)
        if file_digest(checkpoint)[:32]!=request['model_id']:raise ValueError('Checkpoint changed after model selection')
        reconstruct(checkpoint,request['problem'],request['rows'],directory/'coordinate-result',request.get('queries'),**request['inference'],device=request['runtime']['device']);update(status='succeeded',stage='complete',progress=1.,finished_at=time.time())
    except Exception as error:
        import traceback
        (directory/'worker.log').write_text(traceback.format_exc());update(status='failed',stage='failed',error={'code':type(error).__name__,'message':str(error)},finished_at=time.time())
