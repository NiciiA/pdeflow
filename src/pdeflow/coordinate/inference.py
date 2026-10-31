"""Reference-free conditional physical-state generation and learned direct-time queries."""
from pathlib import Path
from functools import lru_cache
import time
import numpy as np
import torch
from pdeflow.runtime import RuntimeSpec,write_json,read_json,atomic_write,file_digest,synchronize,code_identity
from pdeflow.core import ProblemSpec,GridSpec
from .contracts import validate_queries,validate_rows,grid_queries
from .model import Autoencoder,Velocity,decode_chunks
from .training import read_checkpoint,normalized_parameter
from .joint import ARCHITECTURE,active_mask
from .physics import get,family_index,strong_residual,constraint_spec

class CoordinateBundle:
    def __init__(self,path,device='cpu',allow_partial=False):
        self.cp,self.config=read_checkpoint(path,'flow')
        if not(self.cp['completed'] or self.cp.get('inference_ready') or allow_partial):raise ValueError('Complete or explicitly export the conditional stage')
        self.device,_=RuntimeSpec(device=device,threads=2).resolve();self.ae=Autoencoder(self.config,training_encoder=not self.cp.get('portable_inference')).to(self.device).eval().requires_grad_(False)
        if self.cp.get('portable_inference'):self.ae.load_state_dict(self.cp['inference_state'],strict=True)
        else:self.ae.load_state_dict(self.cp['autoencoder'])
        self.velocity=Velocity(self.config).to(self.device).eval().requires_grad_(False);self.velocity.load_state_dict(self.cp['velocity']);self.mean=self.cp['latent_mean'].to(self.device);self.scale=self.cp['latent_scale'].to(self.device);self.fingerprint=file_digest(path)
    def decode(self,z,q,p,known,f,quantity):
        ns=self.cp['normalizations'][('heat','wave')[int(f[0])]][0 if quantity=='u' else 1]
        return decode_chunks(self.ae,z,q,p,getattr(self,'chunk_size',self.config.joint['decoder_chunk']),known,quantity,f)*ns['scale']+ns['mean']
    def sample(self,problem,rows,queries,steps=12,particles=2,seed=91,chunk_size=1024,progress=None,initial_noise=None,solver='heun',correction=None):
        c=self.config;c.compatible(problem);q=validate_queries(c,queries);points,values,sigma=validate_rows(c,rows,problem);phys=get(problem.family)
        if not 1<=steps<=512 or not 1<=particles<=16 or solver not in ('heun','euler'):raise ValueError('Bounded Euler/Heun sampling requires1..512 steps and1..16 samples')
        if not rows and not self.cp['capabilities']['unconditional_trained']:raise ValueError('Model requires measurements; its prior was not trained')
        f=torch.full((particles,),family_index(problem.family),device=self.device,dtype=torch.long);known=torch.full((particles,),phys.parameter in problem.known_parameters,dtype=torch.bool,device=self.device)
        p=normalized_parameter(c,torch.full((particles,),problem.known_parameters[phys.parameter],device=self.device),f) if known.all() else torch.zeros(particles,device=self.device)
        self.chunk_size=int(chunk_size)
        ns=self.cp['normalizations'][problem.family];ids=torch.tensor([0 if r['field']=='u' else 1 for r in rows],dtype=torch.long,device=self.device);n=max(1,len(rows));oq=torch.zeros(particles,n,3,device=self.device);features=torch.zeros_like(oq);valid=torch.zeros(particles,n,dtype=torch.bool,device=self.device);quantities=torch.zeros(particles,n,dtype=torch.long,device=self.device)
        if rows:
            oq[:,:len(rows)]=points.to(self.device);valid[:,:len(rows)]=True;quantities[:,:len(rows)]=ids
            for j in range(1+int(f[0])):
                ix=ids.eq(j);indices=torch.where(ix)[0];features[:,indices,0]=(values.to(self.device)[ix]-ns[j]['mean'])/ns[j]['scale'];features[:,indices,1]=sigma.to(self.device)[ix]/ns[j]['scale'];features[:,indices,2]=1
        observations=(oq,features,valid,quantities);started=time.perf_counter();rng=torch.Generator().manual_seed(seed);noise=torch.randn(particles,c.tokens,c.width,generator=rng) if initial_noise is None else torch.as_tensor(initial_noise)
        if noise.shape!=(particles,c.tokens,c.width) or not torch.isfinite(noise).all():raise ValueError('Explicit initial flow noise shape/values invalid')
        latent=noise.to(self.device)*active_mask(c,known,f)[...,None]
        with torch.no_grad():
            graph=self.ae.graph_encoder(f);conditioning=self.velocity.condition(p,observations,known,f,graph)
            for k in range(steps):
                s=torch.full((particles,),k/steps,device=self.device);v=self.velocity(latent,s,conditioning=conditioning)
                latent=latent+(v+self.velocity(latent+v/steps,s+1/steps,conditioning=conditioning))/(2*steps) if solver=='heun' else latent+v/steps
                if progress:progress(k+1,steps)
        uncorrected=latent.clone();correction_report=None
        if correction:
            from .consistency import refine,Correction
            latent,correction_report=refine(self.ae,c,latent,p,known,f,observations,self.mean,self.scale,ns,Correction(**correction))
        with torch.no_grad():
            z=latent*self.scale+self.mean;scalar=self.ae.decode_parameter(z,p,known,True,f);samples=self.decode(z,q.to(self.device),p,known,f,'u');self.velocity_samples=self.decode(z,q.to(self.device),p,known,f,'v').cpu().numpy() if problem.family=='wave' else None
            sensor=torch.zeros(particles,len(rows),device=self.device)
            for j,name in enumerate(phys.state):
                ix=ids.eq(j)
                if ix.any():sensor[:,ix]=self.decode(z,points.to(self.device)[ix],p,known,f,name)
        # Strong residuals include time derivatives through learned evolution.
        tick=time.perf_counter();probe=q[torch.linspace(0,len(q)-1,min(4,len(q))).long()].to(self.device);residuals=[]
        for b in range(particles):
            r=strong_residual(self.ae,z[b:b+1].detach(),probe,p[b:b+1],known[b:b+1],f[b:b+1],ns)
            residuals.append({key:float(v.detach().square().mean().sqrt()) for key,v in r.items()})
        from .physics import invariant_drift
        diagnostic_grid=GridSpec((8,8),problem.grid.domain,tuple(np.linspace(*c.interval,5)));dq=grid_queries(diagnostic_grid).to(self.device)
        with torch.no_grad():
            du=self.decode(z,dq,p,known,f,'u').cpu().numpy().reshape(particles,5,8,8)
            dv=self.decode(z,dq,p,known,f,'v').cpu().numpy().reshape(particles,5,8,8) if problem.family=='wave' else None
        invariants=invariant_drift(du,dv,diagnostic_grid.times,c.lengths,scalar.cpu().numpy())
        self.decoded_latent=z.detach().cpu();self.parameter=p.cpu();self.known=known.cpu();self.families=f.cpu();synchronize(self.device)
        report={'architecture':ARCHITECTURE,'stage':'flow','checkpoint_sha256':self.fingerprint,'checkpoint_updates':self.cp['progress'],'device':str(self.device),'dtype':'float32','source':code_identity(),'checkpoint_source':self.cp['source'],'physical_problem':problem.to_dict(),'pde_graph':problem.operator_spec,'inferred_parameters':{phys.parameter:([problem.known_parameters[phys.parameter]]*particles if known.all() else scalar.cpu().tolist())},'parameter_status':'known' if known.all() else 'unknown','seed':seed,'reconstructions':particles,'flow_solver':{'name':solver,'steps':steps,'nfe':steps*(2 if solver=='heun' else 1)},'seconds':time.perf_counter()-started,'constraint_seconds':time.perf_counter()-tick,'no_reference_input':True,'observation_count':len(rows),'query_count':len(q),'field_loss_provenance':c.field_loss,'field_normalization_scale':ns[0]['scale'],'normalizations':ns,'physics':{'invariants':invariants,'per_sample':residuals,'method':'Physical coordinate derivatives THROUGH learned evolution, using each sample own scalar;4 bounded query probes','constraints':constraint_spec(problem.family)},'observation_consistency':None if not rows else {'rmse_against_noisy_observed_values_per_sample':(sensor-values.to(self.device)).square().mean(1).sqrt().cpu().tolist(),'predictions_per_sample':sensor.cpu().tolist(),'quantities':[r['field'] for r in rows],'absolute_sigma':sigma.tolist()},'correction':correction_report,'spread':'empirical, not calibrated posterior uncertainty','capability_limit':'Two registered smooth PDEs; no novel-PDE/geometry/dimension or accuracy guarantee'}
        self.uncorrected_samples=None
        if correction:
            with torch.no_grad():self.uncorrected_samples=self.decode(uncorrected*self.scale+self.mean,q.to(self.device),p,known,f,'u').cpu().numpy()
        array=samples.cpu().numpy()
        if not np.isfinite(array).all():raise FloatingPointError('Nonfinite predicted field')
        return array,report
    def decode_initial(self,spatial_queries,chunk_size=1024,quantity='u'):
        q=torch.cat([torch.as_tensor(spatial_queries,dtype=torch.float32),torch.zeros(len(spatial_queries),1)],1).to(self.device);name='initial_u' if quantity=='u' else 'initial_v';ns=self.cp['normalizations'][('heat','wave')[int(self.families[0])]][0 if quantity=='u' else 1]
        with torch.no_grad():v=decode_chunks(self.ae,self.decoded_latent.to(self.device),q,self.parameter.to(self.device),chunk_size,self.known.to(self.device),name,self.families.to(self.device))
        return (v*ns['scale']+ns['mean']).cpu().numpy()

def reconstruct(checkpoint,problem,rows,output,queries=None,steps=12,particles=2,seed=91,chunk_size=1024,device='cpu',progress=None,solver='heun',correction=None,initial_noise=None):
    p=ProblemSpec.from_dict(problem) if isinstance(problem,dict) else problem;q=grid_queries(p.grid) if queries is None else queries;bundle=CoordinateBundle(checkpoint,device);samples,report=bundle.sample(p,rows,q,steps,particles,seed,chunk_size,progress,initial_noise,solver,correction);directory=Path(output);directory.mkdir(parents=True,exist_ok=False);xy=np.unique(np.asarray(q)[:,:2],axis=0)
    arrays={'samples':samples,'queries':np.asarray(q),'initial_queries':xy,'initial_samples':bundle.decode_initial(xy)}
    if p.family=='wave':arrays.update(velocity_samples=bundle.velocity_samples,initial_velocity_samples=bundle.decode_initial(xy,quantity='v'))
    if correction:arrays['uncorrected_samples']=bundle.uncorrected_samples
    atomic_write(directory/'result.npz',lambda f:np.savez_compressed(f,**arrays));atomic_write(directory/'latent.pt',lambda f:torch.save({'latent':bundle.decoded_latent,'parameter':bundle.parameter,'known':bundle.known,'families':bundle.families,'checkpoint_sha256':bundle.fingerprint},f));write_json(directory/'problem.json',p.to_dict());write_json(directory/'observations.json',rows);report.update(regular_grid=queries is None,grid_shape=list(p.grid.shape) if queries is None else None);write_json(directory/'metrics.json',report);return report

def compute_job(directory,checkpoint):
    directory=Path(directory);request=read_json(directory/'request.json')
    def update(**kw):write_json(directory/'job.json',{**read_json(directory/'job.json'),**kw})
    try:
        update(status='running',stage='physical_state_flow',progress=0.)
        if file_digest(checkpoint)[:32]!=request['model_id']:raise ValueError('Checkpoint changed; refresh selection')
        reconstruct(checkpoint,request['problem'],request['rows'],directory/'coordinate-result',request.get('queries'),**request['inference'],device=request['runtime']['device'],progress=lambda k,n:update(completed_steps=k,total_steps=n,progress=.9*k/n));update(status='succeeded',stage='complete',progress=1.,finished_at=time.time())
    except Exception as error:
        import traceback
        (directory/'worker.log').write_text(traceback.format_exc());update(status='failed',stage='failed',error={'code':type(error).__name__,'message':str(error)},finished_at=time.time())

@lru_cache(maxsize=2)
def _decoder(path,mtime,size,device):return CoordinateBundle(path,device)

def decode_saved(directory,checkpoint,time_value,quantity='u'):
    directory=Path(directory);path=Path(checkpoint);stat=path.stat();metrics=read_json(directory/'metrics.json');bundle=_decoder(str(path),stat.st_mtime_ns,stat.st_size,metrics['device']);p=ProblemSpec.from_dict(read_json(directory/'problem.json'));bundle.config.compatible(p)
    if quantity not in get(p.family).state:raise ValueError('Requested quantity absent from physical state')
    saved=torch.load(directory/'latent.pt',map_location='cpu',weights_only=True)
    if saved['checkpoint_sha256']!=bundle.fingerprint:raise ValueError('Checkpoint changed after generation')
    q=validate_queries(bundle.config,grid_queries(GridSpec(p.grid.shape,p.grid.domain,(float(time_value),))))
    with torch.no_grad():a=bundle.decode(saved['latent'].to(bundle.device),q.to(bundle.device),saved['parameter'].to(bundle.device),saved['known'].to(bundle.device),saved['families'].to(bundle.device),quantity)
    return {'time':float(time_value),'frame':a.mean(0).cpu().reshape(p.grid.shape).tolist(),'spread':a.std(0,unbiased=False).cpu().reshape(p.grid.shape).tolist(),'fixed_latent':True,'quantity':quantity,'device':str(bundle.device)}
