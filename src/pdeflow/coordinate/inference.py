"""Reference-free latent ODE and physical-coordinate decoding."""
from pathlib import Path
import time
import numpy as np
import torch
from pdeflow.runtime import RuntimeSpec,write_json,atomic_write,file_digest,process_peak_memory_bytes,synchronize,code_identity
from pdeflow.core import ProblemSpec
from .contracts import validate_queries,validate_rows,padded_observations,grid_queries
from .model import Autoencoder,Velocity,decode_chunks
from .training import read_checkpoint,normalized_parameter

class CoordinateBundle:
    def __init__(self,path,device='cpu',allow_partial=False):
        self.cp,self.config=read_checkpoint(path,'flow')
        if not (self.cp.get('completed') or self.cp.get('inference_ready') or allow_partial):raise ValueError('A completed or explicitly exported partial latent-flow state is required')
        self.runtime=RuntimeSpec(device=device,threads=2);self.device,dtype=self.runtime.resolve()
        self.ae=Autoencoder(self.config).to(self.device).eval().requires_grad_(False);self.velocity=Velocity(self.config).to(self.device).eval().requires_grad_(False)
        self.ae.load_state_dict(self.cp['autoencoder']);self.velocity.load_state_dict(self.cp['velocity'])
        self.mean=self.cp['latent_mean'].to(self.device);self.scale=self.cp['latent_scale'].to(self.device)
        self.fingerprint=file_digest(path)

    def sample(self,problem,rows,queries,steps=12,particles=2,seed=91,chunk_size=1024,progress=None):
        c=self.config;c.compatible(problem);q=validate_queries(c,queries)
        points,values,sigma=validate_rows(c,rows,problem)
        if type(steps)!=int or not 1<=steps<=512 or type(particles)!=int or not 1<=particles<=16 or type(seed)!=int or not 0<=seed<2**63 or len(q)*particles>16_777_216:raise ValueError('Bounded inference requires steps1..512, reconstructions1..16, valid seed, <=64MiB output')
        if not rows and not self.cp['capabilities']['unconditional_trained']:raise ValueError('This model was trained with measurements only; take at least one measurement')
        # A zero-observation prior is available only when trained explicitly.
        observations=padded_observations([(points,values,sigma)]*particles,self.cp['normalization'],self.device)
        p=normalized_parameter(c,torch.full((particles,),problem.known_parameters['diffusivity'],device=self.device))
        started=time.perf_counter()
        if self.device.type=='cuda':torch.cuda.reset_peak_memory_stats(self.device)
        rng=torch.Generator().manual_seed(seed);latent=torch.randn(particles,c.tokens,c.width,generator=rng).to(self.device);h=1/steps
        with torch.no_grad():
            for k in range(steps):
                s=torch.full((particles,),k/steps,device=self.device)
                first=self.velocity(latent,s,p,observations);second=self.velocity(latent+h*first,s+h,p,observations);latent=latent+h*(first+second)/2
                if progress:progress(k+1,steps)
            decoded_latent=latent*self.scale+self.mean
            self.decoded_latent=decoded_latent.detach().cpu();self.parameter=p.detach().cpu()
            normalized=decode_chunks(self.ae,decoded_latent,q.to(self.device),p,chunk_size)
            norm=self.cp['normalization'];samples=normalized*norm['scale']+norm['mean']
            sensors=decode_chunks(self.ae,decoded_latent,points.to(self.device),p,chunk_size)*norm['scale']+norm['mean'] if len(points) else None
        synchronize(self.device)
        # Autodiff at physical coordinates, no reference. Finite diagnostic subset is explicit.
        probe=q[torch.linspace(0,len(q)-1,min(32,len(q))).long()].to(self.device).requires_grad_(True)
        with torch.enable_grad():
            u=self.ae.decode(decoded_latent[:1].detach(),probe,p[:1]).squeeze(0)*norm['scale']+norm['mean']
            grad=torch.autograd.grad(u.sum(),probe,create_graph=True)[0]
            lap=sum(torch.autograd.grad(grad[:,j].sum(),probe,retain_graph=True)[0][:,j] for j in range(c.dimension))
            residual=grad[:,-1]-problem.known_parameters['diffusivity']*lap
            physics=float(residual.square().mean().sqrt())
        elapsed=time.perf_counter()-started
        report={'field_loss_provenance':c.field_loss,'field_normalization_scale':norm['scale'],'temporal':__import__('pdeflow.coordinate.temporal',fromlist=['capability']).capability(c),'source':code_identity(),'checkpoint_source':self.cp['source'],'device':str(self.device),'dtype':'float32','architecture':'set_latent_coordinate','checkpoint_sha256':self.fingerprint,'stage':'flow','checkpoint_updates':self.cp['progress'],'autoencoder_dependency':self.cp['dependency'],'seed':seed,'reconstructions':particles,'flow_solver':{'name':'heun','steps':steps,'nfe':2*steps,'flow_interval':[0,1]},'physical_problem':problem.to_dict(),'observation_count':len(rows),'zero_observation_prior':not bool(rows),'no_reference_input':True,'query_count':len(q),'chunk_size':chunk_size,'seconds':elapsed,'timing_scope':'Latent integration, coordinate decode and bounded physical diagnostics; checkpoint load, export and provenance hashing excluded','parameter_count':sum(x.numel() for x in self.ae.parameters())+sum(x.numel() for x in self.velocity.parameters()),'physics':{'rmse':physics,'sample_index':0,'points':len(probe),'method':'autodiff physical-time derivative minus D times spatial Laplacian, evenly indexed query subset'},'observation_consistency':None if sensors is None else {'rmse_against_noisy_observed_values_per_sample':((sensors.cpu()-values)**2).mean(1).sqrt().tolist(),'meaning':'Decoded at actual sensor coordinates/time; no clean values used','absolute_sigma':sigma.tolist(),'predictions_per_sample':sensors.cpu().tolist()},'memory':{'process_lifetime_peak_rss_bytes':process_peak_memory_bytes(),'cuda_peak_allocated_bytes':int(torch.cuda.max_memory_allocated(self.device)) if self.device.type=='cuda' else None,'mps_driver_bytes_after':int(torch.mps.driver_allocated_memory()) if self.device.type=='mps' else None,'method':'RSS is process-lifetime high-water mark, not incremental run cost; CUDA peak allocated reset before integration; MPS is point-in-time, not peak'},'spread':'empirical sample spread, not calibrated posterior uncertainty','capability_limit':'Query resolution/location flexibility within fixed trained physical contract; fine-scale accuracy unestablished'}
        array=samples.detach().cpu().numpy()
        if not np.isfinite(array).all():raise FloatingPointError('Nonfinite decoded reconstruction')
        return array,report

def reconstruct(checkpoint,problem,rows,output,queries=None,steps=12,particles=2,seed=91,chunk_size=1024,device='cpu',progress=None):
    p=ProblemSpec.from_dict(problem) if isinstance(problem,dict) else problem
    regular=queries is None;queries=grid_queries(p.grid) if regular else queries
    bundle=CoordinateBundle(checkpoint,device);samples,report=bundle.sample(p,rows,queries,steps,particles,seed,chunk_size,progress)
    directory=Path(output);directory.mkdir(parents=True,exist_ok=False)
    atomic_write(directory/'result.npz',lambda f:np.savez_compressed(f,samples=samples,queries=np.asarray(queries)))
    atomic_write(directory/'latent.pt',lambda f:torch.save({'latent':bundle.decoded_latent,'parameter':bundle.parameter,'checkpoint_sha256':bundle.fingerprint},f))
    write_json(directory/'problem.json',p.to_dict());write_json(directory/'observations.json',rows)
    report.update(regular_grid=regular,grid_shape=list(p.grid.shape) if regular else None)
    write_json(directory/'metrics.json',report)
    return report

def compute_job(directory,checkpoint):
    from pdeflow.runtime import read_json
    directory=Path(directory);request=read_json(directory/'request.json')
    def update(**kwargs):write_json(directory/'job.json',{**read_json(directory/'job.json'),**kwargs})
    try:
        update(status='running',stage='latent_flow',progress=0.)
        if file_digest(checkpoint)[:32]!=request['model_id']:raise ValueError('Coordinate checkpoint changed; refresh model selection')
        reconstruct(checkpoint,request['problem'],request['rows'],directory/'coordinate-result',request.get('queries'),**request['inference'],device=request['runtime']['device'],progress=lambda k,n:update(completed_steps=k,total_steps=n,progress=.9*k/n))
        update(status='succeeded',stage='complete',progress=1.,finished_at=time.time(),stop_reason='FINISH_CONFIGURED_FLOW')
    except Exception as error:
        import traceback
        (directory/'worker.log').write_text(traceback.format_exc());update(status='failed',stage='failed',error={'code':type(error).__name__,'message':str(error)},finished_at=time.time())


from functools import lru_cache

@lru_cache(maxsize=2)
def _decoder(path,mtime,size,device):
    cp,c=read_checkpoint(path,'flow');dev,_=RuntimeSpec(device=device).resolve();ae=Autoencoder(c).to(dev).eval().requires_grad_(False);ae.load_state_dict(cp['autoencoder'])
    return ae,c,cp['normalization'],file_digest(path),dev


def decode_saved(directory,checkpoint,time_value):
    """Decode the saved generated latent, never rerun flow when physical time moves."""
    from pdeflow.core import GridSpec
    from pdeflow.heat_continuous import bounded_time
    directory=Path(directory);path=Path(checkpoint);stat=path.stat()
    from pdeflow.runtime import read_json
    device=read_json(directory/'metrics.json')['device']
    ae,c,n,sha,dev=_decoder(str(path),stat.st_mtime_ns,stat.st_size,device)
    state=torch.load(directory/'latent.pt',map_location='cpu',weights_only=True)
    if sha!=state['checkpoint_sha256']:raise ValueError('Model changed since this trajectory was generated')
    t=float(bounded_time(time_value,c.interval));problem=ProblemSpec.from_dict(__import__('pdeflow.runtime',fromlist=['read_json']).read_json(directory/'problem.json'))
    q=grid_queries(GridSpec(problem.grid.shape,problem.grid.domain,(t,)))
    with torch.no_grad():v=decode_chunks(ae,state['latent'].to(dev),q.to(dev),state['parameter'].to(dev))*n['scale']+n['mean']
    return {'time':t,'frame':v.mean(0).cpu().reshape(problem.grid.shape).tolist(),'spread':v.std(0,unbiased=False).cpu().reshape(problem.grid.shape).tolist(),'device':str(dev),'fixed_latent':True,'temporal_mode':c.temporal_mode}
