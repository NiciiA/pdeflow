"""Two explicit resumable stages. AE is frozen before latent CFM; no moving target."""
from pathlib import Path
from dataclasses import replace
import time, signal, os
import numpy as np
import torch
from pdeflow.runtime import RuntimeSpec,atomic_write,write_json,read_json,digest,file_digest,code_identity,environment
from pdeflow.local import run_lock,now
from .contracts import CoordinateConfig,grid_queries,padded_observations
from .model import Autoencoder,Velocity

SCHEMA='heat_latent_coordinate_v1'

def data_config(c):
    from pdeflow.local import preset_config
    from pdeflow.core import GridSpec,DomainSpec
    base=preset_config('heat-2d')
    times=tuple(np.linspace(*c.interval,c.animation_frames)) if c.temporal_mode=='continuous' else c.times
    return replace(base,grid=GridSpec(c.reference_shape,DomainSpec(c.lengths,c.origins),times),data=replace(base.data,seed=c.seed,diffusivity_range=c.diffusivity_range))

def prepare(c,root):
    from pdeflow.data.physical import specification,solve_spec
    preparation_started=time.perf_counter();reference_seconds=0.
    config=data_config(c);q=grid_queries(config.grid)
    values=[];parameters=[];ids=[];splits=[]
    for split,count in [('train',c.train_count),('validation',c.validation_count),('test',c.test_count)]:
        for i in range(count):
            spec,seconds=specification(config,split,i);sample=solve_spec(config,spec,seconds)
            reference_seconds+=sample.timings['reference_seconds']
            values.append(sample.values.reshape(-1).float().numpy());parameters.append(sample.problem.known_parameters['diffusivity']);ids.append(spec['id']);splits.append(split)
    atomic_write(root/'dataset.npz',lambda f:np.savez_compressed(f,values=np.asarray(values),diffusivity=np.asarray(parameters,dtype=np.float32),queries=q.numpy()))
    train=np.asarray(values[:c.train_count]);normalization={'mean':float(train.mean()),'scale':max(float(train.std()),1e-6),'fitted_ids':ids[:c.train_count]}
    record={'schema_version':1,'kind':'coordinate_training_data','ids':ids,'splits':splits,'normalization':normalization,'source':code_identity(),'distribution':__import__('pdeflow.data.physical',fromlist=['distribution']).distribution(config),'config_identity':digest(c.to_dict()),'sha256':file_digest(root/'dataset.npz'),'preparation_seconds':time.perf_counter()-preparation_started,'numerical_reference_seconds':reference_seconds,'reference_solves':len(ids),'meaning':'Physical realizations split before observation variants. One reference discretization per checkpoint. Equal per-problem mean point loss, not point-count weighting.'}
    write_json(root/'dataset.json',record)
    return record

def read_checkpoint(path,stage=None):
    try:cp=torch.load(path,map_location='cpu',weights_only=True)
    except (RuntimeError,EOFError,__import__('pickle').UnpicklingError) as e:raise ValueError('Incomplete/invalid coordinate checkpoint; use an atomic stage export') from e
    if cp.get('schema')!=SCHEMA or cp.get('architecture')!='set_latent_coordinate' or cp.get('stage') not in ['autoencoder','flow']:raise ValueError('Not a supported coordinate-stage checkpoint')
    if stage and cp['stage']!=stage:raise ValueError(f'Requires {stage} checkpoint; actual stage is {cp["stage"]}')
    c=CoordinateConfig(**cp['config']).validate()
    if cp.get('config_identity')!=digest(cp['config']):raise ValueError('Coordinate configuration identity mismatch')
    if not np.isfinite(cp['normalization']['scale']) or cp['normalization']['scale']<=0 or not set(cp['normalization']['fitted_ids'])<=set(cp['training_ids']):raise ValueError('Invalid training-only normalization provenance')
    if cp.get('autoencoder') is None and cp.get('dependency'):
        dependency=Path(path).parent/cp['dependency']['path']
        if file_digest(dependency)!=cp['dependency']['sha256']:raise ValueError('Frozen representation dependency checksum mismatch')
        cp['autoencoder']=read_checkpoint(dependency,'autoencoder')[0]['autoencoder']
    return cp,c

def select_context(q,values,c):
    # Fixed deterministic uniform subset for target encoding, independent of future query grids.
    indices=torch.linspace(0,len(q)-1,min(c.encoder_points,len(q))).long()
    return q[indices],values[:,indices]

def normalized_parameter(c,d):return (torch.log(d)-np.log(c.diffusivity_range[0]))/max(np.log(c.diffusivity_range[1]/c.diffusivity_range[0]),1e-6)

def training_observations(c,q,y,generator,normalization,device,validation=False,force_count=None):
    items=[]
    for values in y:
        n=int(torch.randint(c.observation_counts[0],c.observation_counts[1]+1,(),generator=generator))
        if float(torch.rand((),generator=generator))<c.unconditional_probability:n=0
        if force_count is not None:n=force_count
        n=min(n,len(q));indices=torch.randperm(len(q),generator=generator)[:n]
        sigma=float(torch.empty(()).uniform_(*c.noise_range,generator=generator))
        noisy=values[indices]+torch.randn(n,generator=generator)*sigma
        items.append((q[indices],noisy,torch.full((n,),sigma)))
    return padded_observations(items,normalization,device)

def train(config,output,stage,dependency=None,resume=False,stop_after=None,should_stop=None,event=None,extend=False):
    """Shared local/CLI/notebook trainer; no notebook-specific learning loop."""
    from .engine import train as execute
    path=Path(output)/'run.json';before=path.stat().st_mtime_ns if path.exists() else None
    try:return execute(config,output,stage,dependency,resume,stop_after,should_stop,event,extend)
    except (Exception,KeyboardInterrupt) as error:
        if path.exists() and path.stat().st_mtime_ns!=before:
            try:
                with run_lock(path.parent):
                    record=read_json(path)
                    if record.get('pid')==os.getpid() and record.get('state') not in ('completed','failed'):
                        record.update(state='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error),updated_at=now())
                        write_json(path,record)
            except ValueError:pass
        raise
