"""One shared versioned three-stage operator trainer/export API."""
from pathlib import Path
import math
import torch
from pdeflow.runtime import file_digest,digest
from .contracts import CoordinateConfig
from .joint import ARCHITECTURE,schema

SCHEMA='operator_physical_state_v2'

def normalized_parameter(c,p,families=None):
    families=torch.zeros(len(p),dtype=torch.long,device=p.device) if families is None else families
    lo=p.new_tensor([c.diffusivity_range[0],c.wave_speed_range[0]])[families.long()];hi=p.new_tensor([c.diffusivity_range[1],c.wave_speed_range[1]])[families.long()]
    return torch.log(p/lo)/torch.log(hi/lo)

def data_config(c):
    from dataclasses import replace
    from pdeflow.local import preset_config
    from pdeflow.core import GridSpec,DomainSpec
    import numpy as np
    base=preset_config('heat-2d');return replace(base,grid=GridSpec(c.reference_shape,DomainSpec(c.lengths,c.origins),tuple(np.linspace(*c.interval,c.animation_frames))))

def read_checkpoint(path,stage=None):
    cp=torch.load(path,map_location='cpu',weights_only=True)
    if cp.get('schema')!=SCHEMA or cp.get('architecture')!=ARCHITECTURE:raise ValueError('Legacy architecture — incompatible with operator-aware physical-state v2. Existing weights are preserved; no trajectory-to-state conversion exists.')
    if stage and cp['stage']!=stage:raise ValueError('Requires '+stage+' checkpoint; actual '+cp['stage'])
    c=CoordinateConfig(**cp['config']).validate()
    if cp.get('config_identity')!=digest(cp['config']) or digest(cp.get('joint_schema'))!=digest(schema(c)):raise ValueError('Checkpoint configuration/schema identity mismatch')
    for family,ns in cp['normalizations'].items():
        if family not in c.family_weights or any(not math.isfinite(n['scale']) or n['scale']<=0 for n in ns):raise ValueError('Invalid quantity normalization')
    if not set(sum(cp['normalization_fitted_ids'].values(),[]))<=set(cp['training_ids']):raise ValueError('Normalization contains nontraining identities')
    if cp.get('autoencoder') is None and not cp.get('portable_inference'):
        dep=cp['dependency'];p=Path(path).parent/dep['path']
        if file_digest(p)!=dep['sha256']:raise ValueError('Frozen representation/evolution dependency changed')
        cp['autoencoder']=read_checkpoint(p)[0]['autoencoder']
    return cp,c

def portable_checkpoint(cp):
    out={k:v for k,v in cp.items() if k not in ('optimizer','scheduler','scaler','rng','torch_rng','cuda_rng')}
    state=cp.get('inference_state') or {k:v for k,v in cp['autoencoder'].items() if not k.startswith('encoder.')}
    fp=lambda x:{k:v.detach().cpu().float() if v.is_floating_point() else v.detach().cpu() for k,v in x.items()}
    out.update(autoencoder=None,inference_state=fp(state),velocity=fp(cp['velocity']),portable_inference=True)
    return out

def train(config,output,stage,dependency=None,resume=False,stop_after=None,should_stop=None,event=None,extend=False):
    from .engine import train as execute
    return execute(config,output,stage,dependency,resume,stop_after,should_stop,event,extend)
