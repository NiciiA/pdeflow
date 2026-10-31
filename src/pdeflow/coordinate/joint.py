"""Versioned physical schema, token layout and straight-path group reductions."""
from dataclasses import dataclass,asdict
from enum import IntEnum
import math
import torch

ARCHITECTURE='operator_aware_physical_latent_v2'
class Status(IntEnum):
    known=0
    partial=1
    unknown=2
    absent=3

@dataclass(frozen=True)
class Group:
    name:str
    start:int
    stop:int
    globals:int
    anchor_shape:tuple
    kind:str

def options(c):
    v=dict(encoder_depth=2,decoder_depth=2,observation_depth=1,observation_tokens=8,
           context_depth=1,local_neighbors=4,latent_regularization=1e-5,
           known_probability=.5,warmup_disclosure_coverage=True,group_weights={'initial_u':1.,'initial_v':1.,'coefficient':1.},
           target_cache=True,initial_field_loss=None,auxiliary={'field':0.,'observation':0.,'pde':0.,'minimum_s':.5},
           decoder_chunk=1024)
    if set(c.joint)-v.keys():raise ValueError('Unknown joint settings: '+str(set(c.joint)-v.keys()))
    v.update(c.joint)
    for k in ('encoder_depth','decoder_depth','observation_depth','observation_tokens','context_depth','local_neighbors','decoder_chunk'):
        if type(v[k])!=int or v[k]<1:raise ValueError(k+' must be a positive integer')
    if type(v['warmup_disclosure_coverage'])!=bool or type(v['target_cache'])!=bool:raise ValueError('Disclosure coverage/cache switches must be booleans')
    if not 0<=v['known_probability']<=1 or not math.isfinite(v['latent_regularization']) or v['latent_regularization']<0:raise ValueError('Invalid disclosure/regularization policy')
    if set(v['group_weights'])!={'initial_u','initial_v','coefficient'} or any(not math.isfinite(x) or x<=0 for x in v['group_weights'].values()):raise ValueError('Positive explicit weights required for all joint groups')
    if v['initial_field_loss'] is not None:
        from pdeflow.field_losses import settings
        settings(v['initial_field_loss'])
    a=v['auxiliary']
    if set(a)!={'field','observation','pde','minimum_s'} or any(not math.isfinite(x) or x<0 for x in a.values()) or a['minimum_s']>1:raise ValueError('Invalid optional decoded auxiliary settings')
    return v

def layout(c):
    if c.tokens<5:raise ValueError('Joint heat needs at least five total slots (two field groups and scalar)')
    scalar=2 if c.tokens<64 else 4 if c.tokens<256 else 8
    remaining=c.tokens-scalar;counts=[remaining//2,remaining-remaining//2,scalar];out=[];start=0
    for name,count in zip(('initial_u','initial_v','coefficient'),counts):
        side=max(1,int((count-1)**(1/c.dimension))) if name!='coefficient' else 0
        while side and side**c.dimension>=count:side-=1
        local=side**c.dimension if side else 0
        out.append(Group(name,start,start+count,count-local,(side,)*c.dimension if side else (), 'scalar' if name=='coefficient' else 'field'));start+=count
    return tuple(out)

def schema(c):
    return {'architecture':ARCHITECTURE,'groups':[asdict(g) for g in layout(c)],'statuses':{s.name:int(s) for s in Status},
            'coordinates':'physical x,y,t; periodic space; independent flow time s',
            'quantities':{'initial_u':'u(x,0)','initial_v':'v(x,0), absent for heat','coefficient':'typed D or c'},'state':{'heat':['u'],'wave':['u','v']},'graph_version':'typed_periodic_dynamic_graph_v1','evolution':'learned direct-time complete state; no trajectory latent',
            'forcing':'absent','boundary':'known periodic','domain':'box','equation':'registered heat(D)/wave(c²)'}

def active_mask(c,known,families=None):
    mask=torch.ones(len(known),c.tokens,dtype=torch.bool,device=known.device)
    if families is not None:
        g=layout(c)[1];mask[:,g.start:g.stop]=families[:,None].eq(1)
    g=layout(c)[-1];mask[:,g.start:g.stop]=~known[:,None]
    return mask

def flow_loss(c,prediction,target,known,families=None):
    per=[];weights=[]
    for g in layout(c):
        valid=(~known).float() if g.kind=='scalar' else torch.ones_like(known,dtype=torch.float32)
        if g.name=='initial_v' and families is not None:valid=valid*families.eq(1)
        w=options(c)['group_weights'][g.name]*valid
        per.append((prediction[:,g.start:g.stop].float()-target[:,g.start:g.stop].float()).square().mean((1,2))*w);weights.append(w)
    return (torch.stack(per).sum(0)/torch.stack(weights).sum(0)).mean()

def path(c,target,source,s,known,families=None):
    mask=active_mask(c,known,families)[...,None]
    source=source*mask;target=target*mask
    return (1-s[:,None,None])*source+s[:,None,None]*target,(target-source),mask
