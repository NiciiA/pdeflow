"""Versioned direct sensor attention, with bounded logits and smooth periodic bias."""
from typing import NamedTuple
import math
import torch
from torch import nn


def settings(raw=None):
    value={'version':'pooled_v1','every':2,'query_chunk':16,'sensor_chunk':128,'bias_width':32,**(raw or {})}
    if set(value)!={'version','every','query_chunk','sensor_chunk','bias_width'} or value['version'] not in ('pooled_v1','spatial_v1'):raise ValueError('Conditioning version must be pooled_v1 or spatial_v1')
    for name in ('every','query_chunk','sensor_chunk','bias_width'):
        if type(value[name])!=int or not 1<=value[name]<=1024:raise ValueError('Positive bounded conditioning '+name+' required')
    return value


class Sensors(NamedTuple):
    keys:torch.Tensor
    values:torch.Tensor
    coordinates:torch.Tensor
    valid:torch.Tensor
    quantities:torch.Tensor


def relative_features(c,anchors,local,roles,coordinates,quantities):
    """No fictional anchor for global/scalar slots. All attention remains global."""
    xy=(coordinates[...,:2]-coordinates.new_tensor(c.origins))/coordinates.new_tensor(c.lengths)
    angle=2*math.pi*(xy[:,None,:,:]-anchors[None,:,None,:])
    spatial=torch.cat([angle.sin(),angle.cos()],-1)*local[None,:,None,None]
    b,n,m,_=spatial.shape
    time=((coordinates[...,-1]-c.interval[0])/(c.interval[1]-c.interval[0]))[:,None,:,None].expand(b,n,m,1)
    q=torch.nn.functional.one_hot(quantities.long(),3).to(spatial.dtype)[:,None].expand(b,n,m,3)
    group=torch.nn.functional.one_hot(roles.long(),3).to(spatial.dtype)[None,:,None].expand(b,n,m,3)
    return torch.cat([spatial,time,q,group,local.to(spatial.dtype)[None,:,None,None].expand(b,n,m,1)],-1)


class SensorAttention(nn.Module):
    """Shared reader at configured flow blocks; exact streaming softmax, no dropping.

    Null sensor is prepended and valid, so every query has a finite denominator.
    Keys/values are prepared once per conditioning pass; query/logit buffers are
    bounded by query_chunk*sensor_chunk, independent of total sensor count.
    """
    def __init__(self,c):
        super().__init__();self.c=c;self.cfg=settings(c.joint.get('conditioning'));w=c.width;self.heads=c.heads
        self.q=nn.Linear(w,w);self.k=nn.Linear(w,w,bias=False);self.v=nn.Linear(w,w);self.out=nn.Linear(w,w)
        self.relative=nn.Sequential(nn.Linear(12,self.cfg['bias_width']),nn.SiLU(),nn.Linear(self.cfg['bias_width'],1))
        nn.init.zeros_(self.relative[-1].weight);nn.init.zeros_(self.relative[-1].bias)
        # Import only at construction: the physical model does not depend on this route.
        from .model import Geometry
        geometry=Geometry(c)
        self.register_buffer('anchors',geometry.anchors.clone());self.register_buffer('local',geometry.local.clone());self.register_buffer('roles',geometry.group_ids.clone())
    def prepare(self,tokens,valid,coordinates,quantities):
        b,n,w=tokens.shape;h=self.heads
        return Sensors(self.k(tokens).reshape(b,n,h,w//h).transpose(1,2),self.v(tokens).reshape(b,n,h,w//h).transpose(1,2),coordinates,valid,quantities)
    def forward(self,state,sensors):
        b,n,w=state.shape;h=self.heads;d=w//h;queries=self.q(state).reshape(b,n,h,d).transpose(1,2);pieces=[]
        for start in range(0,n,self.cfg['query_chunk']):
            end=min(n,start+self.cfg['query_chunk']);Q=queries[:,:,start:end].float();maximum=None;denominator=None;numerator=None
            for low in range(0,sensors.keys.shape[2],self.cfg['sensor_chunk']):
                high=min(sensors.keys.shape[2],low+self.cfg['sensor_chunk'])
                features=relative_features(self.c,self.anchors[start:end],self.local[start:end],self.roles[start:end],sensors.coordinates[:,low:high],sensors.quantities[:,low:high])
                bias=self.relative(features).squeeze(-1)
                # Null is not a physical observation: no relative bias is assigned.
                if low==0:bias=torch.cat([torch.zeros_like(bias[:,:,:1]),bias[:,:,1:]],-1)
                scores=Q@sensors.keys[:,:,low:high].float().transpose(-1,-2)/math.sqrt(d)+bias.float()[:,None]
                scores=scores.masked_fill(~sensors.valid[:,None,None,low:high],-torch.inf)
                block_max=scores.amax(-1,keepdim=True)
                if maximum is None:
                    maximum=block_max;weights=(scores-maximum).exp();denominator=weights.sum(-1,keepdim=True);numerator=weights@sensors.values[:,:,low:high].float()
                else:
                    new=torch.maximum(maximum,block_max);factor=(maximum-new).exp();weights=(scores-new).exp();denominator=denominator*factor+weights.sum(-1,keepdim=True);numerator=numerator*factor+weights@sensors.values[:,:,low:high].float();maximum=new
            pieces.append(numerator/denominator)
        result=torch.cat(pieces,2).transpose(1,2).reshape(b,n,w).to(state.dtype)
        return self.out(result)
