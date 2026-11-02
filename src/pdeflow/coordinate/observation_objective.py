"""Decoded straight-path endpoint likelihood; independent of latent FM and field losses.

All sensor likelihoods are evaluated in physical units. Uniform subsampling
estimates a per-case sensor mean (or sum with N/K correction), then cases are
equally averaged by the trainer. Zero-noise observations use a named finite
tolerance penalty, never a fictitious Gaussian with silently floored sigma.
"""
from dataclasses import dataclass, asdict
import math
import torch


@dataclass(frozen=True)
class ObservationObjective:
    beta: float = 0.
    minimum_s: float = .5
    maximum_s: float = 1.
    sensor_budget: int = 8
    query_chunk: int = 8
    selection: str = 'uniform'
    reduction: str = 'mean'
    noiseless_policy: str = 'tolerance'
    tolerance_absolute: float = .001
    tolerance_relative: float = .05  # times global training quantity scale
    discrepancy_absolute: float = 0.
    discrepancy_relative: float = 0.
    gaussian_constant: bool = False
    warmup_updates: int = 0

    def validate(self):
        for key in ('beta','minimum_s','maximum_s','tolerance_absolute','tolerance_relative','discrepancy_absolute','discrepancy_relative'):
            x=getattr(self,key)
            if type(x) not in (int,float) or not math.isfinite(x) or x<0:raise ValueError('Invalid observation auxiliary '+key)
        if not 0<=self.minimum_s<=self.maximum_s<=1:raise ValueError('Observation auxiliary requires 0<=minimum_s<=maximum_s<=1')
        for key in ('sensor_budget','query_chunk'):
            if type(getattr(self,key))!=int or not 1<=getattr(self,key)<=1024:raise ValueError('Bounded positive observation '+key+' required')
        if type(self.warmup_updates)!=int or self.warmup_updates<0 or type(self.gaussian_constant)!=bool:raise ValueError('Invalid observation warmup/constant')
        if self.selection not in ('uniform','all') or self.reduction not in ('mean','sum') or self.noiseless_policy not in ('tolerance','reject'):raise ValueError('Unsupported observation selection/reduction/noise policy; independent scalar noise only')
        if self.noiseless_policy=='tolerance' and self.tolerance_absolute+self.tolerance_relative<=0:raise ValueError('Noiseless penalty needs an explicitly positive tolerance')
        return self


def settings(raw=None):
    if raw is not None and (not isinstance(raw,dict) or set(raw)-ObservationObjective.__dataclass_fields__.keys()):raise ValueError('Unknown observation auxiliary configuration')
    return ObservationObjective(**(raw or {})).validate()


def clean_endpoint(state,velocity,s,mean,scale):
    """Standardized straight path -> physical latent, exactly one inverse transform."""
    return (state.float()+(1-s.float()[:,None,None])*velocity.float())*scale+mean


def decode_points(ae,z,coords,quantities,parameter,known,families,chunk=8):
    """Independent [B,Q,3] queries, batching across cases/times, bounded B*Q copies.

    No sensor is replaced by a display-grid location or time. No reference
    parameter enters unknown conditioning: ae.evolve decodes it from z.
    """
    b,n,_=coords.shape
    owner=torch.arange(b,device=z.device)[:,None].expand(-1,n).reshape(-1)
    q=coords.reshape(-1,3);ids=quantities.reshape(-1);pieces=[]
    graph=ae.graph_encoder(families)
    for start in range(0,len(q),chunk):
        ix=owner[start:start+chunk];points=q[start:start+chunk];k=ids[start:start+chunk]
        evolved=ae.evolve(z[ix],0.,points[:,-1],parameter[ix],known[ix],families[ix],graph[ix])
        # Both heads are batched. Invalid heat-v is rejected at the input boundary.
        u=ae.decode_state(evolved,points[:,:2,None].transpose(1,2),0,families[ix]).squeeze(1)
        v=ae.decode_state(evolved,points[:,:2,None].transpose(1,2),1,families[ix]).squeeze(1)
        pieces.append(torch.where(k.eq(0),u,v))
    return torch.cat(pieces).reshape(b,n)


def endpoint_loss(ae,c,state,velocity,s,mean,scale,parameter,known,families,observations,norms,successful_update,selection_seed):
    cfg=settings(c.joint.get('observation_auxiliary'));b=len(state)
    zero=velocity.float().sum((1,2))*0
    # Exact baseline: no decode, random draw, or reference evaluation.
    if cfg.beta==0:return zero,{'observation_auxiliary_beta':0.}
    points,features,valid,quantities=observations
    eligible=(s>=cfg.minimum_s)&(s<=cfg.maximum_s)
    # One bounded CPU selection pass; never one synchronization per sensor.
    masks=valid.detach().cpu();eligible_cpu=eligible.detach().cpu()
    # Metadata validation/selection is explicitly CPU on every backend. MPS
    # does not implement unique_dim; this never moves the differentiable loss.
    keys_cpu=torch.cat([points.detach(),quantities[...,None].to(points.dtype)],-1).cpu()
    selected=[];population=[]
    for i in range(b):
        g=torch.Generator().manual_seed(int(selection_seed)+i)
        ix=torch.where(masks[i])[0];population.append(len(ix))
        if not eligible_cpu[i]:ix=ix[:0]
        if cfg.selection=='all' and len(ix)>cfg.sensor_budget:raise ValueError('All-sensor auxiliary exceeds sensor_budget; increase it explicitly or use uniform selection')
        if cfg.selection=='uniform' and len(ix)>cfg.sensor_budget:ix=ix[torch.randperm(len(ix),generator=g)[:cfg.sensor_budget]]
        selected.append(ix)
    counts=[len(x) for x in selected];n=max(counts,default=0)
    beta=cfg.beta*(min(1.,(successful_update+1)/cfg.warmup_updates) if cfg.warmup_updates else 1.)
    coverage={'observation_eligible_examples':int((eligible_cpu&masks.any(1)).sum()),'observation_flow_s_min':s.detach().min(),'observation_flow_s_max':s.detach().max(),'observation_effective_beta_case_mean':beta*(eligible&masks.any(1).to(eligible.device)).float().mean()}
    if n==0:return zero,{'observation_auxiliary_beta':beta,'observation_selected':0,**coverage}
    ix=torch.zeros(b,n,dtype=torch.long,device=state.device);mask=torch.zeros(b,n,dtype=torch.bool,device=state.device)
    for i,row in enumerate(selected):
        if len(row):ix[i,:len(row)]=row.to(state.device);mask[i,:len(row)]=True
    def gather(x):return x.gather(1,ix[...,None].expand(-1,-1,x.shape[-1])) if x.ndim==3 else x.gather(1,ix)
    q=gather(points);f=gather(features);k=gather(quantities)
    q=torch.where(mask[...,None],q,torch.zeros_like(q));f=torch.where(mask[...,None],f,torch.zeros_like(f));k=torch.where(mask,k,torch.zeros_like(k))
    if not torch.isfinite(q).all() or not torch.isfinite(f).all() or ((k<0)|(k>1)|(k.eq(1)&families[:,None].eq(0))).any() or (f[...,1]<0).any():raise ValueError('Invalid observation auxiliary coordinates, quantity, value or noise')
    # External rows already reject duplicates. Validate packed training sensors
    # too; conflicting exact/noisy duplicate measurements are never reweighted.
    for i in range(b):
        keys=keys_cpu[i,selected[i]]
        if len(torch.unique(keys,dim=0))!=counts[i]:raise ValueError('Duplicate scalar observation coordinates/time in auxiliary')
    endpoint=clean_endpoint(state,velocity,s,mean,scale)
    prediction=decode_points(ae,endpoint,q,k,parameter,known,families,cfg.query_chunk).float()
    scales=prediction.new_tensor([[x['scale'] for x in norms.get(name,[])]+[1.]*(2-len(norms.get(name,[]))) for name in ('heat','wave')])[families.long()].gather(1,k)
    residual=(prediction-f[...,0])*scales  # mean cancels, physical sigma below
    sigma=f[...,1]*scales;positive=sigma>0
    if cfg.noiseless_policy=='reject' and (mask&~positive).any():raise ValueError('Zero-noise sensor requires explicit tolerance policy')
    variance=sigma.square()+(cfg.discrepancy_absolute+cfg.discrepancy_relative*scales).square()
    tolerance=cfg.tolerance_absolute+cfg.tolerance_relative*scales
    denominator=torch.where(positive,variance,tolerance.square())
    denominator=torch.where(mask,denominator,torch.ones_like(denominator))
    if (denominator<=0).any() or not torch.isfinite(denominator).all():raise ValueError('Observation variance underflow/nonfinite; select an explicit discrepancy policy')
    penalty=.5*residual.square()/denominator
    if cfg.gaussian_constant:penalty=penalty+torch.where(positive,.5*torch.log(2*math.pi*denominator),torch.zeros_like(penalty))
    penalty=torch.where(mask,penalty,torch.zeros_like(penalty))
    if not torch.isfinite(penalty).all():raise FloatingPointError('Nonfinite decoded observation objective; inspect noise and predicted physical fields')
    per=penalty.sum(1)/mask.sum(1).clamp_min(1)
    if cfg.reduction=='sum':per=per*per.new_tensor(population)
    beta=cfg.beta*(min(1.,(successful_update+1)/cfg.warmup_updates) if cfg.warmup_updates else 1.)
    info={**coverage,'observation_auxiliary_beta':beta,'observation_endpoint_loss':per.detach().mean(),'observation_selected':sum(counts),'observation_population':sum(population),'observation_gaussian_values':(mask&positive).sum().detach(),'observation_tolerance_values':(mask&~positive).sum().detach()}
    return beta*per,info
