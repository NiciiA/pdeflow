"""Opt-in A2 strong residuals and structured, phase-sensitive field supervision."""
from dataclasses import dataclass
import math
import numpy as np
import torch
from .model import derivative_attention
from .observation_objective import decode_points


@dataclass(frozen=True)
class PhysicsObjective:
    weight: float = 0.
    query_points: int = 4
    heat_weight: float = 1.
    wave_u_weight: float = 1.
    wave_v_weight: float = 1.


def physics_settings(raw=None):
    if raw is not None and (not isinstance(raw,dict) or set(raw)-PhysicsObjective.__dataclass_fields__.keys()):raise ValueError('Unknown A2 physics setting')
    cfg=PhysicsObjective(**(raw or {}))
    if any(type(x) not in (int,float) or not math.isfinite(x) or x<0 for x in (cfg.weight,cfg.heat_weight,cfg.wave_u_weight,cfg.wave_v_weight)) or type(cfg.query_points)!=int or not 1<=cfg.query_points<=128:raise ValueError('Invalid A2 physics weights/query budget')
    return cfg


def decoded_residuals(ae,z,coords,p,known,f,norms):
    """Per-case independent [B,Q,3] coordinates. Derivatives through actual T/D.

    FP32 math attention avoids fused SDPA's unsupported higher derivatives.
    Normalizations use training-only, global per-family quantity scales.
    """
    with torch.enable_grad(),torch.autocast(z.device.type,enabled=False),derivative_attention():
        q=coords.detach().float().clone().requires_grad_(True);z=z.float();p=p.float()
        ns=[norms[('heat','wave')[int(i)]] for i in f.detach().cpu()]
        us=q.new_tensor([n[0]['scale'] for n in ns])[:,None];um=q.new_tensor([n[0]['mean'] for n in ns])[:,None]
        vs=q.new_tensor([n[1]['scale'] if len(n)>1 else n[0]['scale']/ae.c.interval[1] for n in ns])[:,None];vm=q.new_tensor([n[1]['mean'] if len(n)>1 else 0. for n in ns])[:,None]
        k=torch.zeros(q.shape[:2],dtype=torch.long,device=q.device)
        u=decode_points(ae,z,q,k,p,known,f,chunk=8)*us+um
        grad=torch.autograd.grad(u.sum(),q,create_graph=True)[0]
        lap=sum(torch.autograd.grad(grad[...,j].sum(),q,create_graph=True,retain_graph=True)[0][...,j] for j in range(2))
        scalar=ae.decode_parameter(z,p,known,True,f)[:,None]
        # Heat v head is structurally zero, with no contribution to its loss.
        v=decode_points(ae,z,q,torch.ones_like(k),p,known,f,chunk=8)*vs+vm
        dv=torch.autograd.grad(v.sum(),q,create_graph=True)[0]
        T=ae.c.interval[1]
        return {'heat':(grad[...,-1]-scalar*lap)/(us/T),
                'wave_u':(grad[...,-1]-v)/(us/T),
                'wave_v':(dv[...,-1]-scalar.square()*lap)/(vs/T)}


def physics_loss(c,ae,z0,p,known,f,norms,seed):
    cfg=physics_settings(c.evolution.get('physics'));zero=z0.float().sum((1,2))*0
    if cfg.weight==0:return zero,{}
    g=torch.Generator().manual_seed(int(seed));q=torch.rand(len(z0),cfg.query_points,3,generator=g)
    q[...,:2]=q[...,:2]*q.new_tensor(c.lengths)+q.new_tensor(c.origins);q[...,-1]*=c.interval[1]
    residuals=decoded_residuals(ae,z0,q.to(z0.device),p,known,f,norms)
    heat=residuals['heat'].square().mean(1)*cfg.heat_weight
    wu=residuals['wave_u'].square().mean(1)*cfg.wave_u_weight;wv=residuals['wave_v'].square().mean(1)*cfg.wave_v_weight
    info={}
    for name,val in residuals.items():
        mask=f.eq(0 if name=='heat' else 1)
        if mask.any():info['physics_'+name]=val[mask].detach().square().mean()
    return cfg.weight*torch.where(f.eq(0),heat,(wu+wv)/2),info


def phase_spectral_mse(prediction,reference,frequency_power=0.):
    """Complex DFT difference, forward normalization 1/N², all modes.

    Real separable DFT is equivalent to fft2(norm='forward'), preserves phase,
    and uses the same differentiable implementation on CPU/MPS/CUDA. No FFT of
    scattered supervision. FP32 reductions; bounded N<=64 in configuration.
    """
    if prediction.shape!=reference.shape or prediction.ndim<2 or prediction.shape[-1]!=prediction.shape[-2]:raise ValueError('Spectral supervision requires matching structured square periodic grids')
    n=prediction.shape[-1];a=(prediction.float()-reference.detach().float())
    angle=2*math.pi*torch.arange(n,device=a.device)[:,None]*torch.arange(n,device=a.device)[None,:]/n
    C=angle.cos()/n;S=angle.sin()/n
    real=C@a@C.T-S@a@S.T;imag=-(C@a@S.T+S@a@C.T)
    k=torch.fft.fftfreq(n,device=a.device)*n
    weight=(1+k[:,None].square()+k[None,:].square()).pow(frequency_power/2);weight=weight/weight.mean()
    return ((real.square()+imag.square())*weight).sum((-2,-1))


def structured_field_loss(c,ae,states,rows,f,norms,device):
    from pdeflow.field_losses import settings
    cfg=settings(c.field_loss)
    if cfg.type!='spectral_mse':return states.float().sum((1,2))*0,{}
    n=cfg.spectral_resolution
    axes=[torch.arange(n,dtype=torch.float32)/n*L+o for L,o in zip(c.lengths,c.origins)]
    xy=torch.stack(torch.meshgrid(*axes,indexing='ij'),-1).reshape(-1,2).to(device);losses=[[] for _ in rows]
    for j,name in enumerate(('u','v')):
        active=[i for i,r in enumerate(rows) if j==0 or r.item['family']=='wave']
        if not active:continue
        ix=torch.tensor(active,device=device);targets=[]
        for i in active:
            r=rows[i];norm=norms[r.item['family']][j]
            frame=r.item['solver'].frame(r.tb,r.item['parameter'],c.interval,shape=(n,n),quantity=name)
            targets.append((np.asarray(frame)-norm['mean'])/norm['scale'])
        target=torch.as_tensor(np.stack(targets),device=device,dtype=torch.float32)
        pred=ae.decode_state(states[ix],xy,j,f[ix]).reshape(-1,n,n)
        values=phase_spectral_mse(pred,target,cfg.spectral_frequency_power)
        for k,i in enumerate(active):losses[i].append(values[k])
    result=torch.stack([torch.stack(x).mean() for x in losses])
    return cfg.spectral_weight*result,{'spectral_field':result.detach().mean(),'spectral_additional_queries_per_quantity':n*n}
