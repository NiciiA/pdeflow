"""Separate FGN, straight-path FM and discrete VP-v/DDIM; physical time is absent."""
import math
import torch
from .state import active_mask,enforce


def fgn_noise(c,b,g,device,zero=False):
    if zero:return (torch.zeros(b,c.noise_dim,device=device),torch.zeros(b,c.nodes,c.local_noise_dim,device=device))
    return (torch.randn(b,c.noise_dim,generator=g).to(device),torch.randn(b,c.nodes,c.local_noise_dim,generator=g).to(device))


def flow_path(c,target,noise,s,parameter,known,families):
    mask=active_mask(c,known,families);state=enforce(c,(1-s[:,None,None])*noise+s[:,None,None]*target,parameter,known,families)
    return state,(target-noise)*mask


def flow_endpoint(state,prediction,s):return state.float()+(1-s[:,None,None].float())*prediction.float()


def diffusion_schedule(steps,device='cpu'):
    # Compute the immutable scalar schedule in CPU FP64, then transfer once.
    # MPS has no FP64 tensors; this is the same numerical schedule on all devices.
    grid=torch.arange(steps+1,device='cpu',dtype=torch.float64)/steps
    f=torch.cos((grid+.008)/1.008*math.pi/2).square();beta=(1-f[1:]/f[:-1]).clamp(0,.999);bar=torch.cat([torch.ones(1,device='cpu',dtype=torch.float64),(1-beta).cumprod(0)])
    return bar.sqrt().float().to(device),(1-bar).sqrt().float().to(device)


def diffusion_path(c,target,noise,k,parameter,known,families):
    alpha,sigma=diffusion_schedule(c.sampler['diffusion_steps'],target.device);a=alpha[k,None,None];b=sigma[k,None,None]
    current=enforce(c,a*target+b*noise,parameter,known,families);v=(a*noise-b*target)*active_mask(c,known,families)
    return current,v


def diffusion_endpoint(current,prediction,alpha,sigma):
    return alpha*current.float()-sigma*prediction.float(),sigma*current.float()+alpha*prediction.float()


def generate(model,context,parameter,known,families,g=None,noise=None,steps=None,solver=None):
    c=model.c;device=parameter.device;g=g or torch.Generator().manual_seed(c.seed+71)
    if model.state_head is not None:
        if c.architecture=='mesh_m3_fgn_v1':noise=fgn_noise(c,len(parameter),g,device) if noise is None else noise
        return model.initial(context,parameter,known,families,noise),{'network_evaluations':1,'stochastic':c.kind['generative'],'method':'FGN one-pass structured noise' if c.kind['generative'] else 'deterministic'}
    noise=torch.randn(len(parameter),c.nodes+1,2,generator=g).to(device) if noise is None else torch.as_tensor(noise,device=device)
    if noise.shape!=(len(parameter),c.nodes+1,2) or not torch.isfinite(noise).all():raise ValueError('Wrong/nonfinite base state noise')
    state=enforce(c,noise,parameter,known,families);steps=steps or c.sampler['steps']
    if not 1<=steps<=512:raise ValueError('Sampling steps must be1..512')
    if c.architecture=='mesh_m4_flow_v1':
        solver=solver or c.sampler['solver']
        if solver not in ('euler','heun'):raise ValueError('FM requires Euler or Heun')
        for i in range(steps):
            s=state.new_full((len(state),),i/steps);v=model.generative(state,s,context,known,families)
            if solver=='heun':
                trial=enforce(c,state+v/steps,parameter,known,families);v=(v+model.generative(trial,s+1/steps,context,known,families))/2
            state=enforce(c,state+v/steps,parameter,known,families)
        return state,{'network_evaluations':steps*(2 if solver=='heun' else 1),'method':'straight-path conditional FM','steps':steps,'solver':solver}
    if solver not in (None,'ddim'):raise ValueError('VP diffusion uses DDIM, not a flow solver')
    total=c.sampler['diffusion_steps']
    if steps>total:raise ValueError('DDIM sampling steps cannot exceed discrete forward schedule')
    alpha,sigma=diffusion_schedule(total,device);indices=torch.linspace(total,0,steps+1).round().long().tolist()
    for k,j in zip(indices,indices[1:]):
        v=model.generative(state,state.new_full((len(state),),k/total),context,known,families);clean,eps=diffusion_endpoint(state,v,alpha[k],sigma[k]);state=enforce(c,alpha[j]*clean+sigma[j]*eps,parameter,known,families)
    return state,{'network_evaluations':steps,'method':'VP-v conditional diffusion / deterministic DDIM','steps':steps,'eta':0.,'forward_steps':total,'indices':indices,'terminal_base':'Approximate Gaussian at capped-cosine terminal SNR'}
