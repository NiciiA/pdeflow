"""Explicit optional decoded consistency; no hidden reference enters correction."""
from dataclasses import dataclass
import torch,time,math
from .physics import strong_residual
from .joint import active_mask

@torch.enable_grad()
def terms(ae,c,z,p,known,f,obs,norms,pde=False):
    q,features,valid,quantity=obs;loss=z.new_zeros(len(z))
    for b in range(len(z)):
        for j,name in enumerate(('u','v')):
            ix=valid[b]&quantity[b].eq(j)
            if ix.any():
                pred=ae.decode(z[b:b+1],q[b,ix],p[b:b+1],known[b:b+1],name,f[b:b+1])[0];sigma=features[b,ix,1];den=torch.where(sigma.eq(0),torch.ones_like(sigma),sigma.clamp_min(1e-4));loss[b]+=((pred-features[b,ix,0])/den).square().sum()/valid[b].sum().clamp_min(1)
    physics=z.new_zeros(len(z))
    if pde:
        r=torch.linspace(.15,.85,4,device=z.device);coords=torch.stack([c.origins[0]+c.lengths[0]*r,c.origins[1]+c.lengths[1]*r,c.interval[1]*r],-1)
        for b in range(len(z)):
            residual=strong_residual(ae,z[b:b+1],coords,p[b:b+1],known[b:b+1],f[b:b+1],norms)
            scales=[norms[0]['scale']/c.interval[1]] if int(f[b])==0 else [norms[0]['scale']/c.interval[1],norms[1]['scale']/c.interval[1]]
            physics[b]=torch.stack([(value/scale).square().mean() for value,scale in zip(residual.values(),scales)]).mean()
    return loss,physics

def auxiliary_loss(ae,c,z,s,p,known,f,obs,norms,settings,item=None,g=None):
    if float(s[0])<settings['minimum_s']:return z.sum()*0
    observation,physics=terms(ae,c,z,p,known,f,obs,norms,settings['pde']>0);loss=settings['observation']*observation.mean()+settings['pde']*physics.mean()
    if settings['field']>0:
        from .dataset import state_samples
        q=obs[0][0];target=state_samples(item,q[:,:2].cpu().numpy(),.237,c,norms).to(z.device);q=q.clone();q[:,-1]=.237
        if len(q):loss+=settings['field']*torch.stack([(ae.decode(z,q,p,known,name,f)[0]-target[:,j]).square().mean() for j,name in enumerate(('u','v')[:1+int(f[0])])]).mean()
    return loss

@dataclass(frozen=True)
class Correction:
    steps:int=0
    learning_rate:float=.001
    observation:float=1.
    pde:float=0.
    def validate(self):
        if type(self.steps)!=int or not 0<=self.steps<=20 or not all(math.isfinite(x) for x in (self.learning_rate,self.observation,self.pde)) or self.learning_rate<=0 or min(self.observation,self.pde)<0:raise ValueError('Correction needs0..20 steps and finite valid weights')
        return self

def refine(ae,c,latent,p,known,f,obs,mean,scale,norms,spec):
    spec=spec.validate();started=time.perf_counter();z=latent.detach().clone().requires_grad_(True);history=[];mask=active_mask(c,known,f)[...,None]
    with torch.enable_grad():
        for _ in range(spec.steps):
            observation,physics=terms(ae,c,z*scale+mean,p,known,f,obs,norms,spec.pde>0);loss=(spec.observation*observation+spec.pde*physics).mean()
            grad=torch.autograd.grad(loss,z)[0]
            if not torch.isfinite(grad).all():raise FloatingPointError('Nonfinite consistency gradient')
            z=(z-spec.learning_rate*grad*mask).detach().requires_grad_(True);history.append(float(loss.detach()))
    return z.detach(),{'steps':spec.steps,'objective_history':history,'seconds':time.perf_counter()-started,'meaning':'Bounded active-latent optimization, not exact posterior sampling; uncorrected state retained'}
