"""Registered executable smooth physics. Graph descriptions never guess a solver."""
from dataclasses import dataclass
from pdeflow.core import ProblemSpec,FieldSpec
from .graph import canonical

@dataclass(frozen=True)
class Physics:
    family:str
    parameter:str
    state:tuple
    order:int
    required_initial:tuple
    invariants:tuple
    branch:str='dynamic'
    boundary:str='periodic'
    regularity:str='smooth finite periodic spectral fields'

PHYSICS={'heat':Physics('heat','diffusivity',('u',),1,('u0',),('mean_u','l2_energy_decay')),
         'wave':Physics('wave','wave_speed',('u','v'),2,('u0','v0'),('mean_v','linear_mean_u','energy'))}

def get(family):
    if family not in PHYSICS:raise ValueError('Unregistered physics: only smooth periodic dynamic heat/wave execute; static/weak branches are planned')
    return PHYSICS[family]

def problem(grid,family,parameter=None):
    p=get(family)
    if grid.dimension!=2 or grid.domain.boundary!='periodic':raise ValueError('Current operator specialist supports rectangular periodic2D only')
    return ProblemSpec(family,grid,tuple(FieldSpec(q) for q in p.state),{} if parameter is None else {p.parameter:float(parameter)},canonical(family).to_dict(),(p.parameter,) if parameter is None else (),initial_condition='unknown')

def family_index(family):return ('heat','wave').index(get(family).family)

def strong_residual(model,z,coords,parameter,known,families,norm):
    """Autograd THROUGH direct-time evolution and spatial decoder in physical units."""
    import torch
    with torch.enable_grad():
        coords=coords.detach().clone().requires_grad_(True)
        u=model.decode(z,coords,parameter,known,'u',families)*norm[0]['scale']+norm[0]['mean']
        grad=torch.autograd.grad(u.sum(),coords,create_graph=True)[0]
        lap=sum(torch.autograd.grad(grad[...,j].sum(),coords,create_graph=True,retain_graph=True)[0][...,j] for j in range(2))
        scalar=model.decode_parameter(z,parameter,known,True,families)
        if int(families[0])==0:return {'heat':grad[...,-1]-scalar[:,None]*lap}
        v=model.decode(z,coords,parameter,known,'v',families)*norm[1]['scale']+norm[1]['mean']
        dv=torch.autograd.grad(v.sum(),coords,create_graph=True)[0]
        return {'u_t_minus_v':grad[...,-1]-v,'v_t_minus_c2_lap_u':dv[...,-1]-scalar[:,None].square()*lap}


def constraint_spec(family):
    p=get(family)
    return {'exact':['periodic spatial coordinates','identity E(t,t)','static scalar and known authority'],
            'soft':['initial-state agreement','strong smooth first-order residuals','composition',*p.invariants],
            'optional_correction':'bounded active-latent observation/residual optimization; changes sample distribution',
            'weak_handler':False,'static_handler':False,'regularity':p.regularity}


def invariant_drift(u,v,times,lengths,parameter):
    """Periodic grid quadrature; spectral derivative of the sampled interpolant."""
    import numpy as np
    u=np.asarray(u);times=np.asarray(times);axes=(-2,-1);area=float(np.prod(lengths));mean=u.mean(axes)
    report={'times':times.tolist(),'mean_u':mean.tolist(),'method':'Uniform periodic grid quadrature; spectral spatial derivative of sampled prediction. Resolution-dependent diagnostic, not exact conservation.'}
    if v is None:
        energy=.5*area*np.mean(u*u,axis=axes)
        report.update(mean_drift=(mean-mean[...,0,None]).tolist(),energy=energy.tolist(),energy_increases=np.maximum(np.diff(energy,axis=-1),0).tolist())
    else:
        v=np.asarray(v);mv=v.mean(axes);k=np.meshgrid(*[2*np.pi*np.fft.fftfreq(n,d=L/n) for n,L in zip(u.shape[-2:],lengths)],indexing='ij');coef=np.fft.fftn(u,axes=axes)
        grad2=sum(np.fft.ifftn(1j*w*coef,axes=axes).real**2 for w in k)
        c=np.asarray(parameter)
        while c.ndim<u.ndim:c=c[...,None]
        energy=.5*area*np.mean(v*v+c*c*grad2,axis=axes);den=np.maximum(np.abs(energy[...,0,None]),1e-10)
        report.update(mean_v=mv.tolist(),mean_v_drift=(mv-mv[...,0,None]).tolist(),linear_mean_u_error=(mean-mean[...,0,None]-(times-times[0])*mv[...,0,None]).tolist(),energy=energy.tolist(),energy_relative_drift=((energy-energy[...,0,None])/den).tolist(),energy_absolute_drift=(energy-energy[...,0,None]).tolist())
    return report


def training_spec(c):
    """Resolved executable training families; feature-mixture names are separate."""
    total=sum(c.family_weights.values())
    equations={'heat':'∂u/∂t = D Δu','wave':'∂²u/∂t² = c² Δu; v = ∂u/∂t'}
    return {'family_weights':{f:float(c.family_weights[f]/total) for f in sorted(c.family_weights)},
            'state_schemas':{f:list(get(f).state) for f in sorted(c.family_weights)},
            'equations':{f:equations[f] for f in sorted(c.family_weights)},
            'parameter_ranges':{get(f).parameter:list(c.diffusivity_range if f=='heat' else c.wave_speed_range) for f in sorted(c.family_weights)},
            'feature_mixture_meaning':'Initial-field mixture types describe shapes within a PDE family; they are not PDE families.'}
