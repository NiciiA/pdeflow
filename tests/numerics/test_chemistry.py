import numpy as np
import pytest
import torch
from pdeflow.chemistry import reaction,solve,ConcentrationTransform,ChemistryPhysics,problem
from pdeflow.core import GridSpec,DomainSpec,Conditioning

PARAM={'D_A':.02,'D_B':.03,'D_C':.01,'k_f':1.,'k_r':.2}
def grid(n=8,dim=2,times=(0.,.05,.1,.15)):return GridSpec((n,)*dim,DomainSpec((1.,)*dim),times)
def initial(g):
 x=np.meshgrid(*[p.numpy() for p in g.coordinates()],indexing='ij')
 return np.stack([.2+.05*sum(np.sin(2*np.pi*z+i) for z in x) for i in range(3)])

def rk4_reaction(state,kf,kr,time,steps=10000):
 x=state.copy();h=time/steps
 def rhs(z):
  r=kf*z[0]*z[1]-kr*z[2];return np.array([-r,-r,r])
 for _ in range(steps):
  a=rhs(x);b=rhs(x+h*a/2);c=rhs(x+h*b/2);d=rhs(x+h*c);x+=h*(a+2*b+2*c+d)/6
 return x

def test_exact_reaction_against_independent_ode_and_equilibrium():
 x=np.array([.8,.3,.1]);actual=reaction(x,3.,.4,.8);expected=rk4_reaction(x,3.,.4,.8)
 assert np.allclose(actual,expected,rtol=1e-11,atol=1e-12)
 equilibrium=np.array([.3,.2,.3]);assert np.allclose(reaction(equilibrium,1.,.2,20.),equilibrium,atol=1e-14)
 for rates in [(0,0),(0,10),(1000,0),(1000,500)]:
  y=reaction(x,*rates,100.);assert (y>=0).all();assert y[0]+y[2]==pytest.approx(x[0]+x[2]);assert y[1]+y[2]==pytest.approx(x[1]+x[2])

@pytest.mark.parametrize('dim',[2,3])
def test_positive_true_dimensional_reference_invariants_and_diffusion(dim):
 g=grid(dim=dim);x=initial(g);v,d=solve(x,PARAM,g,.005)
 assert v.shape==(3,4,*g.shape) and v.min()>=0 and d['clipped_values']==0
 assert d['max_stoichiometric_integral_drift']<2e-13 and abs(d['total_species_sum_change'])>1e-5
 # Constant fields with no reactions stay constant, including exact zeros.
 flat=np.stack([np.full(g.shape,z) for z in [0.,.4,0.]])
 z,diag=solve(flat,{**PARAM,'k_f':0.,'k_r':0.},g);assert np.max(np.abs(z-flat[:,None]))<1e-13
 errors=[]
 for n in (8,16,32):
  gg=grid(n,dim,(0.,.02,.04));xx=np.meshgrid(*[p.numpy() for p in gg.coordinates()],indexing='ij');mode=sum(np.cos(2*np.pi*z) for z in xx)
  physical=np.stack([1+.1*mode]*3);out,_=solve(physical,{**PARAM,'k_f':0.,'k_r':0.},gg,.02)
  exact=np.stack([1+.1*mode*np.exp(-PARAM[D]*(2*np.pi)**2*.04) for D in ('D_A','D_B','D_C')])
  errors.append(np.sqrt(np.mean((out[:,-1]-exact)**2)))
 assert errors[0]/errors[1]>3.8 and errors[1]/errors[2]>3.9


def test_strang_physical_time_refinement_and_gradient():
 g=grid(times=(0.,.08,.16));x=initial(g);truth,_=solve(x,{**PARAM,'k_f':8.,'k_r':2.},g,.0005)
 errors=[]
 for h in (.04,.02,.01):
  v,_=solve(x,{**PARAM,'k_f':8.,'k_r':2.},g,h);errors.append(np.linalg.norm(v-truth))
 assert errors[0]/errors[1]>3.7 and errors[1]/errors[2]>3.7
 physical=torch.from_numpy(truth)[None];transform=ConcentrationTransform.fit(physical,['calibration-only']);assert torch.allclose(transform.decode(transform.encode(physical)),physical,atol=1e-15)
 zero=torch.zeros_like(physical);assert torch.allclose(transform.decode(transform.encode(zero)),zero,atol=1e-30)
 state=(transform.encode(physical)+.03*torch.randn_like(physical)).requires_grad_();p=problem(g,PARAM);physics=ChemistryPhysics(g,Conditioning.from_problem(p,1))
 def energy(s):return physics.evaluate(transform.decode(s)).mean_square()
 grad,=torch.autograd.grad(energy(state),state);direction=torch.randn_like(state);direction/=direction.norm();eps=1e-6
 fd=(energy(state+eps*direction)-energy(state-eps*direction))/(2*eps);assert (grad*direction).sum().item()==pytest.approx(fd.item(),rel=2e-6,abs=1e-7)
 assert all(grad[:,i].abs().sum()>0 for i in range(3))


def test_reference_invalid_inputs_and_bounded_steps():
 g=grid();x=initial(g)
 with pytest.raises(ValueError):solve(-x,PARAM,g)
 with pytest.raises(ValueError):solve(x,{**PARAM,'k_f':-1},g)
 with pytest.raises(RuntimeError,match='step budget'):solve(x,PARAM,g,max_steps=1)
