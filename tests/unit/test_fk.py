from dataclasses import replace
import math
import numpy as np
import pytest
import torch
from pdeflow.fk import FKConfig,InvalidPopulation,ParticleState,corrected_drift,gather,normalize,score_from_velocity,simulate,systematic,weight_summary
from pdeflow.fk.toys import gaussian_velocity,gaussian_energy,gaussian_target,mixture_velocity,mixture_target,moments


def test_score_identity_and_stochastic_drift_sign():
 x=torch.randn(3,8,2,dtype=torch.float64);mu=x.new_tensor([.7,-.4]);var=x.new_tensor([1.44,.64]);velocity=gaussian_velocity(mu,var)
 for s in [0.,.2,.9,.999]:
  v=velocity(x,s);q=(1-s)**2+s*s*var
  assert torch.allclose(score_from_velocity(x,s,v),-(x-s*mu)/q,atol=1e-11,rtol=1e-10)
  assert torch.allclose(corrected_drift(x,s,v,.3),v+.3*(1-s)**2*score_from_velocity(x,s,v))
 assert torch.equal(corrected_drift(x,1.,velocity(x,1.),.3),velocity(x,1.))
 with pytest.raises(ValueError):score_from_velocity(x,1.,x)


def test_log_weights_and_invalid_populations():
 logw,_=normalize(torch.tensor([[0.,-100000.,-100000.],[0.,0.,0.]],dtype=torch.float64))
 assert weight_summary(logw)['ess']==pytest.approx([1.,3.])
 huge,_=normalize(torch.full((1,4),-1e300,dtype=torch.float64))
 assert torch.allclose(huge.exp(),torch.full((1,4),.25,dtype=torch.float64))
 for row in [[float('nan'),0.],[float('inf'),0.],[-float('inf'),-float('inf')]]:
  with pytest.raises(InvalidPopulation):normalize(torch.tensor([row]))
 for bad in [float('nan'),float('inf')]:
  with pytest.raises(InvalidPopulation):simulate(lambda x,s:x*0,lambda x:{'energy':torch.full(x.shape[:2],bad)},(1,4,1),2,1)
 with pytest.raises(ValueError):FKConfig(ess_fraction=0.).validate()
 with pytest.raises(ValueError):FKConfig(stepper='heun').validate()


def test_systematic_statistics_isolation_and_joint_gather():
 w=torch.tensor([[.1,.2,.7],[1.,0.,0.]],dtype=torch.float64);log=w.log();g=[torch.Generator().manual_seed(4),torch.Generator().manual_seed(7)]
 counts=torch.zeros_like(w)
 for _ in range(2000):
  ix=systematic(log,g)
  for b in range(2):counts[b].scatter_add_(0,ix[b],torch.ones(3,dtype=torch.float64))
 assert torch.allclose(counts/6000,w,atol=.012)
 assert torch.equal(systematic(torch.full((1,8),-math.log(8)),[torch.Generator().manual_seed(9)]),torch.arange(8)[None])
 ix=torch.tensor([[2,2,0],[0,0,0]]);field=torch.arange(12).reshape(2,3,2).float();param=field[:,:,0].clone()
 state=ParticleState({'model':field,'parameter':param},log,torch.arange(6).reshape(2,3).double(),torch.arange(3).expand(2,3).clone())
 state.resample(ix)
 assert torch.equal(state.components['parameter'],state.components['model'][:,:,0])
 state.components['model'][0,0,0]=-99
 assert state.components['model'][0,1,0]!=-99 # no descendant alias
 assert torch.equal(state.ancestors,ix)


def test_telescoping_initial_terminal_factors_zero_energy_and_chunking():
 v=gaussian_velocity([.3],[.8]);e=gaussian_energy([1.],[.5]);cfg=FKConfig(beta_start=.3,resample_every=99,chunk_size=8)
 run=simulate(v,e,(1,24,1),8,9,cfg)
 total=torch.tensor(run.diagnostics['incremental_log_potentials'],dtype=torch.float64).sum(0)
 terminal=-e(run.state.components['model'])['observation']
 assert torch.allclose(total,terminal,atol=1e-12)
 importance=simulate(v,e,(1,24,1),8,9,cfg,method='importance')
 assert torch.equal(run.state.components['model'],importance.state.components['model'])
 assert torch.allclose(run.state.log_weights,importance.state.log_weights,atol=1e-12)
 zero=lambda x:{'zero':torch.zeros(x.shape[:2],dtype=x.dtype,device=x.device)}
 a=simulate(v,zero,(2,24,1),8,9,cfg);b=simulate(v,zero,(2,24,1),8,9,replace(cfg,chunk_size=3),method='base')
 assert torch.equal(a.state.components['model'],b.state.components['model'])
 assert torch.allclose(a.state.log_weights.exp(),torch.full((2,24),1/24,dtype=torch.float64))
 repeat=simulate(v,zero,(2,24,1),8,9,cfg)
 assert torch.equal(repeat.state.components['model'],a.state.components['model'])
 assert not torch.equal(a.state.components['model'][0],a.state.components['model'][1])
 events=[]
 thinned=simulate(v,zero,(1,8,1),8,9,replace(cfg,diagnostic_every=3),event=events.append)
 assert [t['step'] for t in events]==[3,6,8]
 assert len(thinned.state.history)==8 # diagnostic cadence cannot change the path


def test_gaussian_unweighted_base_and_tilt_multiple_seeds():
 target=gaussian_target([.7,-.4],[1.44,.64],[1.4,.3],[.25,.49]);errors=[]
 for seed in [11,23,37]:
  cfg=FKConfig(chunk_size=256,resample_every=4,ess_fraction=.7,record_particle_history=False,max_particle_evaluations=1048576)
  for method in ['base','fk']:
   r=simulate(gaussian_velocity([.7,-.4],[1.44,.64]),gaussian_energy([1.4,.3],[.25,.49]),(1,4096,2),128,seed,cfg,method=method)
   m,c=moments(r)
   expected=(np.array([.7,-.4]),np.diag([1.44,.64])) if method=='base' else target
   errors.append((float(np.linalg.norm(m[0].numpy()-expected[0])),float(np.linalg.norm(c[0].numpy()-expected[1]))))
 assert np.mean([e[0] for e in errors])<.06
 assert np.mean([e[1] for e in errors])<.10


def test_mixture_target_and_future_descendant_noise():
 cfg=FKConfig(chunk_size=256,resample_every=1,ess_fraction=.95,record_particle_history=False,max_particle_evaluations=1048576)
 r=simulate(mixture_velocity,gaussian_energy([.5,.2],[.9,.8]),(2,4096,2),96,37,cfg)
 m,c=moments(r);target=mixture_target()
 assert np.linalg.norm(m[0].numpy()-target[0])<.15
 assert np.linalg.norm(c[0].numpy()-target[1])<.2
 assert any(t['parents'] is None and any(t['resampled_problems']) for t in r.state.history)
 assert len(torch.unique(r.state.components['model'][0],dim=0))==4096 # independent future increments
 assert (m[0]-m[1]).norm()<.2


def test_incomplete_budget_and_cancel_are_not_terminal():
 cfg=FKConfig(max_particle_evaluations=4,chunk_size=4)
 r=simulate(lambda x,s:x*0,lambda x:{'e':x.square().sum(-1)},(1,4,1),4,7,cfg)
 assert not r.complete and r.stop_reason=='FINISH_BUDGET_EXHAUSTED' and r.state.coordinate<1
 r=simulate(lambda x,s:x*0,lambda x:{'e':x.square().sum(-1)},(1,4,1),4,7,cancelled=lambda:True)
 assert not r.complete and r.stop_reason=='STOP_CANCELLED'
