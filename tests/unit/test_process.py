from dataclasses import replace
from types import SimpleNamespace
import math
import pytest
import torch
from pdeflow.core import DomainSpec,GridSpec,TransformBundle
from pdeflow.equations import heat_problem
from pdeflow.models import TimeChannelLayout
from pdeflow.observations import PointOperator,ObservationSet
from pdeflow.workbench.observations import parse_rows
from pdeflow.process import ProcessConfig,ProcessState
from pdeflow.process.diagnostics import report
from pdeflow.runtime import Config,InferenceConfig
from pdeflow.sampling import sample


def fixture():
 g=GridSpec((4,4),DomainSpec((2.,3.)),(0.,.1,.3))
 p=heat_problem(g,.07);ix=torch.tensor([[0,0,0,0],[0,1,1,2],[0,2,3,1]])
 obs=ObservationSet(PointOperator(g,ix),torch.tensor([[0.,1.,2.]]),torch.tensor([[True,False,True]]),.1,.01)
 return p,obs


def thresholds():return {'pde.u.rmse':1e6,'observation.u.standardized_rmse':1e6,'conservation.u.max_drift':1e6}


def bundle(g):
 class Zero(torch.nn.Module):
  def __init__(self):super().__init__();self.p=torch.nn.Parameter(torch.zeros(()),requires_grad=False)
  def forward(self,x,s,c):return x*0+self.p*0
 c=replace(Config.load('configs/experiments/online_heat_2d.json'),grid=g)
 return SimpleNamespace(config=c,model=Zero(),transform=TransformBundle((0.,),(1.,),('training-only',)),layout=TimeChannelLayout(1,g),metadata={'checkpoint_sha256':'constructed-zero-field-dynamics'})


def test_observation_reports_masks_order_noise_and_gradients():
 p,obs=fixture();x=torch.zeros(2,1,3,4,4,dtype=torch.float64,requires_grad=True)
 r=report(x,p,obs,thresholds=thresholds())
 assert r['observation']['count']==2 and r['observation']['missing_count']==1
 assert r['observation']['raw_residuals']==[[0.,None,-2.],[0.,None,-2.]]
 assert r['entries']['observation.u.rmse']['value']==pytest.approx(math.sqrt(2))
 e=obs.per_problem_energy(x[:1]).sum();grad,=torch.autograd.grad(e,x)
 d=torch.randn_like(x);eps=1e-6
 fd=(obs.per_problem_energy((x+eps*d)[:1])-obs.per_problem_energy((x-eps*d)[:1]))/(2*eps)
 assert float((grad*d).sum())==pytest.approx(float(fd),rel=1e-7)
 noiseless=replace(obs,noise_std=0.)
 assert report(x,p,noiseless)['observation']['noise_standardized_residuals']==[[None,None,None]]*2
 rows=[{'field':'u','time':.1,'x':.3,'y':0.,'value':1}]
 with pytest.raises(ValueError,match='on-grid'):parse_rows(p,rows)
 rows[0]['x']=0.;rows[0]['time']=.2
 with pytest.raises(ValueError,match='interpolation'):parse_rows(p,rows)
 with pytest.raises(ValueError,match='Duplicate'):PointOperator(p.grid,obs.operator.indices[[0,0]])


def test_report_exact_constant_and_perturbation_and_scaling():
 p,obs=fixture();x=torch.ones(1,1,3,4,4,dtype=torch.float64)
 obs=replace(obs,values=torch.ones_like(obs.values))
 t={k:1e-10 for k in thresholds()};r=report(x,p,obs,thresholds=t)
 assert r['criteria_met'] and r['entries']['pde.u.rmse']['value']==0
 x[:,:,1,1,1]+=1
 assert not report(x,p,obs,thresholds=t)['criteria_met']
 # Physical domain scales the Laplacian; no fictitious physical-time derivative.
 from pdeflow.numerics import fd_laplacian
 assert torch.allclose(fd_laplacian(x,p.grid)*4,fd_laplacian(x,replace(p.grid,domain=DomainSpec((1.,1.5)))) )


def test_policy_estimates_never_finish_hysteresis_and_stagnation():
 p,o=fixture();state=ProcessState(ProcessConfig(thresholds=thresholds(),patience=2))
 good=report(torch.zeros(1,1,3,4,4),p,o,thresholds=thresholds())
 assert state.intermediate(good,1,.2,'observation')['action']=='CONTINUE_WITH_CONFIGURED_GUIDANCE'
 assert state.intermediate(good,2,.4,'observation')['action']=='CONTINUE_SAMPLING'
 state.intermediate(good,3,.6,'observation');assert state.stagnant()
 bad={**good,'criteria_met':False};assert state.intermediate(bad,4,.8,'observation')['action']=='CONTINUE_WITH_CONFIGURED_GUIDANCE'
 invalid={**bad,'finite':False};assert state.intermediate(invalid,5,.9,'observation')['action']=='FAIL_INVALID_NUMERICS'


def test_process_determinism_criteria_final_budget_retry_and_unavailable():
 p,o=fixture();b=bundle(p.grid)
 inf=InferenceConfig(steps=3,particles=1,controller={'thresholds':thresholds(),'check_interval':1})
 a=sample(b,p,o,inf);z=sample(b,p,o,inf)
 assert a.stop_reason=='FINISH_CRITERIA_MET' and torch.equal(a.samples,z.samples)
 assert a.diagnostics['process']['actions']==z.diagnostics['process']['actions']
 assert a.diagnostics['final_verification']['label']=='final_field'
 assert a.inference_state.generation_time==1.
 bad=replace(inf,controller={'max_attempts':2,'thresholds':{k:0. for k in thresholds()},'request_resampling':True,'refine':True,'check_interval':1})
 r=sample(b,p,o,bad)
 actions=[e['action'] for e in r.diagnostics['process']['actions']]
 assert 'RETRY_WITH_DEFINED_POLICY' in actions and 'REFINE_UNAVAILABLE' in actions and 'RESAMPLE_UNAVAILABLE' in actions
 assert r.stop_reason=='FINISH_BUDGET_EXHAUSTED' and r.diagnostics['process']['consumed']['attempts']==2
 small=sample(b,p,o,replace(inf,controller={'max_network_evaluations':1}))
 assert small.stop_reason=='FINISH_BUDGET_EXHAUSTED' and small.diagnostics['process']['consumed']['network_evaluations']==1
 cancel=sample(b,p,o,inf,cancelled=lambda:True)
 assert cancel.stop_reason=='STOP_CANCELLED'
 with pytest.raises(ValueError):ProcessConfig(max_attempts=0).validate()
 with pytest.raises(ValueError):ProcessConfig(thresholds={'pde.u.rmse':float('nan')}).validate()


def test_numerical_failure_preserves_last_finite_state():
 p,o=fixture();b=bundle(p.grid)
 b.model.forward=lambda x,s,c:x*float('nan')
 r=sample(b,p,o,InferenceConfig(steps=2,particles=1,controller={}))
 assert r.stop_reason=='FAIL_INVALID_NUMERICS' and torch.isfinite(r.samples).all()


def test_guidance_zero_and_physics_directional_effect():
 from pdeflow.steering import ReconstructionEnergy,SplitGradientGuidance
 from pdeflow.equations import HeatPhysics
 p,o=fixture();b=bundle(p.grid)
 inf=InferenceConfig(steps=3,particles=1,observation_strength=0.,physics_strength=0.)
 assert torch.equal(sample(b,p,o,inf).samples,sample(b,p,o,inf,'observation_pde').samples)
 x=torch.randn(1,3,4,4,dtype=torch.float64)
 decode=lambda v:v[:,None]*2.3+.7
 energy=ReconstructionEnergy(o,HeatPhysics(p.grid,torch.tensor([.07])),decode,0.,1.)
 corrected,d=SplitGradientGuidance(energy,1e-7,1e10).correct(x,1.,1.,lambda x,s:x)
 assert energy(corrected).item()<energy(x).item() and d['gradient_norm'][0]>0
 # Physical affine normalization must be part of the chain rule.
 v=x.clone().requires_grad_(True);g,=torch.autograd.grad(energy(v).sum(),v)
 direction=torch.randn_like(v);eps=1e-6
 fd=(energy(v+eps*direction)-energy(v-eps*direction))/(2*eps)
 assert float((g*direction).sum())==pytest.approx(float(fd),rel=1e-6)


def test_true_refinement_request_and_diagnostic_transfer():
 from pdeflow.refinement import refine_request,restrict_nodes,prolong_nodes
 p,o=fixture();fine=bundle(replace(p.grid,shape=(8,8)))
 q,remap=refine_request(p,o,fine,InferenceConfig(),1000)
 assert q.grid.shape==(8,8) and torch.equal(remap.values,o.values) and remap.operator.coordinates()==o.operator.coordinates()
 x=torch.randn(1,1,3,4,4);assert torch.equal(restrict_nodes(prolong_nodes(x,(8,8))),x)
 inf=InferenceConfig(steps=2,particles=1,controller={'refine':True,'max_attempts':2,'thresholds':{k:0. for k in thresholds()}})
 r=sample(bundle(p.grid),p,o,inf,fine_bundle=fine)
 assert r.samples.shape[-2:]==(8,8) and r.diagnostics['process']['consumed']['refinement_levels']==1
 assert any(e['action']=='REFINE' and e['compatibility_rechecked'] for e in r.diagnostics['process']['actions'])
 with pytest.raises(ValueError,match='2x'):refine_request(p,o,bundle(p.grid),InferenceConfig(),1000)


def test_split_placement_once_per_euler_or_heun_interval():
 p,o=fixture();b=bundle(p.grid);results=[]
 for stepper in ['euler','heun']:
  inf=InferenceConfig(steps=3,particles=1,stepper=stepper,physics_strength=0.,correction_strength=.00001)
  r=sample(b,p,o,inf,'observation');results.append(r.samples)
  assert r.diagnostics['gradient_evaluations']==3
  assert r.diagnostics['model_evaluations']==(3 if stepper=='euler' else 6)+2
  assert [e['generation_time'] for e in r.diagnostics['guidance_trace']]==pytest.approx([1/3,2/3,1.])
 assert torch.equal(*results) # same zero velocity; no accidental Heun double correction


@pytest.mark.parametrize('limit',['max_sampling_steps','max_guidance_steps'])
def test_each_step_budget_is_enforced(limit):
 p,o=fixture();inf=InferenceConfig(steps=4,particles=1,controller={limit:1})
 r=sample(bundle(p.grid),p,o,inf,'observation')
 assert r.stop_reason=='FINISH_BUDGET_EXHAUSTED'
 key='sampling_steps' if limit=='max_sampling_steps' else 'guidance_steps'
 assert r.diagnostics['process']['consumed'][key]==1
 assert r.diagnostics['final_verification']['label']=='incomplete_generation_field'


def test_grid_population_and_deterministic_clock_limits(monkeypatch):
 from pdeflow.process import BudgetReached
 p,o=fixture();b=bundle(p.grid)
 for config in [{'max_grid_points':8},{'max_particles':1}]:
  with pytest.raises(ValueError,match='grid/particles'):sample(b,p,o,InferenceConfig(particles=2,controller=config))
 state=ProcessState(ProcessConfig(wall_seconds=1.),started=0.)
 monkeypatch.setattr('pdeflow.process.time.monotonic',lambda:2.)
 with pytest.raises(BudgetReached,match='wall_time'):state.budget('sampling_steps')


def test_refinement_costs_and_guidance_cover_every_attempt():
 p,o=fixture();b=bundle(p.grid);fine=bundle(replace(p.grid,shape=(8,8)))
 inf=InferenceConfig(steps=2,particles=1,controller={'refine':True,'max_attempts':2})
 r=sample(b,p,o,inf,'observation',fine_bundle=fine)
 costs=r.diagnostics['attempt_costs'];assert len(costs)==2
 assert r.diagnostics['model_evaluations']==sum(c['model_evaluations'] for c in costs)==r.diagnostics['process']['consumed']['network_evaluations']
 assert len(r.diagnostics['guidance_trace'])==r.diagnostics['gradient_evaluations']==4
 assert r.diagnostics['process_seconds_total']>=r.diagnostics['seconds']
 assert {e['attempt'] for e in r.diagnostics['guidance_trace']}=={1,2}
