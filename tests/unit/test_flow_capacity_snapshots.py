from dataclasses import replace
import numpy as np
import torch
import pytest
from pdeflow.generative import LinearCFM
from pdeflow.models import TimeChannelLayout,make_model,architecture_metadata
from pdeflow.core import Conditioning
from pdeflow.capacity import capacity
from pdeflow.local import preset_config
from pdeflow.sampling import Euler,Heun
from pdeflow.runtime import Config,RuntimeSpec,InferenceConfig,read_json
from pdeflow.training import load_inference

@pytest.mark.parametrize('dimension',[1,2,3])
def test_cfm_path_velocity_layout_and_all_element_reduction(dimension):
    u=torch.arange(2*3*4*4**dimension,dtype=torch.float64).reshape(2,3,4,*([4]*dimension))/50
    sg=torch.Generator().manual_seed(10);tg=torch.Generator().manual_seed(11)
    x,v,s=LinearCFM().training_state(u,sg,tg)
    z=torch.randn(u.shape,generator=torch.Generator().manual_seed(10),dtype=u.dtype);b=s.reshape(2,*([1]*(u.ndim-1)))
    assert torch.equal((1-0)*z+0*u,z) and torch.equal((1-1)*z+1*u,u)
    torch.testing.assert_close(x,(1-b)*z+b*u);torch.testing.assert_close(v,u-z)
    h=1e-5;torch.testing.assert_close((((1-b-h)*z+(b+h)*u)-((1-b+h)*z+(b-h)*u))/(2*h),v)
    assert v.shape==u.shape and s.shape==(2,)
    assert LinearCFM().loss(v+2,v)==4 # no observation masking or cross-field weighting


def test_analytic_gaussian_velocity_integration_order():
    mu=1.25;var=.65**2;z=torch.linspace(-2,2,100,dtype=torch.float64)[:,None]
    def velocity(x,s):return mu+((s*var-(1-s))/((1-s)**2+s*s*var))*(x-s*mu)
    errors={}
    for name,stepper in [('euler',Euler()),('heun',Heun())]:
        rows=[]
        for steps in [20,40,80]:
            x=z.clone()
            for j in range(steps):x=stepper.advance(velocity,x,j/steps,1/steps).state
            rows.append(float((x-(mu+var**.5*z)).norm()))
        errors[name]=rows
    assert errors['euler'][0]/errors['euler'][1]>1.8
    assert errors['heun'][0]/errors['heun'][1]>3.5 and errors['heun'][-1]<errors['euler'][-1]/10

@pytest.mark.parametrize('name',['small','medium','large'])
def test_capacity_native_metadata_and_real_gradient(name):
    c=preset_config('quickstart-heat-2d');c=replace(c,model=capacity(name),grid=replace(c.grid,shape=(16,16))).validate()
    layout=TimeChannelLayout(1,c.grid);model=make_model(c.model,layout);condition=Conditioning.from_problem(__import__('pdeflow.equations',fromlist=['heat_problem']).heat_problem(c.grid,.07),1,dtype=torch.float32)
    x=torch.randn(1,4,16,16,requires_grad=True);s=torch.tensor([.4]);y=model(x,s,condition);assert y.shape==x.shape
    opt=torch.optim.Adam(model.parameters(),lr=.001);before=next(model.parameters()).detach().clone();loss=(y-1).square().mean();loss.backward();assert torch.isfinite(x.grad).all();opt.step();assert not torch.equal(before,next(model.parameters()))
    meta=architecture_metadata(model,c.model,layout);assert meta['parameters']==sum(p.numel() for p in model.parameters()) and meta['resolution_policy'].startswith('exact')
    assert Config.from_dict(c.to_dict()).model==c.model
    with pytest.raises(ValueError,match='layout'):model(torch.zeros(1,4,32,32),s,condition)

@pytest.mark.parametrize('mode',['unguided','observation_pde','fk'])
def test_real_snapshots_are_passive_and_match_returned_native_state(mode):
    from pathlib import Path
    from pdeflow.core import ProblemSpec
    from pdeflow.observations import ObservationSet
    from pdeflow.sampling import sample
    root=Path('outputs/accepted_cpu_2d');bundle=load_inference(root/'training/inference.pt',RuntimeSpec())
    p=ProblemSpec.from_dict(read_json(root/'data/test_problem.json'));obs=ObservationSet.from_dict(read_json(root/'data/test_observations.json'),p.grid)
    cfg=InferenceConfig(steps=4,particles=4,fk={'chunk_size':2,'resample_every':1,'ess_fraction':1.,'observation_weight':.01} if mode=='fk' else None)
    a=sample(bundle,p,obs,cfg,mode);b=sample(bundle,p,obs,replace(cfg,snapshots={'max_frames':5,'field_indices':[0],'time_indices':[0]}),mode)
    torch.testing.assert_close(a.samples,b.samples,rtol=0,atol=0)
    if a.weights is not None:torch.testing.assert_close(a.weights,b.weights,rtol=0,atol=0)
    assert a.stop_reason==b.stop_reason
    frames=b.generation_snapshots;assert frames['metadata']['frames'][0]['generation_position']==0 and frames['metadata']['frames'][-1]['generation_position']==1
    np.testing.assert_allclose(frames['arrays']['current_physical'][-1,:,0,0],b.samples[0,:,0,0])
    np.testing.assert_allclose(frames['arrays']['endpoint_physical'][-1],frames['arrays']['current_physical'][-1])
    assert len(frames['metadata']['frames'])==5


def test_real_learned_gaussian_flow(tmp_path):
    from pdeflow.flow_verification import gaussian_toy
    r=gaussian_toy(tmp_path/'toy',updates=1000,particles=2048,steps=48)
    assert r['actual_optimizer_updates']==1000 and r['weights_changed']
    # Broad predeclared execution/learning bound, not exact moment or calibration claim.
    assert r['generation']['mean_error']<.2 and r['generation']['std_error']<.2


def test_sparse_fk_snapshots_compose_actual_parent_maps():
    from pathlib import Path
    from pdeflow.core import ProblemSpec
    from pdeflow.observations import ObservationSet
    from pdeflow.sampling import sample
    root=Path('outputs/accepted_cpu_2d');b=load_inference(root/'training/inference.pt',RuntimeSpec())
    p=ProblemSpec.from_dict(read_json(root/'data/test_problem.json'));o=ObservationSet.from_dict(read_json(root/'data/test_observations.json'),p.grid)
    r=sample(b,p,o,InferenceConfig(steps=6,particles=8,fk={'chunk_size':4,'resample_every':1,'ess_fraction':1.,'observation_weight':.01},snapshots={'max_frames':3,'max_particles':4}),'fk')
    composed=np.arange(8)
    for trace in r.diagnostics['fk']['trace'][:3]:
        if trace['parents'] is not None:composed=composed[trace['parents'][0]]
    assert r.generation_snapshots['metadata']['frames'][1]['step']==3
    assert composed[:4].tolist()==r.generation_snapshots['metadata']['frames'][1]['parents_at_previous_snapshot']
