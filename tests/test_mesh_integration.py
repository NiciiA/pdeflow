from dataclasses import replace
from pathlib import Path
import json
import pytest
import torch
from pdeflow.mesh.contracts import preset,CANDIDATES
from pdeflow.mesh.data import Data
from pdeflow.mesh.model import MeshModel
from pdeflow.mesh.objectives import loss,physical_loss,observation_loss
from pdeflow.mesh.training import draw,slice_noise,ownership
from pdeflow.mesh.session import MeshSession
from pdeflow.mesh.suite import Suite,suite_plan
from pdeflow.mesh.provenance import read_checkpoint
from pdeflow.mesh.evaluation import create_cases,compare
from pdeflow.runtime import read_json


def small(name='mesh_m2_v1',**kw):
    torch.set_num_threads(2)
    return preset('pipeline-check',name,'cpu','fp32',{'data':{'query_points':12},**kw})

@pytest.mark.parametrize('name',CANDIDATES)
def test_batched_loss_and_gradient_equal_accumulation(tmp_path,name):
    c=small(name);torch.manual_seed(71);model=MeshModel(c);data=Data(c,tmp_path/name);batch=data.prepare(0,fixed=True);noise,s,k,q=draw(c,2,torch.Generator().manual_seed(5),'cpu')
    assert set(batch.families.tolist())=={0,1}
    for stage in c.kind['stages']:
        ownership(model,stage);model.zero_grad(set_to_none=True);full,_=loss(model,batch,stage,data.norms,noise,s,k,q);full.mean().backward();expected={n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None};model.zero_grad(set_to_none=True)
        actual=[]
        for i in range(2):
            v,_=loss(model,batch.slice(i,i+1),stage,data.norms,slice_noise(noise,i,i+1),s[i:i+1],k[i:i+1],q[i:i+1]);(v.sum()/2).backward();actual.append(v.detach())
        torch.testing.assert_close(full.detach(),torch.cat(actual),atol=3e-6,rtol=3e-5)
        for n,p in model.named_parameters():
            if n in expected:torch.testing.assert_close(p.grad,expected[n],atol=8e-6,rtol=4e-4)


@pytest.mark.parametrize('settings',[{'type':'region_balanced_mse','alpha':.5,'region_queries':16,'analysis_resolution':16},{'type':'spectral_mse','spectral_weight':.05,'spectral_resolution':8}])
def test_region_spectral_and_frozen_endpoint_gradient(tmp_path,settings):
    c=small(field_loss=settings,objective={'observation':.001});d=Data(c,tmp_path/'region');b=d.prepare(0,fixed=True);m=MeshModel(c);noise,s,k,q=draw(c,2,torch.Generator().manual_seed(9),'cpu');v,parts=loss(m,b,'reconstruction',d.norms,noise,s,k,q);v.mean().backward();assert (settings['type']!='spectral_mse' or parts['spectral'].sum()>0) and m.conditioner.sensor.token[0].weight.grad.abs().sum()>0
    # Frozen physical modules must still differentiate their inputs at exact sensor times.
    m.decoder.requires_grad_(False);m.evolution.requires_grad_(False);state=b.target_state.clone().requires_grad_();obs=observation_loss(m,state,b,d.norms);obs.mean().backward();assert torch.isfinite(state.grad).all() and state.grad.abs().sum()>0
    changed=b.observations[0].clone();changed[...,-1]=0.;other=replace(b,observations=(changed,*b.observations[1:]));assert not torch.allclose(obs,observation_loss(m,state.detach(),other,d.norms))
    penalty,terms=physical_loss(m,state.detach().requires_grad_(),b,d.norms,q,True);penalty.mean().backward();assert torch.isfinite(penalty).all() and all(torch.isfinite(x).all() for x in terms.values())


def test_grouped_queries_and_time_derivative(tmp_path):
    c=small();d=Data(c,tmp_path);b=d.prepare(0,fixed=True);m=MeshModel(c);xy=torch.rand(40,2);q=torch.cat([xy,torch.full((40,1),.237)],-1)
    grouped=m.query(b.target_state,q,b.families);independent=m.query(b.target_state,q.clone().requires_grad_(),b.families);torch.testing.assert_close(grouped,independent,atol=3e-6,rtol=3e-5)
    q=q[:2].requires_grad_();u=m.query(b.target_state,q,b.families);grad=torch.autograd.grad(u.sum(),q)[0];h=1e-3;plus=q.detach().clone();minus=plus.clone();plus[:,-1]+=h;minus[:,-1]-=h;fd=(m.query(b.target_state,plus,b.families)-m.query(b.target_state,minus,b.families)).sum((0,2))/(2*h);torch.testing.assert_close(grad[:,-1],fd,atol=.003,rtol=.01)


def test_suite_resume_remaining_and_verified_restore(tmp_path):
    plan=suite_plan('pipeline-check','cpu','fp32',candidates=['mesh_m1_v1','mesh_m2_v1'],overrides={'data':{'query_points':8}});s=Suite.new(plan,tmp_path/'local',tmp_path/'store');first=s.run(['mesh_m1_v1']);assert first['state']=='partial';p=s.root/'runs/mesh_m1_v1/stages/reconstruction/resume.pt';original=p.read_bytes();second=s.run();assert second['state']=='completed' and p.read_bytes()==original
    restored=Suite.from_store(tmp_path/'store',s.root.name,tmp_path/'restored');assert restored.run()['state']=='completed'


def test_warm_start_explicit_new_run_and_comparison(tmp_path):
    c=small('mesh_m2_v1',budgets={s:1 for s in small().budgets});parent=MeshSession(c,tmp_path/'M2');assert parent.run()['state']=='completed'
    cp=parent.root/'export/model.pt';new=replace(c,architecture='mesh_m3_fgn_v1',warm_start={'checkpoint':str(cp)});child=MeshSession(new,tmp_path/'M3');assert child.run()['state']=='completed';warm=read_json(child.root/'bootstrap/initialization.json');assert warm['copied_keys'] and warm['newly_initialized_keys'] and warm['parent_sha256']
    cases=tmp_path/'cases';create_cases(c,cases,count=2);report=compare({'M2':cp,'M3':child.root/'export/model.pt'},cases,tmp_path/'comparison',particles=2,steps=2);assert len(report['models'])==2
    for row in report['models']['M3']['records']:
        variants=row['variants'];assert variants['tuple_permuted']['maximum_change_from_correct']<2e-5
        assert variants['correct']['joint_quantity_space_time_score']['available']
    with pytest.raises(FileExistsError):create_cases(c,cases,count=2)
