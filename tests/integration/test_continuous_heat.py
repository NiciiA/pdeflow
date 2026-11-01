"""Real analytic, placement, continuous training/resume and local API checks."""
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from pdeflow.heat_continuous import FourierHeat
from pdeflow.heat_placement import place,evenly_spaced,features
from pdeflow.colab import preset,Session
from pdeflow.coordinate.temporal import points,sample_times,capability
from pdeflow.coordinate.dataset import Dataset,example,identity
from pdeflow.coordinate.training import train,read_checkpoint
from pdeflow.coordinate.contracts import CoordinateConfig,validate_rows,validate_queries,grid_queries
from pdeflow.coordinate.model import Coordinates,decode_chunks
from pdeflow.coordinate.inference import CoordinateBundle,decode_saved,reconstruct
from pdeflow.coordinate.optimization import model_contract
from pdeflow.runtime import read_json,write_json,file_digest
from pdeflow.workbench.playground import Heat,Measurements,reference,measurements,candidates
from pdeflow.workbench.api import create_app
from pdeflow.equations import heat_problem
from pdeflow.core import GridSpec,DomainSpec


def config():return preset('pipeline-check','cpu','fp32')


def test_arbitrary_time_modes_physical_domain_origin_endpoints_and_bounds():
    L=(2.,.75);o=(-1.,3.);x,y=np.meshgrid(o[0]+np.arange(32)*L[0]/32,o[1]+np.arange(24)*L[1]/24,indexing='ij')
    initial=2+.7*np.cos(2*np.pi*(x-o[0])/L[0])-.3*np.sin(4*np.pi*(y-o[1])/L[1])
    solver=FourierHeat(initial,L,o);rng=np.random.default_rng(1);q=np.c_[o+rng.random((100,2))*L,rng.uniform(0,1,100)];q[:2,2]=[0,1];D=.04
    expected=2+.7*np.cos(2*np.pi*(q[:,0]-o[0])/L[0])*np.exp(-D*(2*np.pi/L[0])**2*q[:,2])-.3*np.sin(4*np.pi*(q[:,1]-o[1])/L[1])*np.exp(-D*(4*np.pi/L[1])**2*q[:,2])
    np.testing.assert_allclose(solver.evaluate(q,D),expected,atol=2e-14)
    np.testing.assert_allclose(solver.frame(0,D),initial,atol=2e-14)
    np.testing.assert_allclose(FourierHeat(np.full((8,8),-3.)).evaluate([[.237,.1,1],[0,0,0]],.2),-3,atol=1e-14)
    np.testing.assert_allclose(FourierHeat(np.full((8,8),1e-14)).evaluate([[.2,.3,1]],.2),1e-14,rtol=1e-14,atol=0)
    for t in (-.001,1.001,np.nan):
        with pytest.raises(ValueError,match='Physical time'):solver.evaluate([[0,3.2,t]],D)
    assert solver.evaluate([[0,3.2,1+1e-9]],D)==pytest.approx(solver.evaluate([[0,3.2,1]],D))


def test_time_distributions_stable_context_identity_and_physical_derivatives(tmp_path):
    c=config();g=torch.Generator().manual_seed(2);t=sample_times(c,1000,g)
    assert int((t==0).sum())==250 and int((t==1).sum())==50
    assert len(t.unique())>650 and torch.any((t>0)&(t<.001))
    assert torch.equal(t,sample_times(c,1000,torch.Generator().manual_seed(2)))
    assert not torch.equal(points(c,32,g),points(c,32,g))
    a=example(c,'train',0);b=example(replace(c,animation_frames=75,times=(0.,.8,1.)),'train',0)
    assert identity(c)==identity(replace(c,animation_frames=75,times=(0.,.8,1.))) and model_contract(c)==model_contract(replace(c,animation_frames=75))
    assert np.array_equal(a['context'],b['context']) and np.array_equal(a['context_values'],b['context_values'])
    assert len(np.unique(a['context'][:,-1]))>10
    assert identity(c)!=identity(replace(c,time_interval=(0.,.8)))
    assert identity(c)!=identity(replace(c,time_sampling={'query':{'initial':.3,'early':.3}}))
    for policy in ({'early_min':0},{'query':{'initial':-.1}},{'encoder':{'uniform':0,'early':.6}}):
        with pytest.raises(ValueError):replace(c,time_sampling=policy).validate()
    # Coordinates normalize time, and autograd retains the physical chain rule.
    c2=replace(c,time_interval=(0.,2.));q=torch.tensor([[.2,.3,.237]],requires_grad=True);f=Coordinates(c2)(q)
    derivative=torch.autograd.grad(f[0,-3],q)[0]
    assert derivative[0,-1]==.5
    assert validate_queries(c,[[.2,.3,1+8e-8]])[0,-1]==1
    with pytest.raises(ValueError):validate_queries(c,[[.2,.3,1+2e-7]])
    legacy=CoordinateConfig().validate();assert not capability(legacy)['trained_continuous']
    with pytest.raises(ValueError,match='Legacy snapshot'):legacy.compatible(heat_problem(GridSpec((8,8),DomainSpec((1.,1.)),(0.,1.)),.07))


@pytest.mark.parametrize('n',[1,2,7,25,101,2000])
def test_evenly_spaced_exact_balanced_count_rectangular_no_endpoint_duplicates(n):
    L=(2.,.5);o=(-2.,3.);xy=evenly_spaced(n,L,o)
    assert xy.shape==(n,2) and len(set(map(tuple,xy)))==n
    assert np.all(xy>=o) and np.all(xy<np.asarray(o)+L)
    np.testing.assert_allclose(xy.mean(0),np.asarray(o)+np.asarray(L)/2,atol=1e-13)


def test_heat_focus_signed_sources_periodic_edges_background_and_bounded_rejection():
    x,y=np.meshgrid(np.arange(64)/64,np.arange(64)/64,indexing='ij')
    def bump(a,b):return np.exp(-(np.sin(np.pi*(x-a))**2+np.sin(np.pi*(y-b))**2)/.02)
    field=4+2*bump(.01,.2)-bump(.6,.7)
    centres,message=features(field,(1,1),(0,0));assert len(centres)>=2
    xy,m=place(25,'heat',9,field=field)
    assert m['background_count']==7 and sum(m['allocation'])==18 and min(m['allocation'])>=1
    assert m['reference_assisted'] and m['analysis_shape']==[64,64] and m['spread_fraction_min_length']>=.04
    assert np.min(np.linalg.norm(np.minimum(np.abs(np.asarray(centres)-[.01,.2]),1-np.abs(np.asarray(centres)-[.01,.2])),axis=1))<.05
    assert any(np.linalg.norm(np.asarray(c)-[.6,.7])<.05 for c in centres)
    assert np.array_equal(xy,place(25,'heat',9,field=field)[0])
    fallback,m=place(25,'heat',9,field=np.full((32,32),17.));assert m['background_count']==25 and 'Negligible' in m['message']
    _,m=place(2000,'heat',9,field=field,minimum_distance=.1);assert m['distance_relaxations']>0
    _,m=place(2,'heat',9,field=field);assert 'smaller' in m['message']
    uniform,_=place(2000,'random',4);assert np.max(np.abs(uniform.mean(0)-.5))<.03


def test_api_exact_time_placement_noise_multitime_metadata_and_manual(tmp_path):
    from pdeflow.workbench.playground import _features
    _features.cache_clear()
    root=tmp_path/'refs';root.mkdir();r=reference(Heat(preset='two'),root)
    base=dict(reference_id=r['id'],count=25,time=.237,seed=4)
    for policy in ('random','even','heat'):
        spec=Measurements(**base,placement=policy,pattern="current");candidate=candidates(spec,root);a=measurements(spec,root);b=measurements(Measurements(**base,placement=policy,noise=.1,pattern="current"),root)
        assert len(a['rows'])==25 and all(row['time']==.237 for row in a['rows'])
        assert [[row['x'],row['y']] for row in a['rows']]==candidate['positions']==[[row['x'],row['y']] for row in b['rows']]
        assert a['placement']['reference_assisted']==(policy=='heat') and a['placement']['design_time']==.237
        stored=read_json(root/r['id']/'samplings'/a['id']/'sampling.json');assert stored==a
        assert 'clean' not in json.dumps(a)
    assert _features.cache_info().hits>=1
    scattered=measurements(Measurements(**base,pattern='across'),root)
    assert len({p['time'] for p in scattered['rows']})==25 and all(0<=p['time']<=1 for p in scattered['rows'])
    fixed=measurements(Measurements(**base,placement='heat',pattern='fixed',times=[0,.237,1]),root)
    assert fixed['placement']['spatial_sensors']==9 and fixed['placement']['sampled_times']==3
    assert len({(p['x'],p['y']) for p in fixed['rows']})==9
    with pytest.raises(ValueError,match='Fixed sensors'):candidates(Measurements(**base,pattern='fixed',adaptive=True),root)
    adaptive=measurements(Measurements(**base,placement='heat',pattern='across',times=[0,.237,1],adaptive=True),root);assert len(adaptive['rows'])==25 and adaptive['placement']['time_adaptive'] and len(adaptive['placement']['time_designs'])==3
    manual=measurements(Measurements(**{**base,'count':2},pattern='current',placement='manual',positions=[[.1,.2],[.3,.4]]),root);assert len(manual['rows'])==2
    with pytest.raises(ValueError):measurements(Measurements(**base,pattern='current',placement='manual',positions=[[1,.5]]),root)
    c=config();validate_rows(c,manual['rows']);assert validate_rows(c,[{**manual['rows'][0],'time':.237}])[0][0,-1]==pytest.approx(.237)
    with pytest.raises(ValueError):validate_rows(c,[{**manual['rows'][0],'time':1.01}])
    with TestClient(create_app(workspace=tmp_path/'api')) as client:
        ref=client.post('/api/playground/reference',json={'end':1}).json()
        at=client.get(f'/api/playground/reference/{ref["id"]}/at?time=.237');assert at.status_code==200
        np.testing.assert_allclose(at.json()['frame'],at.json()['exact'],atol=1e-14)
        assert client.get(f'/api/playground/reference/{ref["id"]}/at?time=1.01').status_code==422


def test_continuous_training_exact_resume_stable_targets_export_and_query_order(tmp_path):
    c=config();full=tmp_path/'full';part=tmp_path/'part'
    train(c.to_dict(),full,'autoencoder');train(c.to_dict(),part,'autoencoder',stop_after=1);train(c.to_dict(),part,'autoencoder',resume=True)
    a=read_checkpoint(full/'autoencoder.pt')[0];b=read_checkpoint(part/'autoencoder.pt')[0]
    assert a['capabilities']['temporal']['trained_continuous'] and a['capabilities']['temporal']['interval']==[0.,1.]
    assert all(torch.equal(v,b['autoencoder'][k]) for k,v in a['autoencoder'].items())
    from pdeflow.coordinate.temporal_validation import representation
    from pdeflow.training_workflow import representation_quality
    validation=representation(full/'autoencoder.pt',tmp_path/'ae-report.json',subset=2)
    gate=representation_quality(full/'autoencoder.pt',validation)
    assert gate['validation_mse']==validation['validation_mse']
    assert gate['mse_ratio']==pytest.approx(validation['validation_mse']/validation['constant_baseline_mse'])
    assert abs(gate['validation_mse']-np.mean([r['mse_per_sample'][0] for r in validation['records']]))>1e-5
    before=file_digest(part/'resume.pt')
    with pytest.raises(ValueError,match='Incompatible model/data'):train(replace(c,time_interval=(0.,.8)).to_dict(),part,'autoencoder',resume=True)
    assert file_digest(part/'resume.pt')==before
    flow=tmp_path/'flow';train(c.to_dict(),flow,'flow',full/'autoencoder.pt')
    bundle=CoordinateBundle(flow/'coordinate.pt');p=heat_problem(GridSpec((8,8),DomainSpec((1.,1.)),(0.,1.)),.07)
    rows=[{'field':'u','x':.2,'y':.3,'time':.237,'value':.1,'sigma':.01}];q=np.array([[.2,.3,.237],[.8,.2,1.],[.1,.2,0],[.2,.3,.001]],np.float32)
    x,diag=bundle.sample(p,rows,q,steps=2,particles=2,seed=3,chunk_size=1);y,_=bundle.sample(p,rows,q[[2,0,3,1]],steps=2,particles=2,seed=3,chunk_size=3)
    np.testing.assert_allclose(x[:,[2,0,3,1]],y,atol=1e-7)
    reconstruct(flow/'coordinate.pt',p,rows,tmp_path/'result',steps=2,particles=2)
    first=decode_saved(tmp_path/'result',flow/'coordinate.pt',.237);last=decode_saved(tmp_path/'result',flow/'coordinate.pt',.237)
    assert first==last and first['fixed_latent']
    assert not diag.get('reference') and diag['no_reference_input']
    from pdeflow.coordinate.temporal_validation import flow as evaluate
    report=evaluate(flow/'coordinate.pt',tmp_path/'comparison.json',subset=1,counts=[5],noises=[0],placements=['random','even','heat'],seeds=[1,2],steps=1,particles=1)
    assert len(report['records'])==6 and all('uniform_space_time_mse' in r and len(r['time_strata_rmse'])==5 for r in report['records'])
    from pdeflow import training_workflow
    plan=training_workflow.plan({'model':'resolution-flexible','problem':'heat-2d','budget':'pipeline-check','device':'cpu'})
    assert plan['config']['temporal_mode']=='continuous' and plan['grid']['times'][-1]==1
