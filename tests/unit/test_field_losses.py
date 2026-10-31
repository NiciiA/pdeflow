import numpy as np
import pytest
import torch
from pdeflow.field_losses import analyze,field_metrics,differentiable_loss,periodic_components,grouped_queries,settings,AnalysisCache
from pdeflow.coordinate.contracts import CoordinateConfig

CFG={'type':'region_balanced_mse','analysis_resolution':16,'minimum_area_fraction':.003,'absolute_threshold':.01}
def spots():
    a=np.zeros((16,16));a[2:4,2:4]=3;a[8:12,8:12]=3;return a

def test_constant_fallback_and_background():
    a=np.full((16,16),2.);r=field_metrics(a+1,a,[2,3],CFG)
    assert r['region_count']==0 and r['region_balanced_mse']==r['mse']==1
    b=spots();r=field_metrics(b+1,b,[1,1],CFG)
    assert r['region_count']==2 and r['background_mse']==1
    for alpha in (0.,1.):
        q=field_metrics(b+2,b,[1,1],{**CFG,'alpha':alpha})
        assert q['region_balanced_mse']==pytest.approx(q['mse'] if alpha==0 else q['mean_region_mse'])

def test_equal_regions_signed_topology_noise():
    a=spots();a[0,8]=1e-7;r=field_metrics(a+1,a,[1,1],CFG)
    assert r['region_count']==2 and [q['mse'] for q in r['regions']]==[1,1]
    assert r['regions'][0]['area_fraction']*4==r['regions'][1]['area_fraction']
    u=spots();u[8:12,8:12]*=-1
    assert {r['sign'] for r in analyze(u,[1,1],CFG)['regions']}=={'hot','cold'}
    mask=np.zeros((8,8),bool);mask[0,0]=mask[-1,0]=mask[0,-1]=mask[-1,-1]=True
    labels=periodic_components(mask);assert len(set(labels[mask]))==1
    # Diagonal alone is disconnected under declared four-neighbour topology.
    mask[:]=False;mask[0,0]=mask[1,1]=True
    assert len(set(periodic_components(mask)[mask]))==2

def test_detached_fp32_gradients_and_reference_only_weights():
    ref=torch.tensor(spots(),requires_grad=True);a=analyze(ref.detach().numpy(),[1,1],CFG)
    weights=torch.tensor(a['contribution_weights']/ref.numel()).reshape(1,-1).requires_grad_()
    pred=torch.ones(1,256,dtype=torch.bfloat16,requires_grad=True)
    loss=differentiable_loss(pred,ref.reshape(1,-1),weights)
    loss.backward();assert loss.dtype==torch.float32 and torch.isfinite(pred.grad).all()
    assert ref.grad is None and weights.grad is None
    assert np.array_equal(a['labels'],analyze(ref.detach().numpy(),[1,1],CFG)['labels'])

def test_normalization_offset_scale_and_near_constant():
    b=spots();p=b+1
    m=field_metrics(p,b,[2,3],CFG,2)
    n=field_metrics(p*5+7,b*5+7,[2,3],CFG,10)
    assert m['region_count']==n['region_count']
    assert n['region_balanced_mse']==pytest.approx(m['region_balanced_mse']*25)
    assert analyze(np.ones((16,16))+np.random.default_rng(1).normal(0,1e-9,(16,16)),[1,1],CFG)['region_count']==0
    assert field_metrics(np.ones((16,16)),np.zeros((16,16)),[1,1],CFG)['relative_l2'] is None

@pytest.mark.parametrize('key,value',[('alpha',-1),('alpha',1.1),('minimum_area_fraction',0),('analysis_resolution',1000),('type','bogus')])
def test_settings_fail_early(key,value):
    with pytest.raises(ValueError):settings({**CFG,key:value})

def test_grouped_estimator_all_regions_covered_per_example_time_and_cost():
    from pdeflow.heat_continuous import FourierHeat
    c=CoordinateConfig(family_weights={'heat':1.},temporal_mode='continuous',data_mode='procedural',reference_shape=(16,16),train_count=12,validation_count=6,test_count=6,query_points=128,field_loss={**CFG,'time_groups':2,'region_queries':64},observation_counts=(1,8)).validate()
    items=[{'solver':FourierHeat(spots(),(1,1),(0,0)),'id':'a','D':.0001}, {'solver':FourierHeat(np.ones((16,16)),(1,1),(0,0)),'id':'b','D':.0001}]
    q,t,w,stats=grouped_queries(c,items,torch.Generator().manual_seed(2),{'scale':1},AnalysisCache())
    assert q.shape==(2,192,3) and torch.allclose(w.sum(1),torch.ones(2),atol=1e-6)
    assert stats[1]['active_queries']==128 # no regions, padding has zero weight
    assert torch.count_nonzero(w[1])==128
    # Constant point error has exactly equal solution/time reductions regardless of region counts.
    assert differentiable_loss(t+2,t,w).item()==pytest.approx(4,rel=1e-6)
    small={**c.to_dict(),'field_loss':{**c.field_loss,'region_queries':1}}
    with pytest.raises(ValueError,match='No region was dropped'):grouped_queries(CoordinateConfig(**small),items[:1],torch.Generator().manual_seed(2),{'scale':1},AnalysisCache())

def test_contribution_quadrature_and_weighted_estimator():
    b=spots();p=b.copy();p[2:4,2:4]+=2;p[8:12,8:12]+=1
    r=field_metrics(p,b,[1,1],{**CFG,'alpha':1})
    assert r['region_balanced_mse']==pytest.approx((4+1)/2)
    assert np.mean((p-b)**2*r['analysis']['contribution_weights'])==pytest.approx(r['region_balanced_mse'])
    assert settings().type=='mse' and settings().estimator=='legacy_uniform_points'

@pytest.mark.parametrize('kind',['mse','weighted_mse','region_balanced_mse'])
def test_unbiased_nonuniform_supervision_estimator(kind):
    from pdeflow.heat_continuous import FourierHeat
    from dataclasses import replace
    c=CoordinateConfig(family_weights={'heat':1.},temporal_mode='continuous',data_mode='procedural',reference_shape=(16,16),train_count=12,validation_count=6,test_count=6,
       query_points=8192,field_loss={**CFG,'type':kind,'time_groups':1,'region_queries':8192},observation_counts=(1,8)).validate()
    item={'solver':FourierHeat(spots(),(1,1),(0,0)),'id':'a','D':.0001}
    q,t,w,stats=grouped_queries(c,[item],torch.Generator().manual_seed(12),{'scale':1},AnalysisCache())
    error2=1+.7*torch.cos(2*torch.pi*q[:,:,0])+.3*torch.cos(2*torch.pi*q[:,:,1])
    measured=float((error2*w).sum())
    a=analyze(item['solver'].frame(stats[0]['physical_times'][0],item['D'],c.interval,shape=(16,16)),[1,1],c.field_loss)
    x=np.arange(16)/16;cell_error=1+np.sinc(1/16)*(.7*np.cos(2*np.pi*x[:,None])+.3*np.cos(2*np.pi*x[None,:]))
    expected=1.
    if kind=='weighted_mse':expected=float(np.mean(cell_error*a['contrast_weights'])/a['contrast_weights'].mean())
    if kind=='region_balanced_mse' and a['region_count']:
        expected=.5+.5*np.mean([cell_error[a['labels']==r['label']].mean() for r in a['regions']])
    assert measured==pytest.approx(expected,abs=.025)


def test_coarse_evaluation_does_not_invent_or_drop_region_errors():
    # Reference region on analysis cells can disappear on an extremely coarse metric grid.
    a=spots();from pdeflow.field_losses import project
    coarse=project(a,(2,2));r=field_metrics(coarse,coarse,[1,1],CFG)
    if r['unrepresented_region_count']:assert r['region_balanced_mse'] is None


def test_small_large_region_gradients_equal_total_importance_and_background_gradient():
    a=spots();analysis=analyze(a,[1,1],{**CFG,'alpha':1})
    target=torch.tensor(a,dtype=torch.float32).reshape(1,-1)
    pred=(target+1).detach().requires_grad_()
    weights=torch.tensor(analysis['contribution_weights']/a.size,dtype=torch.float32).reshape(1,-1)
    differentiable_loss(pred,target,weights).backward()
    gradients=pred.grad.reshape(a.shape).numpy()
    totals=[gradients[analysis['labels']==r['label']].sum() for r in analysis['regions']]
    np.testing.assert_allclose(totals,[1,1],atol=1e-6)
    assert np.all(gradients[analysis['labels']==0]==0)
    blend=analyze(a,[1,1],CFG)['contribution_weights'];assert np.all(blend[analysis['labels']==0]>.0)


def test_detection_physical_area_consistency_and_periodic_smoothing():
    areas=[]
    for n in (32,64):
        x=np.arange(n)/n;u=np.exp(-(np.sin(np.pi*x[:,None])**2+np.sin(np.pi*x[None,:])**2)/.07)
        r=analyze(u,[2,.5],{**CFG,'analysis_resolution':n,'smoothing_fraction':.01})
        assert r['region_count']==1
        areas.append(r['regions'][0]['physical_area'])
    assert abs(areas[0]-areas[1])<.01

@pytest.mark.parametrize('value',[None,float('nan'),True,'0.5'])
def test_invalid_alpha_is_descriptive(value):
    with pytest.raises(ValueError,match='field loss'):settings({**CFG,'alpha':value})
