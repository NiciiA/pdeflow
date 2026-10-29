"""Independent expressions, quadrature and disclosure, without neural weights."""
from dataclasses import replace
import json
import numpy as np
import pytest
from pdeflow.reference import create,load,preview
from pdeflow.reference.examples import example
from pdeflow.reference.solvers import solve
from pdeflow.reference.fields import field_array
from pdeflow.reference.observations import sample,ObservationSamplingSpec
from pdeflow.observations import ObservationSet
from pdeflow.runtime import read_json
from pdeflow.evaluation.comparison import compare,align,EvaluationSpec

@pytest.mark.parametrize('dimension',[1,2,3])
def test_finite_fourier_exact_and_numeric(dimension):
    s=example(f'heat-{dimension}d');p,v,d=solve(s)
    coords=np.meshgrid(*[c.numpy() for c in p.grid.coordinates()],indexing='ij')
    expected=np.array([.2+.6*np.cos(2*np.pi*sum(coords))*np.exp(-.07*(2*np.pi)**2*dimension*t) for t in p.grid.times])
    np.testing.assert_allclose(v[0],expected,atol=3e-15)
    np.testing.assert_allclose(solve(replace(s,solver='numerical'))[1],v,atol=3e-15)
    assert 'finite-mode' in d['classification']
    with pytest.raises(ValueError,match='finite Fourier'):solve(replace(s,inputs={'initial_u':{'expression':'sin(2*pi*x)'}},solver='analytical'))

@pytest.mark.parametrize('source',["__import__('os').system('id')","x.__class__","[x for x in y]","sin(x, out=x)","x**100","open('secret')"])
def test_safe_field_parser(source):
    p,_,_=solve(example('heat-1d'))
    with pytest.raises(ValueError):field_array({'expression':source},p.grid)

@pytest.mark.parametrize('name',['darcy-2d','darcy-3d','chemistry-2d','chemistry-3d'])
def test_physical_references(name,tmp_path):
    r=create(example(name),tmp_path/'r');p,v,_=load(tmp_path/'r')
    assert r['diagnostics'] and np.isfinite(v).all()
    if name.startswith('darcy'):
        assert (v[:,0]>0).all() and abs(v[:,1].mean())<1e-12
        with pytest.raises(ValueError):solve(replace(example(name),inputs={'permeability':-1}))
    else:assert (v>=0).all()
    with pytest.raises(ValueError,match='analytical'):solve(replace(example(name),solver='analytical'))
    if p.grid.dimension==3:
        for a in range(3):
            got=preview(tmp_path/'r',axis=a,index=3)
            np.testing.assert_array_equal(got['values'],np.take(v[0,0,0],3,axis=a))

@pytest.mark.parametrize('pattern',['fixed_sensors','regular','random_spatial','space_time'])
def test_sampling_reproducible_noisy_exact_mapping(pattern,tmp_path):
    create(example('heat-2d'),tmp_path/'r');p,v,_=load(tmp_path/'r')
    spec=ObservationSamplingSpec(pattern=pattern,count=5,time_indices=(0,2),noise_std=.02)
    a=sample(tmp_path/'r',spec,tmp_path/'a');b=sample(tmp_path/'r',spec,tmp_path/'b')
    assert a['rows']==b['rows'] and a['count']==(5 if pattern=='space_time' else 10)
    obs=ObservationSet.from_dict(read_json(tmp_path/'a/observations.json'),p.grid)
    with np.load(tmp_path/'a/clean-samples-evaluation-only.npz') as clean:
        np.testing.assert_array_equal(clean['values'],obs.operator.apply(__import__('torch').from_numpy(v)).numpy())
        assert not np.array_equal(clean['values'],obs.values.numpy())
    inverse=read_json(tmp_path/'a/inverse-request.json')
    assert inverse['problem']['initial_condition']=='unknown' and not inverse['problem']['known_fields']
    assert 'initial_u' not in json.dumps(inverse) and 'clean' not in json.dumps(inverse)
    c=sample(tmp_path/'r',replace(spec,pattern='reuse',reuse_indices=tuple(map(tuple,obs.operator.indices.tolist()))),tmp_path/'c')
    assert a['rows']==c['rows']


def test_positions_and_darcy_disclosure(tmp_path):
    create(example('darcy-2d'),tmp_path/'r');p,_,_=load(tmp_path/'r')
    row={'field':'u','x':.0625,'y':.1875}
    s=ObservationSamplingSpec(pattern='positions',fields=('u',),positions=(row,),noise_std=0,disclosure={'a':'unknown','u':'partially_measured'})
    a=sample(tmp_path/'r',s,tmp_path/'a')
    assert a['count']==1 and set(a['problem']['known_fields'])=={'f'}
    with pytest.raises(ValueError):sample(tmp_path/'r',replace(s,positions=({**row,'x':.12},)),tmp_path/'b')
    with pytest.raises(ValueError,match='joint'):sample(tmp_path/'r',replace(s,disclosure={'a':'fully_known'}),tmp_path/'c')
    with pytest.raises(ValueError,match='fully known'):sample(tmp_path/'r',replace(s,disclosure={'f':'unknown'}),tmp_path/'d')


def test_weighted_metrics_quadrature_masks_and_zero_norm(tmp_path):
    create(example('heat-1d'),tmp_path/'r');p,v,_=load(tmp_path/'r')
    a=sample(tmp_path/'r',ObservationSamplingSpec(count=1,time_indices=(0,),noise_std=0),tmp_path/'a')
    obs=ObservationSet.from_dict(read_json(tmp_path/'a/observations.json'),p.grid)
    truth=np.ones_like(v)*2
    population=np.stack([truth[0]+1,truth[0]-1])[None]
    r=compare(population,p,obs,truth,weights=[.5,.5]);f=r['fields']['u']
    assert f['ensemble_mean']['full']['mse']==0 and f['weighted_mean_of_sample_errors']['full']['mse']==1
    assert f['ensemble_mean']['observed']['point_count']==1
    assert f['ensemble_mean']['unobserved']['point_count']==31
    assert f['per_time'][1]['observed']['rmse'] is None
    r=compare(population,p,obs,np.zeros_like(truth),weights=[.75,.25])['fields']['u']['ensemble_mean']['full']
    assert r['relative_l2'] is None and r['rmse']==2.5
    # Nonuniform physical-time quadrature: trapezoid integral, not frame average.
    grid=replace(p.grid,times=(0.,.01,.04,.15));q=replace(p,grid=grid)
    pop=np.zeros((1,1,1,4,8));pop[0,0,0,-1]=1
    rr=compare(pop,q,obs,np.zeros_like(v))['fields']['u']['ensemble_mean']['full']
    assert rr['mse']==pytest.approx((.15-.04)/2/.15)


def test_coordinate_transfer_and_shared_times(tmp_path):
    p,v,_=solve(example('heat-2d'));fine,w,_=solve(example('heat-2d',shape=[16,16]))
    a,_,meta=align(w[None],fine,p,EvaluationSpec('restrict_reference'))
    np.testing.assert_allclose(a,v,atol=3e-15)
    assert 'coincident' in meta['spatial_transfer']
    with pytest.raises(ValueError,match='matching coordinates'):align(w[None],fine,p,EvaluationSpec())
    with pytest.raises(ValueError,match='domain'):align(v[None],replace(p,grid=replace(p.grid,domain=replace(p.grid.domain,origins=(.1,0.)))),p,EvaluationSpec())
    other=replace(p,grid=replace(p.grid,times=(0.,.05,.15)))
    with pytest.raises(ValueError,match='Physical times'):align(v[:,[0,1,3]][None],other,p,EvaluationSpec())
    _,indices,_=align(v[:,[0,1,3]][None],other,p,EvaluationSpec(shared_times_only=True));assert indices==[0,1,3]
    d,dv,_=solve(example('darcy-2d'));df,dfv,_=solve(example('darcy-2d',shape=[16,16]))
    d=replace(d,known_fields={'f':np.asarray(df.known_fields['f']).reshape(8,2,8,2).mean((1,3)).tolist()})
    restricted,_,_=align(dfv[None],df,d,EvaluationSpec('restrict_reference'))
    np.testing.assert_allclose(restricted[0,0],dfv[0,0].reshape(8,2,8,2).mean((1,3)))


def test_numerical_heat_grid_convergence_and_bad_gauge(tmp_path):
    from pdeflow.observations import PointOperator
    import torch
    values=[]
    for n in [8,16,64]:
        s=example('heat-1d',shape=[n]);s=replace(s,inputs={'initial_u':{'expression':'exp(cos(2*pi*x))'}},solver='numerical')
        p,v,_=solve(s);values.append(v[0,-1,::n//8])
    assert np.linalg.norm(values[1]-values[2]) < np.linalg.norm(values[0]-values[2])/100
    p,v,_=solve(example('darcy-2d'))
    op=PointOperator(p.grid,torch.tensor([[1,0,0,0]]),2);obs=ObservationSet(op,torch.ones(1,1),torch.ones(1,1,dtype=torch.bool),.03)
    bad=v.copy();bad[1]+=1
    with pytest.raises(ValueError,match='zero-mean pressure gauge'):compare(v[None,None],p,obs,bad[None])


def test_reference_budget_precedes_spatial_expression_allocation():
    s=example('darcy-3d');s.request['domain']['shape']=[100000,100000,100000]
    with pytest.raises(ValueError,match='before evaluating inputs'):solve(s)
