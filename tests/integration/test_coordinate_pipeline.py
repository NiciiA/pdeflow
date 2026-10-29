from pathlib import Path
from dataclasses import asdict,replace
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from pdeflow.coordinate.training import train,read_checkpoint
from pdeflow.coordinate.inference import CoordinateBundle,reconstruct
from pdeflow.coordinate.contracts import grid_queries
from pdeflow.coordinate.registry import runs
from pdeflow.core import GridSpec,DomainSpec
from pdeflow.equations import heat_problem
from pdeflow.reference import create,load
from pdeflow.reference.examples import example
from pdeflow.reference.imports import import_reference
from pdeflow.reference.observations import sample,ObservationSamplingSpec
from pdeflow.runtime import read_json,file_digest
from pdeflow.workbench.api import create_app
from pdeflow.local import run_lock

@pytest.fixture
def tiny():return {'dimension':2,'reference_shape':[4,4],'times':[0,.05,.1],'lengths':[1,1],'origins':[0,0],'width':8,'heads':2,'tokens':3,'layers':1,'ae_steps':2,'flow_steps':2,'train_count':3,'validation_count':1,'test_count':1,'encoder_points':24,'query_points':24,'checkpoint_every':1,'batch_size':1,'observation_counts':[0,8],'runtime':{'device':'cpu','threads':1}}

def test_real_two_stages_resume_freeze_export_queries_reference_isolation_and_registry(tmp_path,tiny):
    ae=tmp_path/'coordinate/ae';flow=tmp_path/'coordinate/flow'
    first=train(tiny,ae,'autoencoder',stop_after=1);assert first['state']=='interrupted'
    before_seconds=read_checkpoint(ae/'resume.pt')[0]['training_seconds'];assert before_seconds>0
    original=(ae/'run.json').read_bytes()
    with pytest.raises(FileExistsError):train(tiny,ae,'autoencoder')
    assert (ae/'run.json').read_bytes()==original
    train(tiny,ae,'autoencoder',resume=True);assert read_checkpoint(ae/'autoencoder.pt')[0]['training_seconds']>=before_seconds;assert (ae/'history.json').is_file();ae_hash=file_digest(ae/'autoencoder.pt')
    train(tiny,flow,'flow',ae/'autoencoder.pt',stop_after=1);train(tiny,flow,'flow',resume=True)
    assert file_digest(ae/'autoencoder.pt')==ae_hash
    cp,c=read_checkpoint(flow/'coordinate.pt','flow');assert set(cp['normalization']['fitted_ids']).isdisjoint(cp['split_ids'][3:])
    bundle=CoordinateBundle(flow/'coordinate.pt');g=GridSpec((6,6),DomainSpec((1.,1.)),(0.,.025,.1));p=heat_problem(g,.07)
    rows=[{'field':'u','x':.13,'y':.21,'time':.025,'value':.2,'sigma':.03}]
    q=grid_queries(g);x,r=bundle.sample(p,rows,q,steps=2,particles=2,chunk_size=7);y,_=bundle.sample(p,rows,q,steps=2,particles=2,chunk_size=99)
    np.testing.assert_allclose(x,y,atol=1e-6);assert r['no_reference_input'] and np.isfinite(r['physics']['rmse']);assert r['observation_count']==1
    fine=replace(p,grid=replace(g,shape=(8,8)));assert bundle.sample(fine,rows,grid_queries(fine.grid),steps=1,particles=1)[0].shape==(1,192)
    scattered=q[[2,0,6]].clone();z,_=bundle.sample(p,rows,scattered,steps=2,particles=2);np.testing.assert_allclose(z,x[:,[2,0,6]],atol=1e-6)
    reconstruct(flow/'coordinate.pt',p,rows,tmp_path/'result',steps=1,particles=1)
    assert read_json(tmp_path/'result/metrics.json')['no_reference_input']
    records=runs(tmp_path);assert len(records)==2 and all(r['available'] for r in records)
    with run_lock(flow):assert next(r for r in runs(tmp_path) if r['stage']=='flow')['active']
    app=create_app(workspace=tmp_path)
    with TestClient(app) as client:
        model=next(r for r in records if r['stage']=='flow')['model_id'];body={'model_id':model,'problem':p.to_dict(),'rows':rows,'particles':1,'steps':1}
        assert client.post('/api/coordinate/validate',json=body).status_code==200
        assert not client.post('/api/coordinate/validate',json=body).json()['temporal']['trained_continuous']
        rejected=client.post('/api/coordinate/validate',json={**body,'require_continuous':True})
        assert rejected.status_code==422 and 'snapshot' in rejected.text
        assert client.post('/api/coordinate/validate',json={**body,'reference':[[1]]}).status_code==422
        math=example('heat-2d',(4,4)).request;math['observation_operator']['kind']='coordinate_point_set';math['measurements']['rows']=rows
        assert client.post('/api/coordinate/validate',json={**body,'mathematical_request':math}).status_code==200
        import json
        browser_problem=p.to_dict();browser_problem['operator_spec']=json.loads(json.dumps(browser_problem['operator_spec']),parse_float=lambda x:int(float(x)) if float(x).is_integer() else float(x))
        assert client.post('/api/coordinate/validate',json={**body,'problem':browser_problem,'mathematical_request':math}).status_code==200
        math['equations']=['u_τ=D*lap(u)+u']
        assert client.post('/api/coordinate/validate',json={**body,'mathematical_request':math}).status_code==422
        assert client.post('/api/coordinate/validate',json={**body,'problem':replace(p,known_parameters={'diffusivity':.9}).to_dict()}).status_code==422


def test_reference_import_manufactured_sampling_reproducibility_noise_and_no_leak(tmp_path):
    spec=example('heat-2d',(4,4));created=create(spec,tmp_path/'reference');p,y,m=load(tmp_path/'reference')
    payload={'problem':p.to_dict(),'coordinates':{a:c.tolist() for a,c in zip(p.grid.axis_names,p.grid.coordinates())},'physical_times':list(p.grid.times),'fields':['u'],'values':y[0].tolist(),'units':'nondimensional','provenance':{'solver':'analytic finite mode'}}
    import_reference(payload,tmp_path/'import','a'*32);np.testing.assert_array_equal(load(tmp_path/'import')[1],y)
    bad={**payload,'coordinates':{**payload['coordinates'],'x':[0,0,.5,.75]}}
    with pytest.raises(ValueError):import_reference(bad,tmp_path/'bad','b'*32)
    sampled=[]
    for name in ('one','two'):
        sampled.append(sample(tmp_path/'import',ObservationSamplingSpec(count=3,fields=('u',),time_indices=(0,2),noise_std=.1,noise_mode='relative_rms'),tmp_path/name))
    assert sampled[0]['rows']==sampled[1]['rows'] and sampled[0]['count']==6
    assert all('sigma' in r for r in sampled[0]['rows'])
    obs=read_json(tmp_path/'one/observations.json');assert len(obs['measurement_noise_std'])==6;assert 'clean_values' not in obs
    from pdeflow.observations import ObservationSet
    restored=ObservationSet.from_dict(obs,p.grid);assert restored.to_dict()==obs
    from pdeflow.reference.solvers import ReferenceProblem,solve
    candidate=ReferenceProblem(spec.request,{'initial_u':{'expression':'cos(2*pi*x)'},'analytical_reference':{'expression':'cos(2*pi*x)'}},'manufactured')
    _,field,diagnostic=solve(candidate);assert not diagnostic['verification']['passed'];assert 'unverified' in diagnostic['classification'];assert np.isfinite(field).all()
    with pytest.raises(ValueError):solve(replace(candidate,solver='analytical'))
    singular=replace(candidate,inputs={'initial_u':0,'analytical_reference':{'expression':'1/(x-x)'}})
    with pytest.raises(ValueError):solve(singular)


def test_ae_evaluation_portable_flow_and_failed_dependencies(tmp_path,tiny):
    import shutil
    from pdeflow.coordinate.autoencoder_evaluation import evaluate
    protected=tmp_path/'user-output';protected.mkdir();(protected/'notes.txt').write_text('keep')
    with pytest.raises(FileExistsError):train(tiny,protected,'autoencoder')
    assert (protected/'notes.txt').read_text()=='keep' and not (protected/'dataset.npz').exists()
    ae=tmp_path/'coordinate/ae';train(tiny,ae,'autoencoder')
    report=evaluate(ae/'autoencoder.pt',tmp_path/'ae-evaluation.json')
    assert len(report['records'])==1 and report['records'][0]['spectral_power_relative_l2']>=0
    flow=tmp_path/'coordinate/flow';train(tiny,flow,'flow',ae/'autoencoder.pt')
    isolated=tmp_path/'portable.pt';shutil.copyfile(flow/'coordinate.pt',isolated)
    # Full reference/data/AE stage are absent from this standalone inference directory.
    from pdeflow.coordinate.registry import register_checkpoint
    registered=register_checkpoint(isolated,tmp_path/'portable-workspace');assert registered['progress']==2
    assert runs(tmp_path/'portable-workspace')[0]['available']
    ae.rename(tmp_path/'offline-ae');flow.rename(tmp_path/'offline-flow')
    bundle=CoordinateBundle(isolated);p=heat_problem(GridSpec((4,4),DomainSpec((1.,1.)),(0.,.05,.1)),.07)
    samples,_=bundle.sample(p,[],grid_queries(p.grid),steps=1,particles=1);assert np.isfinite(samples).all()
    cp,c=read_checkpoint(isolated);assert cp['history'][0]['step']==1
    with pytest.raises(ValueError):train(tiny,tmp_path/'coordinate/bad','flow',isolated)
    assert read_json(tmp_path/'coordinate/bad/run.json')['state']=='failed'
    assert next(r for r in runs(tmp_path) if r['id']=='bad')['state']=='failed'


def test_full_field_api_import_export_and_noise_row_roundtrip(tmp_path):
    create(example('heat-2d',(4,4)),tmp_path/'reference')
    from pdeflow.reference.imports import export_reference
    artifact=export_reference(tmp_path/'reference');app=create_app(workspace=tmp_path/'workspace')
    with TestClient(app) as client:
        imported=client.post('/api/references/import',json={'artifact':artifact});assert imported.status_code==201
        identifier=imported.json()['id'];again=client.get(f'/api/references/{identifier}/artifacts/reference-field.json');assert again.status_code==200
        assert again.json()['values']==artifact['values']
        observed=client.post(f'/api/references/{identifier}/sample',json={'sampling':{'pattern':'space_time','count':3,'noise_std':0,'noise_mode':'absolute'}});assert observed.status_code==201
        rows=observed.json()['rows'];validated=client.post('/api/observations/validate',json={'problem':artifact['problem'],'rows':rows});assert validated.status_code==200
        assert validated.json()['rows']==rows
        oversized=client.post(f'/api/references/{identifier}/sample',json={'sampling':{'pattern':'fixed_sensors','count':17}});assert oversized.status_code==422

@pytest.mark.parametrize('dimension',[1,3])
def test_true_spatial_dimension_small_training_reload_and_scattered_queries(tmp_path,tiny,dimension):
    config={**tiny,'dimension':dimension,'reference_shape':[4]*dimension,'lengths':[1]*dimension,'origins':[0]*dimension}
    ae=tmp_path/'ae';flow=tmp_path/'flow';train(config,ae,'autoencoder');train(config,flow,'flow',ae/'autoencoder.pt')
    p=heat_problem(GridSpec((5,)*dimension,DomainSpec((1.,)*dimension),(0.,.025,.1)),.07)
    q=grid_queries(p.grid);bundle=CoordinateBundle(flow/'coordinate.pt');samples,report=bundle.sample(p,[],q,steps=1,particles=1,chunk_size=19)
    assert samples.shape==(1,3*5**dimension) and np.isfinite(samples).all() and np.isfinite(report['physics']['rmse'])
