"""Two physical families through the common pipeline and equation router."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from pdeflow.core import Conditioning,ProblemSpec
from pdeflow.data import generate,load_dataset
from pdeflow.observations import ObservationSet
from pdeflow.runtime import Config,RuntimeSpec,read_json
from pdeflow.sampling import sample
from pdeflow.training import train,load_inference
from pdeflow.symbolic import analyze,presets
from pdeflow.workbench.api import create_app
from pdeflow.workbench.observations import observation_rows,parse_rows
from test_workbench import wait


def mathematical(p,rows=()):
    preset=next(x for x in presets() if x['id']==p.family)
    domain=json.loads(json.dumps(p.to_dict()['grid']))
    if p.grid.temporal_kind=='static': domain['times']=[]
    return dict(schema_version=1,equations=preset['equations'],declarations=preset['declarations'],domain=domain,
        boundary=dict(type='periodic',status='known'),initial=dict(status=p.initial_condition),known_fields=p.known_fields,
        measurements=dict(rows=list(rows),noise_model='independent_gaussian',noise_std=.005,status='known'),
        observation_operator=dict(kind='point_on_grid',status='known'),units='nondimensional',preset='misleading-label')


@pytest.fixture(scope='module')
def suite(tmp_path_factory):
    root=tmp_path_factory.mktemp('specialists');configs={}
    for family,dim in [('heat',2),('darcy',2),('darcy',3)]:
        c=Config.load(f'configs/experiments/{"darcy_" if family=="darcy" else ""}cpu_{dim}d.json')
        c=replace(c,data=replace(c.data,count=6,storage="sharded_npz",split_counts=(4,1,1),shard_size=2),model=replace(c.model,width=4,blocks=1),training=replace(c.training,steps=2,checkpoint_every=1))
        dest=root/f'{family}{dim}';generate(c,dest/'data');train(c,dest/'data',dest/'training',stop_after=1);train(c,dest/'data',dest/'training',resume=dest/'training/checkpoint.pt')
        configs[family,dim]=(c,dest)
    return root,configs


def test_routing_variants_roles_ranges_and_exclusions(suite,tmp_path):
    root,configs=suite
    with TestClient(create_app([root],tmp_path/'jobs',tmp_path/'refs')) as client:
        choices=client.get('/api/models').json();assert len(choices)==3
        for family,dim in configs:
            model=next(m for m in choices if m['problem']['family']==family and len(m['problem']['grid']['shape'])==dim)
            p=ProblemSpec.from_dict(model['problem']);draft=mathematical(p)
            for eq in (['u_τ=D lap(u)','u_τ-D lap(u)=0'] if family=='heat' else ['-div(a grad(u))=f','-div(a grad(u))-f=0']):
                if family=='heat' and '=' not in eq:eq+=''
                draft['equations']=[eq]
                response=client.post('/api/mathematics/parse',json={'request':draft});assert response.status_code==200,response.text
                r=response.json();assert r['recognized_family']==family and r['routing']['selected_checkpoint']==model['id'],r
                assert len(r['routing']['exclusions'])==2
            wrong=next(m for m in choices if m['problem']['family']!=family)
            r=client.post('/api/mathematics/parse',json={'request':draft,'model_id':wrong['id']}).json()
            assert not r['status']['checkpoint'] and r['routing']['selected_checkpoint'] is None
            draft['equations'][0]=draft['equations'][0].replace('=0','+u=0')
            assert client.post('/api/mathematics/parse',json={'request':draft}).json()['problem'] is None
            draft=mathematical(p);draft['boundary']['type']='neumann'
            assert client.post('/api/mathematics/parse',json={'request':draft}).json()['problem'] is None
            if family=='darcy':
                draft=mathematical(p);draft['declarations'][1]['status']='known'
                assert analyze(draft)['problem'] is None
                draft=mathematical(p);draft['known_fields']['f']=np.ones(p.grid.shape).tolist()
                assert analyze(draft)['problem'] is None


@pytest.mark.parametrize('dim',[2,3])
def test_static_joint_guidance_and_reference_isolation(suite,dim,tmp_path):
    root,configs=suite;c,dest=configs['darcy',dim]
    p=ProblemSpec.from_dict(read_json(dest/'data/test_problem.json'));obs=ObservationSet.from_dict(read_json(dest/'data/test_observations.json'),p.grid)
    rows=observation_rows(obs,p);assert all('time' not in r for r in rows)
    assert parse_rows(p,rows).operator.coordinates()==obs.operator.coordinates()
    data=load_dataset(dest/'data',c);assert set(data.transform().fitted_ids).isdisjoint({data.metadata['ids'][i] for i in data.indices('test')})
    b=load_inference(dest/'training/inference.pt',RuntimeSpec(device='cpu'))
    inf=replace(c.inference,steps=2,particles=1)
    results={mode:sample(b,p,obs,inf,mode).samples for mode in ['unguided','observation','observation_pde']}
    assert not torch.equal(results['unguided'],results['observation']) and not torch.equal(results['observation'],results['observation_pde'])
    for x in results.values():
        assert x.shape==(1,1,2,1,*p.grid.shape) and torch.isfinite(x).all() and (x[:,:,0]>0).all()
        assert x[:,:,1].mean().abs()<1e-7
    physical=data.values[[0]].double().requires_grad_();single=rows[:1];single[0]['value']=0.
    for field in ['a','u']:
        selected=[r for r in rows if r['field']==field][:1];o=parse_rows(p,selected,noise_std=.005,tolerance=.02)
        gradient=torch.autograd.grad(o.per_problem_energy(physical).sum(),physical,retain_graph=True)[0]
        assert gradient[:,['a','u'].index(field)].abs().sum()>0
    with TestClient(create_app([root],tmp_path/'jobs',tmp_path/'refs')) as client:
        model=next(m for m in client.get('/api/models').json() if m['problem']['family']=='darcy' and len(m['problem']['grid']['shape'])==dim)
        demo=client.post('/api/demos',json={'model_id':model['id'],'problem':model['problem'],'sensors':4,'noise_std':.005}).json()
        p=ProblemSpec.from_dict(demo['problem']);draft=mathematical(p,demo['rows'])
        req=dict(model_id=model['id'],problem=p.to_dict(),rows=demo['rows'],noise_std=.005,mode='observation_pde',inference={'steps':2,'particles':1},mathematical_request=draft)
        wrong=next(m for m in client.get('/api/models').json() if m['problem']['family']=='heat')
        assert client.post('/api/jobs',json={**req,'model_id':wrong['id']}).status_code==422
        draft['equations']=['-div(a grad(u))+u=f'];assert client.post('/api/jobs',json=req).status_code==422
        draft['equations']=['-div(a grad(u))-f=0']
        response=client.post('/api/jobs',json=req);assert response.status_code==202,response.text
        identifier=response.json()['id'];assert wait(client,identifier)['status']=='succeeded'
        import io
        url=f'/api/jobs/{identifier}'
        before=np.load(io.BytesIO(client.get(url+'/artifacts/samples.npz').content))['samples'].copy()
        assert client.post(url+'/reference',json={'reference_id':demo['reference_id']}).status_code==200
        for field in (0,1):
            for axis in range(dim):
                r=client.get(url+f'/slice?field_index={field}&axis={axis}&index=2').json();assert r['physical_time'] is None and r['field']==['a','u'][field]
        assert client.post(url+'/reference',json={'reference_id':None}).status_code==200
        after=np.load(io.BytesIO(client.get(url+'/artifacts/samples.npz').content))['samples']
        assert np.array_equal(before,after)
        # Reference file is inaccessible to inference, including across all modes.
        np.savez(dest/'data/test_reference.npz',values=np.full((1,2,1,*p.grid.shape),999.))
        for mode in results:assert torch.equal(results[mode],sample(b,ProblemSpec.from_dict(read_json(dest/'data/test_problem.json')),obs,inf,mode).samples)


def test_checkpoint_replacement_retires_stale_content_id(suite,tmp_path):
    import shutil
    root,configs=suite
    export=tmp_path/'models/inference.pt';export.parent.mkdir()
    shutil.copyfile(configs['heat',2][1]/'training/inference.pt',export)
    with TestClient(create_app([export.parent],tmp_path/'jobs',tmp_path/'refs')) as client:
        old=client.get('/api/models').json()[0]
        shutil.copyfile(configs['darcy',2][1]/'training/inference.pt',export)
        current=client.get('/api/models').json()
        assert len(current)==1 and current[0]['specialist']=='darcy' and current[0]['id']!=old['id']
        assert client.post('/api/demos',json={'model_id':old['id']}).status_code==422


def test_queued_worker_rejects_replaced_export_before_sampling(suite,tmp_path):
    import shutil
    from pdeflow.runtime import write_json,file_digest
    from pdeflow.workbench.jobs import compute_job
    root,configs=suite;export=tmp_path/'inference.pt'
    shutil.copyfile(configs['heat',2][1]/'training/inference.pt',export)
    model_id=file_digest(export)[:32]
    shutil.copyfile(configs['darcy',2][1]/'training/inference.pt',export)
    job=tmp_path/'job';job.mkdir()
    write_json(job/'request.json',{'model_id':model_id,'runtime':{'device':'cpu'}})
    write_json(job/'job.json',{'status':'queued'})
    compute_job(job,export)
    record=read_json(job/'job.json')
    assert record['status']=='failed' and 'changed after selection' in record['error']['message']
    assert not (job/'result').exists()


def test_heat_rejects_extra_spatial_condition_instead_of_dropping_it(suite):
    root,configs=suite;c,dest=configs['heat',2]
    p=ProblemSpec.from_dict(read_json(dest/'data/test_problem.json'))
    draft=mathematical(p);draft['known_fields']={'f':np.zeros(p.grid.shape).tolist()}
    report=analyze(draft)
    assert report['recognized_family']=='heat' and report['problem'] is None
    assert 'known spatial-field' in report['reasons'][0]


def test_conventional_baseline_reference_attachment_cannot_change_inverse(suite,tmp_path):
    import io
    root,configs=suite
    with TestClient(create_app([root],tmp_path/'jobs',tmp_path/'refs')) as client:
        model=next(m for m in client.get('/api/models').json() if m['specialist']=='darcy' and len(m['problem']['grid']['shape'])==2)
        demo=client.post('/api/demos',json={'model_id':model['id'],'sensors':4}).json()
        request={'model_id':model['id'],'problem':demo['problem'],'rows':demo['rows'],'inference':{'steps':2,'particles':1}}
        response=client.post('/api/jobs',json=request);assert response.status_code==202
        identifier=response.json()['id'];assert wait(client,identifier)['status']=='succeeded'
        baseline=tmp_path/'jobs'/identifier/'baseline_darcy_inverse'
        with np.load(baseline/'result.npz') as a:before=a['samples'].copy()
        report=client.get(f'/api/jobs/{identifier}/result').json()['baselines']['darcy_inverse']
        assert 'fields' not in report and report['cost']['forward_solves']>0
        assert client.post(f'/api/jobs/{identifier}/reference',json={'reference_id':demo['reference_id']}).status_code==200
        with np.load(baseline/'result.npz') as a:attached=a['samples'].copy()
        assert np.array_equal(before,attached)
        assert set(client.get(f'/api/jobs/{identifier}/result').json()['baselines']['darcy_inverse']['fields'])=={'a','u','log_a'}
        assert client.post(f'/api/jobs/{identifier}/reference',json={'reference_id':None}).status_code==200
        with np.load(baseline/'result.npz') as a:after=a['samples'].copy()
        assert np.array_equal(before,after)
        assert 'fields' not in client.get(f'/api/jobs/{identifier}/result').json()['baselines']['darcy_inverse']
