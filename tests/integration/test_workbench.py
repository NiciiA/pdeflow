import json
import time
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from pdeflow.workbench.api import create_app
from pdeflow.workbench.jobs import JobManager,contained
from pdeflow.runtime import write_json,read_json


def fail_worker(directory,checkpoint):
    raise RuntimeError('controlled worker failure')


def slow_worker(directory,checkpoint):
    time.sleep(30)


def wait(client,identifier):
    deadline=time.monotonic()+30
    while time.monotonic()<deadline:
        record=client.get('/api/jobs/'+identifier).json()
        if record['status'] not in ('queued','running'):return record
        time.sleep(.1)
    raise AssertionError('job timeout')


@pytest.fixture
def client(tmp_path):
    from dataclasses import replace
    from pdeflow.runtime import Config
    from pdeflow.data import generate
    from pdeflow.training import train
    for dimension in (2,3):
        config=Config.load(f'configs/experiments/cpu_{dimension}d.json')
        config=replace(config,data=replace(config.data,count=6),model=replace(config.model,width=4,blocks=1),training=replace(config.training,steps=1))
        generate(config,tmp_path/f'dataset{dimension}')
        train(config,tmp_path/f'dataset{dimension}',tmp_path/'models'/f'model{dimension}')
    app=create_app(model_roots=[tmp_path/'models'],output_root=tmp_path/'jobs',data_root=tmp_path/'data')
    with TestClient(app) as c:yield c


def payload(client,dimension=2):
    models=client.get('/api/models').json();model=next(m for m in models if len(m['problem']['grid']['shape'])==dimension)
    demo=client.post('/api/demos',json={'model_id':model['id'],'sensors':4}).json()
    request={'model_id':model['id'],'problem':demo['problem'],'rows':demo['rows'],'inference':{'steps':3,'particles':2},'submission_token':'test-token'}
    return request,demo


def test_api_job_real_result_reference_isolation_and_slices(client):
    request,demo=payload(client,3)
    assert client.post('/api/reconstruction/validate',json=request).status_code==200
    submitted=client.post('/api/jobs',json=request);assert submitted.status_code==202
    identifier=submitted.json()['id']
    assert client.post('/api/jobs',json=request).json()['id']==identifier
    progress=[]
    result=wait(client,identifier);assert result['status']=='succeeded',result
    report=client.get(f'/api/jobs/{identifier}/result').json();assert not report['reference_available']
    assert 'relative_l2_unobserved_mean' not in report['metrics']
    before=client.get(f'/api/jobs/{identifier}/artifacts/samples.npz').content
    assert client.post(f'/api/jobs/{identifier}/reference',json={'reference_id':demo['reference_id']}).status_code==200
    for axis in range(3):
        response=client.get(f'/api/jobs/{identifier}/slice?axis={axis}&index=3&time_index=2&sample=-2')
        assert response.status_code==200;assert response.json()['shape']==[8,8];assert response.json()['reference'] is None
    assert client.get(f'/api/jobs/{identifier}/slice?index=999').status_code==422
    assert client.post(f'/api/jobs/{identifier}/reference',json={'reference_id':None}).status_code==200
    import io,numpy as np
    after=client.get(f'/api/jobs/{identifier}/artifacts/samples.npz').content
    assert np.array_equal(np.load(io.BytesIO(before))['samples'],np.load(io.BytesIO(after))['samples'])
    assert client.get(f'/api/jobs/{identifier}/artifacts/secret').status_code==422


def test_validation_and_path_rejection(client):
    request,demo=payload(client)
    bad=dict(request,reference=demo['reference_id']);assert client.post('/api/jobs',json=bad).status_code==422
    request['problem']['family']='darcy';assert client.post('/api/jobs',json=request).status_code==422
    request['problem']['family']='heat';request['problem']['known_parameters']['diffusivity']=-1
    assert client.post('/api/jobs',json=request).status_code==422
    request['problem']['known_parameters']['diffusivity']=.07
    for rows in ([],[{'field':'u','x':0,'time':0,'value':1}],[{**request['rows'][0],'x':.123}],[request['rows'][0]]*2):
        assert client.post('/api/observations/validate',json={'problem':request['problem'],'rows':rows}).status_code==422
    assert client.post('/api/observations/validate',json={'problem':request['problem'],'text':'field,x,value\nu,0,1','format':'csv'}).status_code==422
    assert client.get('/api/jobs/bad-id').status_code==422
    assert client.post('/api/problems/validate',json={'problem':{}}).status_code==422
    assert client.post('/api/jobs',content=b'x'*2_000_001,headers={'content-type':'application/json'}).status_code==413


def test_cancel_failure_and_restart(tmp_path):
    root=tmp_path/'jobs'
    request={'model_id':'test','inference':{'seed':7}}
    manager=JobManager(root,worker=slow_worker)
    job=manager.submit(request,'unused','a');time.sleep(.2)
    assert manager.cancel(job['id'])['status']=='cancelled'
    queued=manager.submit(request,'unused','b');manager.close()
    assert read_json(root/queued['id']/'job.json')['status']=='interrupted'
    # Startup recovery from an actual persisted active state, with no orphan worker.
    record=read_json(root/queued['id']/'job.json');record['status']='running';write_json(root/queued['id']/'job.json',record)
    manager=JobManager(root,worker=fail_worker)
    assert manager.get(queued['id'])['status']=='interrupted'
    failed=manager.submit(request,'unused','c')
    for _ in range(100):
        if manager.get(failed['id'])['status']=='failed':break
        time.sleep(.1)
    assert manager.get(failed['id'])['status']=='failed'
    manager.close()
    with pytest.raises(ValueError):contained(root,'../escape')


def test_symbolic_authority_rejects_stale_source_and_preserves_full_request(client):
    from copy import deepcopy
    from pdeflow.symbolic import presets
    request,demo=payload(client)
    heat=presets()[0]
    mathematical={'schema_version':1,'equations':heat['equations'],'declarations':heat['declarations'],
        'domain':request['problem']['grid'],'boundary':{'type':'periodic','status':'known'},'initial':{'status':'unknown'},
        'measurements':{'rows':request['rows'],'noise_model':'independent_gaussian','noise_std':.03,'status':'known'},
        'observation_operator':{'kind':'point_on_grid','status':'known'},'units':'nondimensional','preset':None}
    assert len(client.get('/api/mathematics/presets').json())==9
    parsed=client.post('/api/mathematics/parse',json={'request':mathematical,'model_id':request['model_id']})
    assert parsed.status_code==200;assert parsed.json()['status']['checkpoint']
    request['mathematical_request']=mathematical
    # JavaScript JSON.stringify writes 0.0/-1.0 as 0/-1. Physical equality
    # must survive that numeric spelling without weakening source validation.
    def browser_numbers(value):
        if isinstance(value,dict):return {k:browser_numbers(v) for k,v in value.items()}
        if isinstance(value,(tuple,list)):return [browser_numbers(v) for v in value]
        return int(value) if isinstance(value,float) and value.is_integer() else value
    request=browser_numbers(request);mathematical=request['mathematical_request']
    assert client.post('/api/reconstruction/validate',json=request).status_code==200
    saved=deepcopy(mathematical)
    for equation in ['u_τ=', 'u_τ=D lap(u)+u']:
        mathematical['equations']=[equation]
        response=client.post('/api/jobs',json=request);assert response.status_code==422,response.text
    request['mathematical_request']=saved
    response=client.post('/api/jobs',json=request);assert response.status_code==202,response.text
    identifier=response.json()['id'];assert wait(client,identifier)['status']=='succeeded'
    actual=client.get(f'/api/jobs/{identifier}/result').json()['request']['mathematical_request']
    assert actual==saved
    saved['domain']['shape']=[16,16];request['problem']['grid']['shape']=[16,16]
    assert client.post('/api/reconstruction/validate',json=request).status_code==422
    saved['domain']['shape']=[8,8];request['problem']['grid']['shape']=[8,8]
    saved['measurements']['noise_std']=.1
    assert client.post('/api/jobs',json=request).status_code==422
