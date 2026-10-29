"""Real bounded reconstruction and managed children, with separate truth artifacts."""
from dataclasses import asdict,replace
from pathlib import Path
import io,os,signal,uuid,zipfile
import numpy as np
import pytest,torch
from fastapi.testclient import TestClient
from pdeflow.reference import create
from pdeflow.reference.examples import example
from pdeflow.runtime import read_json,write_json,file_digest
from pdeflow import suites,local
from pdeflow.workbench.api import create_app
from test_workbench import wait
ROOT=Path(__file__).resolve().parents[2]

@pytest.mark.parametrize('family',['heat','darcy','chemistry'])
def test_real_api_experiment_isolation_export(tmp_path,family):
    roots={'heat':'outputs/accepted_cpu_2d/training','darcy':'outputs/multispecialist-20261021/cpu-2d/training','chemistry':'outputs/online-chemistry-20261021/chemistry-2d/training'}
    root=ROOT/roots[family]
    if not root.exists():pytest.skip('Optional historical smoke checkpoint unavailable')
    app=create_app(model_roots=[root],output_root=tmp_path/'jobs',data_root=tmp_path/'data',workspace=tmp_path/'workspace')
    with TestClient(app) as c:
        spec=example(family+'-2d');r=c.post('/api/references',json=asdict(spec));assert r.status_code==201,r.text
        rid=r.json()['id'];assert c.get(f'/api/references/{rid}/slice').status_code==200
        fields=['u'] if family=='darcy' else ['a'] if family=='chemistry' else ['u']
        sampling=c.post(f'/api/references/{rid}/sample',json={'sampling':{'count':3,'fields':fields}});assert sampling.status_code==201,sampling.text
        observed=sampling.json();p=observed['problem'];model=c.get('/api/models').json()[0]
        req={'model_id':model['id'],'problem':p,'rows':observed['rows'],'noise_std':.03,'inference':{'steps':2,'particles':2},'mode':'observation_pde','mathematical_request':observed['mathematical_request'],'submission_token':uuid.uuid4().hex}
        bad={**req,'reference_id':rid};assert c.post('/api/jobs',json=bad).status_code==422
        wrong=replace(spec,inputs={**spec.inputs,('permeability' if family=='darcy' else 'initial_a' if family=='chemistry' else 'initial_u'):1.1})
        other=c.post('/api/references',json=asdict(wrong));assert other.status_code==201,other.text
        ids=[]
        for j in range(2):
            req['submission_token']=uuid.uuid4().hex
            submitted=c.post('/api/jobs',json=req);assert submitted.status_code==202,submitted.text
            identifier=submitted.json()['id'];ids.append(identifier);assert wait(c,identifier)['status']=='succeeded'
        a=np.load(tmp_path/'jobs'/ids[0]/'result/result.npz');b=np.load(tmp_path/'jobs'/ids[1]/'result/result.npz')
        np.testing.assert_array_equal(a['samples'],b['samples'])
        for key in a.files:
            if key!='reference':np.testing.assert_array_equal(a[key],b[key])
        before=a['samples'].copy();report=read_json(tmp_path/'jobs'/ids[0]/'result/metrics.json')
        for refid,metrics in [(rid,['mse','rmse','mae','relative_l2']),(other.json()['id'],['mse']), (None,['rmse'])]:
            response=c.post(f'/api/jobs/{ids[0]}/reference',json={'reference_id':refid,'metrics':metrics});assert response.status_code==200,response.text
            after=np.load(tmp_path/'jobs'/ids[0]/'result/result.npz');np.testing.assert_array_equal(before,after['samples'])
            current=read_json(tmp_path/'jobs'/ids[0]/'result/metrics.json')
            for key in ['stop_reason','cost','provenance']:assert current[key]==report[key]
        assert c.post(f'/api/jobs/{ids[0]}/reference',json={'reference_id':rid,'sampling_id':observed['sampling_id']}).status_code==200
        z=zipfile.ZipFile(io.BytesIO(c.get(f'/api/jobs/{ids[0]}/experiment.zip').content));assert z.testzip() is None
        assert 'sampling/clean-samples-evaluation-only.npz' in z.namelist() and 'reference/reference-problem.json' in z.namelist()
        assert 'reference' not in read_json(tmp_path/'jobs'/ids[0]/'request.json')
        csv=c.get(f'/api/references/{rid}/artifacts/observations.csv?sampling_id={observed["sampling_id"]}');assert csv.status_code==200 and 'field' in csv.text


def test_suite_plan_real_interrupt_resume_reuse_and_discovery(tmp_path,monkeypatch):
    planned=suites.plan('quickstart','cpu',tmp_path,include=('heat-2d','darcy-2d'))
    assert len(planned['entries'])==2 and not tmp_path.joinpath('runs').exists()
    original=torch.optim.Adam.step;calls=0
    def interrupted(self,*a,**k):
        nonlocal calls
        value=original(self,*a,**k);calls+=1
        if calls==3:os.kill(os.getpid(),signal.SIGINT)
        return value
    with monkeypatch.context() as patch:
        patch.setattr(torch.optim.Adam,'step',interrupted)
        result=suites.start('quickstart','cpu',tmp_path,include=('heat-2d','darcy-2d'))
    assert result['state']=='interrupted' and [e['state'] for e in result['entries']]==['completed','interrupted']
    first=Path(result['entries'][0]['run_path']);second=Path(result['entries'][1]['run_path'])
    first_history=(first/'training/history.json').read_bytes();fixed=(second/'evaluation/manifest.json').read_bytes();calibration=(second/'training/calibration.json').read_bytes()
    with local.run_lock(Path(result['path'])):
        with pytest.raises(ValueError,match='Another process'):suites.resume(workspace=tmp_path)
    complete=suites.resume(workspace=tmp_path)
    assert complete['state']=='completed' and first_history==(first/'training/history.json').read_bytes()
    assert fixed==(second/'evaluation/manifest.json').read_bytes() and calibration==(second/'training/calibration.json').read_bytes()
    assert len(local.model_entries(tmp_path))>=2
    reused=suites.start('quickstart','cpu',tmp_path,include=('heat-2d','darcy-2d'),reuse=True)
    assert all(e['reused'] for e in reused['entries'])
    manifest=read_json(Path(complete['path'])/'suite.json');manifest['entries'][0]['resolved_config']['training']['steps']=999
    write_json(Path(complete['path'])/'suite.json',manifest)
    with pytest.raises(ValueError,match='configuration changed'):suites.resume(complete['path'])


def test_suite_failure_policy(tmp_path,monkeypatch):
    def failed(**kwargs):raise RuntimeError('Deliberate preparation failure; no numerical result')
    monkeypatch.setattr(local,'start_training',failed)
    fail=suites.start('quickstart','cpu',tmp_path/'fail',include=('heat-2d','darcy-2d'))
    assert [e['state'] for e in fail['entries']]==['failed','pending']
    continued=suites.start('quickstart','cpu',tmp_path/'continue',include=('heat-2d','darcy-2d'),continue_on_error=True)
    assert continued['state']=='completed_with_failures' and all(e['state']=='failed' for e in continued['entries'])


def test_capacity_suite_plan_and_wrong_budget_are_explicit(tmp_path):
    p=suites.plan('capacity-local','mps',tmp_path)
    assert [r['preset'] for r in p['entries']]==['heat-2d-medium','heat-2d-large']
    assert [r['grid'] for r in p['entries']]==[[64,64],[128,128]] and not (tmp_path/'runs').exists()
    with pytest.raises(ValueError,match='selected suite'):suites.plan('capacity-local','mps',tmp_path,budgets={'heat-2d':1})
