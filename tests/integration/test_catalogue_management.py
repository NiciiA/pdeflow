"""Real bounded managed training: partial coverage, frozen snapshots and liveness."""
from pathlib import Path
import json
from pdeflow import local,suites
from pdeflow.catalogue import coverage
from pdeflow.runtime import read_json,write_json
from pdeflow.workbench.visuals import training_runs
from pdeflow.workbench.api import create_app
from fastapi.testclient import TestClient


def test_partial_suite_export_does_not_cover_full_budget_and_frozen_snapshot_survives(tmp_path):
    run=suites.start('quickstart','cpu',tmp_path,include=('heat-2d',),stop_after=1)
    assert run['state']=='interrupted';path=Path(run['path']);frozen=(path/'suite.json').read_bytes()
    child=Path(run['entries'][0]['run_path']);assert (child/'training/inference.pt').is_file()
    row=next(t for t in coverage(tmp_path)['targets'] if t['id']=='heat-2d')
    assert row['state']=='partial' and row['maximum_updates']==1 and row['compatible_discoverable_checkpoint']
    assert not row['completed_run_ids']
    full=suites.plan('full','cpu',tmp_path,missing=True)
    assert 'heat-2d' in [r['preset'] for r in full['entries']]
    status=suites.status(str(path));assert 'heat-2d-512' in status['current_targets_not_in_snapshot']
    assert (path/'suite.json').read_bytes()==frozen
    complete=suites.resume(str(path));assert complete['state']=='completed' and complete['original_catalogue_completed']
    assert not next(t for t in coverage(tmp_path)['targets'] if t['id']=='heat-2d')['completed_run_ids']
    assert suites.plan('quickstart','cpu',tmp_path,include=('heat-2d',),missing=True)['entries']==[]
    rows=training_runs(tmp_path);assert not rows[0]['active'];assert rows[0]['progress']==2
    assert rows[0]['live']['step']==2 and rows[0]['live']['unique_physical_problems']>0
    assert rows[0]['heartbeat']['pid'] and (child/'heartbeat.json').exists()
    record=read_json(child/'run.json');record['state']='training';write_json(child/'run.json',record)
    assert 'owner exited' in local.status(str(child))['state']
    assert not training_runs(tmp_path)[0]['active']
    with local.run_lock(child):
        write_json(child/'progress.json',{'step':1,'total':2,'updates_per_second':None})
        assert local.status(str(child))['current_update']==2 # checkpoint is newer than throttled telemetry


def test_api_plan_storage_preview_and_exact_compatibility_are_independent_of_frontend(tmp_path):
    from dataclasses import asdict
    from pdeflow.reference.examples import example
    from pdeflow.symbolic import analyze
    from pdeflow.core import ProblemSpec
    from pdeflow.inference import compatibility_report
    app=create_app(workspace=tmp_path)
    with TestClient(app) as c:
        before=list((tmp_path/'runs').glob('*'))
        r=c.get('/api/training-plan');assert r.status_code==200 and len(r.json()['entries'])==11
        assert before==list((tmp_path/'runs').glob('*'))
        assert len(c.get('/api/training-coverage').json()['targets'])==11
        inventory=c.get('/api/storage?refresh=true');assert inventory.status_code==200
        p=c.post('/api/storage/preview');assert p.status_code==200 and not p.json()['applied']
        assert c.post('/api/storage/pin',json={'path':'../escape','pinned':True}).status_code==422
        problem=analyze(example('heat-2d').request)['problem']
        templates=c.post('/api/analytical-examples',json={'problem':problem});assert templates.status_code==200
        t=templates.json()[0];spec=asdict(example('heat-2d'));spec['inputs']={'initial_u':{'expression':t['initial_expression']},'analytical_reference':{'expression':t['reference_expression']}}
        verified=c.post('/api/analytical-verify',json=spec);assert verified.status_code==200 and verified.json()['passed']
        spec['inputs']['analytical_reference']['expression']=t['initial_expression']
        bad=c.post('/api/analytical-verify',json=spec);assert bad.status_code==200 and not bad.json()['passed']
        assert c.post('/api/references',json=spec).status_code==422


def test_export_inspection_uses_real_shapes_without_mutating_rng(tmp_path):
    import torch
    from pdeflow.catalogue import export_record
    result=local.start_training(preset='quickstart-heat-2d',workspace=tmp_path,device='cpu',steps=1)
    path=Path(result['path'])/'training/inference.pt'
    rng=torch.random.get_rng_state().clone();summary=export_record(path)
    assert torch.equal(rng,torch.random.get_rng_state()) and summary['architecture']['parameters']>0
    checkpoint=torch.load(path,weights_only=True)
    key=next(k for k,v in checkpoint['model'].items() if v.ndim>1)
    checkpoint['model'][key]=checkpoint['model'][key].reshape(-1)
    bad=tmp_path/'bad-shape.pt';torch.save(checkpoint,bad)
    import pytest
    with pytest.raises(ValueError,match='shape mismatch'):export_record(bad)


def test_api_controller_budget_error_does_not_claim_a_model_is_missing(tmp_path):
    from pdeflow.specialists import get_specialist
    from pdeflow.workbench.api import create_app
    c=local.preset_config('heat-2d-512');p=get_specialist('heat').default_problem(c.grid)
    with TestClient(create_app(workspace=tmp_path)) as client:
        response=client.post('/api/reconstruction/validate',json={'problem':p.to_dict(),'model_selection':{'mode':'auto'},'rows':[{'field':'u','x':0,'y':0,'time':0,'value':1.}],'inference':{'controller':{}}})
        assert response.status_code==422 and 'Feedback controller budget' in response.json()['error']['message']
        assert 'No compatible trained model' not in response.json()['error']['message']
        assert not list((tmp_path/'workbench/jobs').glob('*/job.json'))
