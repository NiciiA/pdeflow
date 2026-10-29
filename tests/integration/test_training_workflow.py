"""Real bounded training/evaluation, plus explicit control-plane failure probes."""
from pathlib import Path
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from pdeflow import training_workflow as workflow
from pdeflow.coordinate.training import read_checkpoint
from pdeflow.runtime import read_json, write_json, file_digest
from pdeflow.workbench.api import create_app


def request(model='resolution-flexible'):
    advanced = {'temporal_mode':'snapshots','data_mode':'materialized','reference_shape':[4,4], 'width':8, 'heads':2, 'tokens':3, 'layers':1,
                'train_count':3, 'validation_count':1, 'test_count':1, 'encoder_points':32,
                'query_points':32, 'ae_steps':2, 'flow_steps':2, 'checkpoint_every':1,
                'batch_size':1, 'observation_counts':[0,4]}
    if model == 'fixed-grid':
        advanced = {'model':{'width':8,'blocks':1}, 'grid':{'shape':[4,4]},
                    'provider':{'validation_count':1,'test_count':1,'calibration_count':3},
                    'inference':{'steps':1,'particles':1,'sensors':2}}
    return {'problem':'heat-2d','model':model,'budget':'pipeline-check','device':'cpu','advanced':advanced}


@pytest.mark.parametrize('model', ['resolution-flexible', 'fixed-grid'])
def test_one_action_real_training_validation_registry_selection_and_preview(tmp_path, model):
    row = workflow.start(request(model), tmp_path, token=model, background=False)
    assert row['state'] == 'completed', row.get('error')
    assert row['plan']['pipeline_check'] and 'not evidence of accuracy' in row['meaning']
    assert row['validation']['split'] == 'validation'
    assert set(row['finished_stages']) == set(row['plan']['stages'])
    assert not (tmp_path/'selected-model.json').exists()  # completion never promotes
    entries = workflow.model_list(tmp_path)
    own = next(m for m in entries if m['id'] == row['model_id'])
    assert own['pipeline_check'] and own['validation']
    selected = workflow.select_model(row['id'], tmp_path)
    assert selected['id'] == row['model_id']
    assert read_json(tmp_path/'selected-model.json')['id'] == row['model_id']
    plot = workflow.preview(row['id'], tmp_path)
    assert np.isfinite(plot['values']).all() and np.isfinite(plot['error']).all()
    assert np.asarray(plot['reference']).shape == np.asarray(plot['values']).shape
    with TestClient(create_app(workspace=tmp_path)) as client:
        assert client.get('/api/training/runs').json()[0]['id'] == row['id']
        assert client.get(f"/api/training/runs/{row['id']}/preview").status_code == 200
        models = client.post('/api/training/models', json={'problem':own['problem']}).json()
        assert next(m for m in models if m['id']==row['model_id'])['compatibility']['compatible']
        bad = {**own['problem'], 'family':'custom'}
        assert client.post('/api/training/use', json={'id':row['model_id'],'problem':bad}).status_code == 422
        adopted = client.post('/api/training/use', json={'id':row['model_id'],'problem':bad,'adopt_training_problem':True})
        assert adopted.status_code == 200 and adopted.json()['adopt_training_problem']
        if model == 'resolution-flexible':
            # Inverse accepts only actual measurements; registration adds no truth.
            check = client.post('/api/coordinate/validate', json={'model_id':row['model_id'],'problem':own['problem'],'rows':[],'steps':1,'particles':1})
            assert check.status_code == 200
        else:
            assert row['model_id'] in [m['id'] for m in client.get('/api/models').json()]


@pytest.mark.parametrize('model', ['resolution-flexible', 'fixed-grid'])
def test_stop_resume_real_zero_update_checkpoint_and_idempotent_submission(tmp_path, monkeypatch, model):
    launched = []
    monkeypatch.setattr(workflow, '_spawn', lambda path: launched.append(path) or 12345)
    row = workflow.start(request(model), tmp_path, token='repeat')
    again = workflow.start(request(model), tmp_path, token='repeat')
    duplicate = workflow.start(request(model), tmp_path, token='different-token')
    assert row['id']==again['id']==duplicate['id'] and len(launched)==1
    with pytest.raises(ValueError, match='different settings'):
        workflow.start({**request(model),'device':'auto'}, tmp_path, token='repeat')
    path = launched[0]
    # Stop at the first update boundary through the real reused trainer, then
    # complete through the same workflow. This is not a mocked optimizer.
    if model == 'resolution-flexible':
        from pdeflow.coordinate import training
    else:
        from pdeflow import training
    original = training.train
    first = [True]
    def stop_first(*args, **kwargs):
        if first[0]:
            first[0] = False
            kwargs['should_stop'] = lambda: True
        return original(*args, **kwargs)
    monkeypatch.setattr(training, 'train', stop_first)
    workflow.execute(path)
    stopped = workflow.status(row['id'], tmp_path)
    assert stopped['state']=='interrupted' and stopped['can_resume']
    if model == 'resolution-flexible':
        cp, _ = read_checkpoint(path/'coordinate-ae/resume.pt')
        assert cp['progress']==0 and not cp['completed']
    else:
        cp, _, _ = training._read_checkpoint(path/'fixed/training/checkpoint.pt')
        assert cp['progress']==0
    resumed = workflow.resume(row['id'], tmp_path, background=False)
    assert resumed['state']=='completed', resumed.get('error')
    assert resumed['attempts']==2


def test_stage_failure_is_persistent_and_does_not_launch_flow(tmp_path, monkeypatch):
    from pdeflow.coordinate import training
    def fail(*args, **kwargs):
        raise FloatingPointError('intentional stage failure probe')
    monkeypatch.setattr(training, 'train', fail)
    row = workflow.start(request(), tmp_path, background=False)
    assert row['state']=='failed' and 'intentional stage failure' in row['error']
    path = tmp_path/'training-workflows'/row['id']
    assert not (path/'coordinate-flow').exists() and row['model_id'] is None
    assert workflow.status(row['id'], tmp_path)['state']=='failed'


def test_validation_gate_stops_development_until_explicit_recorded_override(tmp_path):
    r = request(); r.update(budget='development', representation_max_mse_ratio=.000001)
    row = workflow.start(r, tmp_path, background=False)
    assert row['state']=='needs_review' and not row['representation_quality']['passed']
    root = tmp_path/'training-workflows'/row['id']
    assert not (root/'coordinate-flow').exists()
    ae_hash = file_digest(root/'coordinate-ae/autoencoder.pt')
    continued = workflow.resume(row['id'], tmp_path, allow_poor_representation=True, background=False)
    assert continued['state']=='completed', continued.get('error')
    assert continued['gate_decision']=='explicit override' and continued['quality_override']['reason']
    assert file_digest(root/'coordinate-ae/autoencoder.pt')==ae_hash
    assert not continued['plan']['pipeline_check']


def test_backend_restart_and_real_detached_worker_persistence(tmp_path):
    with TestClient(create_app(workspace=tmp_path)) as client:
        payload = {**request(), 'submission_token':'real-worker'}
        first = client.post('/api/training/start', json=payload)
        assert first.status_code==202
        identifier = first.json()['id']
        assert client.post('/api/training/start', json=payload).json()['id']==identifier
    # Closing/reopening the actual backend doesn't own the detached trainer.
    with TestClient(create_app(workspace=tmp_path)) as client:
        deadline = time.monotonic()+45
        while time.monotonic()<deadline:
            row=client.get(f'/api/training/runs/{identifier}').json()
            if row['state'] not in workflow.ACTIVE:break
            time.sleep(.1)
        assert row['state']=='completed', row
        assert len(client.get('/api/training/runs').json())==1
        assert client.post('/api/training/models',json={}).json()[0]['id']==row['model_id']


def test_devices_profiles_and_unavailable_choices_no_output_allocation(tmp_path, monkeypatch):
    options=workflow.options()
    auto=workflow.plan({'device':'auto'})
    assert auto['resolved_device']==options['auto_device']
    assert workflow.plan({'device':'cpu'})['resolved_device']=='cpu'
    monkeypatch.setattr(__import__('torch').cuda,'is_available',lambda:False)
    with pytest.raises(ValueError):workflow.start({'device':'cuda'},tmp_path)
    assert not (tmp_path/'training-workflows').exists()
    for problem in workflow.PROBLEMS:
        for model in ('fixed-grid','resolution-flexible') if problem.startswith('heat') else ('fixed-grid',):
            p=workflow.plan({'problem':problem,'model':model,'device':'cpu'})
            assert p['grid']['shape'] and p['pipeline_check']
    with pytest.raises(ValueError,match='heat only'):
        workflow.plan({'problem':'darcy-2d','model':'resolution-flexible','device':'cpu'})
    assert workflow.plan({'model':'resolution-flexible','budget':'extended','device':'cpu'})['updates']['flow']==10000
    assert workflow.plan({'device':'cpu','advanced':{'provider':{'mode':'offline','offline_count':7}}})['examples']['training']==7
    assert 'pool up to 5' in workflow.plan({'device':'cpu','advanced':{'provider':{'mode':'hybrid','capacity':5}}})['examples']['training']
    assert workflow.plan({'device':'cpu'})['request']==workflow.plan({'device':'cpu','advanced':{},'allow_poor_representation':False,'representation_max_mse_ratio':1})['request']


def test_preparation_restart_with_original_provenance(tmp_path, monkeypatch):
    from pdeflow.coordinate import training
    prepare=training.prepare
    def fail(*args):raise RuntimeError('failed preparation')
    monkeypatch.setattr(training,'prepare',fail)
    row=workflow.start(request(),tmp_path,background=False)
    assert row['state']=='failed'
    monkeypatch.setattr(training,'prepare',prepare)
    result=workflow.resume(row['id'],tmp_path,background=False)
    assert result['state']=='completed',result.get('error')


def test_lost_worker_is_interrupted_without_signalling_stale_pid(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow,'_spawn',lambda path:999999)
    row=workflow.start(request(),tmp_path)
    file=tmp_path/'training-workflows'/row['id']/'workflow.json'
    record=read_json(file);record['launched_unix']=time.time()-30;write_json(file,record)
    assert workflow.status(row['id'],tmp_path)['state']=='interrupted'
    assert workflow.status(row['id'],tmp_path)['can_resume']


def test_fixed_train_further_preserves_previous_budget_and_validation(tmp_path):
    row=workflow.start(request('fixed-grid'),tmp_path,background=False)
    assert row['state']=='completed'
    root=tmp_path/'training-workflows'/row['id']
    old_hash=file_digest(root/'fixed/training/inference.pt')
    old_report=(root/'model-validation.json').read_bytes()
    continued=workflow.extend(row['id'],2,tmp_path,background=False)
    assert continued['state']=='completed',continued.get('error')
    assert continued['plan']['updates']=={'fixed':4} and not continued['plan']['pipeline_check']
    assert file_digest(root/'budgets/2/inference.pt')==old_hash
    assert (root/'model-validation.json').read_bytes()==old_report
    assert (root/'model-validation-4.json').is_file()
    assert {row['model_id'],continued['model_id']} <= {m['id'] for m in workflow.model_list(tmp_path)}
    assert workflow.preview(row['id'],tmp_path)['error'] is not None


def test_saved_models_and_runs_are_discovered_without_special_workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow.local,'project_root',lambda:tmp_path)
    old=tmp_path/'outputs/earlier/workspace'
    row=workflow.start(request(),old,background=False)
    assert row['state']=='completed'
    default=tmp_path/'outputs/local'
    assert row['id'] in [r['id'] for r in workflow.runs(default)]
    entries=workflow.model_list(default)
    assert row['model_id'] in [m['id'] for m in entries]
    assert workflow.select_model(row['id'],default)['id']==row['model_id']
    assert (old/'training-workflows'/row['id']/'coordinate-flow/coordinate.pt').is_file()
