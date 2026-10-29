"""Real local orchestration over the existing trainer, not successful CLI mocks."""
from dataclasses import replace
import json
import os
from pathlib import Path
import signal

import pytest
import torch
from fastapi.testclient import TestClient
from pdeflow import local
from pdeflow.runtime import Config, read_json, write_json
from pdeflow.training import _read_checkpoint
from pdeflow.workbench.api import create_app


def test_headless_setup_missing_node_and_frontend_error(tmp_path,monkeypatch):
    monkeypatch.setattr(local.shutil,'which',lambda name:None)
    assert local.setup(tmp_path,False)['training_launched'] is False
    sentinel=tmp_path/'settings.json';sentinel.write_text('keep me')
    local.setup(tmp_path,False);assert sentinel.read_text()=='keep me'
    with pytest.raises(ValueError,match='Frontend needs Node.*headless training'):
        local.setup(tmp_path,True)
    monkeypatch.setattr(local,'node_info',lambda:{'node':'v20.0.0','npm':'10','compatible':False,'required':'^22.13.0','action':'nvm install'})
    with pytest.raises(ValueError,match='detected node=v20.0.0.*nvm install'):local.require_node()
    run=local.start_training('quickstart-heat-2d',workspace=tmp_path,device='cpu',steps=2,stop_after=1)
    assert run['progress']==1 and not (Path(run['path'])/'data').exists()
    assert local.doctor(tmp_path)['ok']


def test_real_optimizer_resume_extension_and_lock(tmp_path,monkeypatch):
    workspace=tmp_path/'workspace'
    full=local.start_training('quickstart-heat-2d',workspace=workspace,device='cpu',steps=4,name='full')
    run=local.start_training('quickstart-heat-2d',workspace=workspace,device='cpu',steps=4,stop_after=1,name='resume')
    path=Path(run['path']);evaluation=(path/'evaluation/manifest.json').read_bytes();calibration=(path/'training/calibration.json').read_bytes()
    before=_read_checkpoint(path/'training/checkpoint.pt')[0]
    assert before['progress']==1 and before['optimizer']['state']
    with local.run_lock(path):
        with pytest.raises(ValueError,match='Another process owns run'):local.resume_training('latest',workspace)
    resumed=local.resume_training('latest',workspace)
    a=_read_checkpoint(Path(full['path'])/'training/checkpoint.pt')[0];b=_read_checkpoint(path/'training/checkpoint.pt')[0]
    assert all(torch.equal(v,b['model'][k]) for k,v in a['model'].items())
    assert [r['physical_ids'] for r in a['history']]==[r['physical_ids'] for r in b['history']]
    assert evaluation==(path/'evaluation/manifest.json').read_bytes()
    assert calibration==(path/'training/calibration.json').read_bytes()
    events=[json.loads(line) for line in (path/'training/data-events.jsonl').read_text().splitlines()]
    assert sum(e['phase']=='started' and e['specification']['namespace']=='calibration' for e in events)==4
    assert resumed['state']=='completed'
    # Stale metadata is harmless: the OS owns the lock, not its persisted PID.
    (path/'.run.lock').write_text('{"pid":99999999}')
    extended=local.resume_training(str(path),additional_steps=2)
    ckpt=_read_checkpoint(path/'training/checkpoint.pt')[0]
    assert extended['progress']==6 and extended['budget']==6
    assert ckpt['optimizer']['param_groups'][0]['lr']==a['optimizer']['param_groups'][0]['lr']
    assert ckpt['continuations'][-1]['from_steps']==4 and ckpt['continuations'][-1]['to_steps']==6
    assert local.resolve_run('latest',workspace)[0]==path
    with pytest.raises(ValueError,match='requested darcy'):local.resolve_run(str(path),specialist='darcy')


def test_actual_signal_safe_boundary_through_managed_run(tmp_path,monkeypatch):
    original=torch.optim.Adam.step
    def interrupt(self,*args,**kwargs):
        result=original(self,*args,**kwargs);os.kill(os.getpid(),signal.SIGINT);return result
    with monkeypatch.context() as patch:
        patch.setattr(torch.optim.Adam,'step',interrupt)
        run=local.start_training('quickstart-heat-2d',workspace=tmp_path,device='cpu',steps=3)
    assert run['state']=='interrupted' and run['progress']==1
    local.resume_training('latest',tmp_path)
    record=local.status('latest',tmp_path)
    history=read_json(Path(record['path'])/'training/history.json')
    ids=[i for row in history for i in row['physical_ids']]
    assert record['progress']==3 and len(ids)==len(set(ids))==6


def test_evaluation_discovery_and_reference_isolation(tmp_path,monkeypatch):
    monkeypatch.delenv('PDEFLOW_MODEL_ROOTS',raising=False)
    run=local.start_training('quickstart-heat-2d',workspace=tmp_path,device='cpu',steps=2)
    path=Path(run['path'])
    with TestClient(create_app(workspace=tmp_path)) as client:
        rows=client.get('/api/models').json()
        assert len(rows)>=1 and all(r['run_id']==run['id'] for r in rows)
        assert rows[0]['training']['updates']==2
        assert 'scientifically' not in rows[0]['training']['status'] or 'unevaluated' in rows[0]['training']['status']
        assert set(rows[0]['export_kinds'])=={'latest','validation-selected'}
        assert len(client.get('/api/models').json())==len(rows)
    result=local.evaluate_run('latest',tmp_path)
    assert result['split']=='validation' and result['case_count']==1 and 'fourier_inverse' in result['reports']['0']
    test=local.evaluate_run('latest',tmp_path,split='test')
    assert test['split']=='test' and 'must not become tuning' in test['purpose']
    # A different evaluation truth changes only scores, never inference.
    import numpy as np
    truth=path/'evaluation/validation/0/reference.npz'
    with np.load(truth) as a:wrong=np.full_like(a['values'],999.)
    np.savez(truth,values=wrong)
    other=local.evaluate_run('latest',tmp_path)
    for mode in ['unguided','observation','observation_pde']:
        with np.load(Path(result['output'])/'0'/mode/'result.npz') as a,np.load(Path(other['output'])/'0'/mode/'result.npz') as b:
            assert np.array_equal(a['samples'],b['samples'])
    # An incomplete export is never advertised as loadable.
    (path/'training/inference.pt').write_bytes(b'partial')
    (path/'training/best-inference.pt').write_bytes(b'partial')
    assert not local.models(tmp_path)
    with TestClient(create_app(workspace=tmp_path)) as client:assert client.get('/api/models').json()==[]


def test_partial_preparation_recovery_and_explicit_mps_rejection(tmp_path,monkeypatch):
    from pdeflow.data import providers
    c=local.preset_config('quickstart-heat-2d');c=replace(c,runtime=replace(c.runtime,device='cpu'))
    original=providers.reference_sample
    def fail(c,namespace,index,event=None):
        if namespace=='test':raise RuntimeError('controlled preparation failure')
        return original(c,namespace,index,event)
    with monkeypatch.context() as patch:
        patch.setattr(providers,'reference_sample',fail)
        with pytest.raises(RuntimeError,match='preparation failure'):providers.prepare(c,tmp_path/'eval',resume=True)
    before=(tmp_path/'eval/validation/0/sample.json').read_bytes()
    providers.prepare(c,tmp_path/'eval',resume=True)
    assert before==(tmp_path/'eval/validation/0/sample.json').read_bytes()
    monkeypatch.setattr(torch.backends.mps,'is_available',lambda:False)
    with pytest.raises(ValueError,match='MPS requested but unavailable'):
        local.start_training('heat-2d',workspace=tmp_path/'no-fallback',device='mps',steps=1)
    assert not (tmp_path/'no-fallback').exists()


def test_presets_plan_and_no_private_paths(tmp_path):
    rows=local.presets();assert len(rows)>=7
    assert local.preset_config('heat-2d').training.steps>=1000
    assert {'heat','darcy','chemistry'}=={r['specialist'] for r in rows}
    for row in rows:
        c=local.preset_config(row['name']);c.validate(check_device=False)
    report=local.start_training('heat-2d',workspace=tmp_path/'plan',device='cpu',dry_run=True)
    assert not (tmp_path/'plan').exists() and report['duration_estimate'].startswith('not measured')
    source=Path(local.__file__).read_text()
    assert '/Users/' not in source and '.codex' not in source


def test_external_output_discovery_and_server_lock(tmp_path,monkeypatch):
    monkeypatch.delenv('PDEFLOW_MODEL_ROOTS',raising=False)
    workspace=tmp_path/'workspace';output=tmp_path/'custom-output'
    run=local.start_training('quickstart-heat-2d',workspace=workspace,output=output,device='cpu',steps=1)
    assert local.resolve_run('latest',workspace)[0]==output
    with TestClient(create_app(workspace=workspace)) as first:
        assert first.get('/api/models').json()[0]['run_id']==run['id']
        with pytest.raises(ValueError,match='Another process owns run'):
            with TestClient(create_app(workspace=workspace)):pass
    with TestClient(create_app(workspace=workspace)) as restarted:
        assert restarted.get('/api/health').status_code==200


def test_preferred_notebook_uses_managed_commands(tmp_path,monkeypatch):
    import sys
    from pdeflow.cli import main
    notebook=json.loads(Path('notebooks/colab.ipynb').read_text())
    source=next(''.join(cell['source']) for cell in notebook['cells'] if cell['id']=='pdeflow-local-1')
    source=source.replace('"colab-heat-2d"','"quickstart-heat-2d"').replace('"cuda"','"cpu"')
    commands=[]
    def cli(*args):
        args=[str(a) for a in args]
        if args[0]=='train':args+=['--steps','2']
        commands.append(args)
        monkeypatch.setattr(sys,'argv',['pdeflow',*args]);main()
    scope={'OUTPUT_ROOT':tmp_path,'cli':cli}
    exec(compile(source,'local-notebook','exec'),scope)
    exec(compile(source,'local-notebook-resume','exec'),scope)
    assert any(c[0]=='resume' for c in commands)
    assert local.status('latest',scope['LOCAL_WORKSPACE'])['progress']==2


def test_port_preflight_matches_reusable_server_restart():
    import socket
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        server.bind(('127.0.0.1',0));server.listen()
        port=server.getsockname()[1]
        assert not local.port_available('127.0.0.1',port)
        with socket.create_connection(('127.0.0.1',port)) as client:
            peer,_=server.accept();peer.close()
            assert client.recv(1)==b''
    assert local.port_available('127.0.0.1',port)
