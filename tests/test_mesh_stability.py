from pathlib import Path
import json,time,threading,copy,io
import torch
import pytest
from pdeflow.mesh.contracts import preset
from pdeflow.mesh.backbone import SensorReader
from pdeflow.mesh.numerical import fp32_matmul,squared_error
from pdeflow.mesh.session import MeshSession
from pdeflow.mesh.monitoring import run,status
from pdeflow.mesh.diagnostics import checkpoint_health
from pdeflow.runtime import read_json,file_digest


def config(name='mesh_m1_v1',updates=3):
    torch.set_num_threads(2);return preset('pipeline-check',name,'cpu','fp32',{'data':{'query_points':8},'budgets':{s:updates for s in preset().budgets},'checkpoint_every':1,'validation_every':updates})


def test_actual_autocast_overflow_boundary_and_reader():
    c=config();m=SensorReader(c)
    with torch.no_grad():m.q.weight.zero_();m.q.bias.fill_(300);m.k.weight.zero_();m.k.bias.fill_(300)
    q=torch.full((1,2,c.width),300.);obs=(torch.zeros(1,1,3),torch.zeros(1,1,3),torch.ones(1,1,dtype=torch.bool),torch.zeros(1,1,dtype=torch.long))
    with torch.autocast('cpu',dtype=torch.float16):
        bad=q.float()@q.float().transpose(-1,-2);good=fp32_matmul(q,q.transpose(-1,-2));out=m(torch.zeros(1,c.nodes,c.width),torch.zeros(c.nodes,2),obs);loss=squared_error(out,torch.zeros_like(out)).mean()
    assert not torch.isfinite(bad).all() and good.dtype==torch.float32 and torch.isfinite(out).all();loss.backward();assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    half=torch.tensor([300.],dtype=torch.float16);assert torch.isinf(half.square()).all();torch.testing.assert_close(squared_error(half,torch.zeros_like(half)),torch.tensor([90000.]))


def test_invalid_forward_preserves_checkpoint_and_captures_before_backward(tmp_path,monkeypatch):
    from pdeflow.mesh import training
    c=config();s=MeshSession(c,tmp_path/'run',tmp_path/'store',save_updates=1);assert s.run(stop_after=1)['state']=='interrupted';path=s.root/'stages/reconstruction/resume.pt';before=file_digest(path);original=training.loss;calls=[]
    def bad(*a,**kw):
        values,components=original(*a,**kw);calls.append(True);return values*float('nan'),components
    monkeypatch.setattr(training,'loss',bad)
    with pytest.raises(FloatingPointError,match='before backward'):s.run()
    assert file_digest(path)==before and checkpoint_health(path)['healthy'];record=read_json(path.parent/'run.json');assert record['progress']==1 and record['attempted_updates']==2 and record['failed_attempts']==1
    latest=read_json(path.parent/'failure/latest.json');failure=path.parent/'failure'/latest['directory'];report=read_json(failure/'report.json');assert report['last_good_checkpoint']['sha256']==before and (failure/'batch.pt').exists() and (failure/'replay.pt').exists();assert read_json(s.root/'sync-status.json')['status']=='verified'
    assert not read_json(path.parent/'error.json')['diagnostics_error'];monkeypatch.setattr(training,'loss',original);assert s.run()['state']=='completed'


def test_invalid_gradient_never_steps_or_replaces_checkpoint(tmp_path,monkeypatch):
    from pdeflow.mesh import training
    c=config();s=MeshSession(c,tmp_path/'run');s.run(stop_after=1);path=s.root/'stages/reconstruction/resume.pt';before=file_digest(path);original=training.loss
    def bad(*a,**kw):
        values,parts=original(*a,**kw);values.register_hook(lambda x:torch.full_like(x,float('nan')));return values,parts
    monkeypatch.setattr(training,'loss',bad)
    with pytest.raises(RuntimeError,match='non-finite'):s.run()
    assert file_digest(path)==before and checkpoint_health(path)['healthy'];assert read_json(path.parent/'run.json')['failed_attempts']==1


def test_read_only_status_and_live_output_before_completion(tmp_path,capsys,monkeypatch):
    from pdeflow.mesh import training
    c=config(updates=8);s=MeshSession(c,tmp_path/'run',tmp_path/'store',save_updates=2);original=training.loss
    def slow(*a,**kw):time.sleep(.06);return original(*a,**kw)
    monkeypatch.setattr(training,'loss',slow);before=torch.get_rng_state().clone();missing=tmp_path/'does-not-exist';assert status(missing)['state']=='not started' and not missing.exists();assert torch.equal(before,torch.get_rng_state())
    result=run(s,refresh_seconds=.05,plots=False);text=capsys.readouterr().out;assert result['state']=='completed' and 'initialization' in text and 'training' in text and 'Successful' in text and 'precision fp32' in text
    snap=status(s.root);assert snap['successful_updates']==8 and snap['seconds_per_successful_update'] and snap['persistent']['snapshot'];assert snap['history'][-1].get('validation_loss') is not None
    before={str(p):p.stat().st_mtime_ns for p in s.root.rglob('*') if p.is_file()};rng=torch.get_rng_state().clone();status(s.root);after={str(p):p.stat().st_mtime_ns for p in s.root.rglob('*') if p.is_file()};assert before==after and torch.equal(rng,torch.get_rng_state())


def test_bounded_scaler_skip_semantics_without_cuda_claim(tmp_path,monkeypatch):
    from pdeflow.mesh import training
    c=config();original=training.optimizer_step;attempts=[]
    def injected(*a,**kw):
        attempts.append(1)
        return (torch.tensor(float('inf')),False,True) if len(attempts)<=2 else original(*a,**kw)
    monkeypatch.setattr(training,'optimizer_step',injected);s=MeshSession(c,tmp_path/'run');assert s.run()['state']=='completed';cp=torch.load(s.root/'stages/reconstruction/resume.pt',weights_only=True);assert cp['attempted_updates']==5 and cp['skipped_updates']==2 and cp['successful_updates']==3 and cp['failed_attempts']==0


def test_diagnostic_and_drive_error_do_not_suppress_original_failure(tmp_path,monkeypatch):
    from pdeflow.mesh import training
    from pdeflow import persistence
    c=config();s=MeshSession(c,tmp_path/'run',tmp_path/'store',save_updates=1);s.run(stop_after=1);file=s.root/'stages/reconstruction/resume.pt';before=file_digest(file);original=training.loss;publish=persistence.publish
    def bad(*a,**kw):
        values,parts=original(*a,**kw);return values*float('nan'),parts
    def failed_save(*a,**kw):
        if kw.get('tag')=='failed-safe-boundary':raise OSError('injected Drive failure')
        return publish(*a,**kw)
    monkeypatch.setattr(training,'loss',bad);monkeypatch.setattr(training,'capture_failure',lambda *a,**kw:(_ for _ in ()).throw(OSError('injected diagnostic failure')));monkeypatch.setattr(persistence,'publish',failed_save)
    with pytest.raises(FloatingPointError,match='before backward'):s.run()
    assert file_digest(file)==before;assert 'injected diagnostic failure' in read_json(file.parent/'error.json')['diagnostics_error'];sync=read_json(s.root/'sync-status.json');assert sync['status']=='failed' and sync['snapshot'] and sync['local_diagnostics_retained']


def test_live_suite_and_failure_state_are_actual(tmp_path,monkeypatch,capsys):
    from pdeflow.mesh.suite import Suite,suite_plan
    from pdeflow.mesh import training
    plan=suite_plan('pipeline-check','cpu','fp32',candidates=['mesh_m1_v1','mesh_m2_v1'],overrides={'data':{'query_points':8}});s=Suite.new(plan,tmp_path/'suites');original=training.loss
    def conditional(model,*a,**kw):
        values,parts=original(model,*a,**kw)
        return (values*float('nan'),parts) if model.c.architecture=='mesh_m2_v1' else (values,parts)
    monkeypatch.setattr(training,'loss',conditional)
    with pytest.raises(FloatingPointError):run(s,refresh_seconds=.05,plots=False)
    r=status(s.root);assert r['state']=='failed' and r['candidates']['mesh_m1_v1']=='completed' and r['candidates']['mesh_m2_v1']=='failed';assert 'Suite' in capsys.readouterr().out
    with pytest.raises(ValueError,match='explicit'):s.run()
    first=file_digest(s.root/'runs/mesh_m1_v1/stages/reconstruction/resume.pt');monkeypatch.setattr(training,'loss',original);assert s.run(selected=['mesh_m2_v1'])['state']=='completed';assert file_digest(s.root/'runs/mesh_m1_v1/stages/reconstruction/resume.pt')==first


def test_preflight_failure_retains_replay_after_temporary_cleanup(tmp_path,monkeypatch):
    from pdeflow.mesh import training
    from pdeflow.mesh.benchmark import profile
    original=training.loss
    def bad(*a,**kw):
        values,parts=original(*a,**kw);return values*float('nan'),parts
    monkeypatch.setattr(training,'loss',bad)
    with pytest.raises(FloatingPointError):profile(config(),tmp_path,updates=7,warmup=2)
    kept=list(tmp_path.glob('mesh-preflight-failed-*'));assert len(kept)==1 and (kept[0]/'source/source.zip').exists();assert list(kept[0].glob('stages/reconstruction/failure/*/batch.pt'))


def test_real_ipython_display_updates_before_run_returns(tmp_path):
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    shell=InteractiveShell.instance()
    s=MeshSession(config(updates=8),tmp_path/'run')
    with capture_output(display=True) as published:result=run(s,refresh_seconds=.05,plots=True)
    assert result['state']=='completed';htmls=[o.data.get('text/html','') for o in published.outputs];assert htmls and 'Initializing' in htmls[0]
    # IPython publishes display-id updates in the live primary output, not just final JSON.
    assert any('Successful' in text and 'Device cpu' in text for text in htmls)


def test_committed_store_status_reads_metadata_without_restore(tmp_path,monkeypatch):
    from pdeflow import persistence
    c=config(updates=2);s=MeshSession(c,tmp_path/'local',tmp_path/'store');s.run();before={str(p):p.stat().st_mtime_ns for p in s.persistent.rglob('*') if p.is_file()}
    monkeypatch.setattr(persistence,'restore',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('status restored data')))
    r=status(s.persistent);assert r['kind']=='snapshot' and 'remote activity unknown' in r['state'] and r['persistent']['snapshot'];assert r['successful_updates']==2
    after={str(p):p.stat().st_mtime_ns for p in s.persistent.rglob('*') if p.is_file()};assert before==after


def test_unnormalized_attention_value_accumulation_and_known_scalar_amp():
    from pdeflow.mesh.state import enforce
    c=config();m=SensorReader(c)
    with torch.no_grad():
        for parameter in m.parameters():parameter.zero_()
        m.v.bias.fill_(2000);m.out.weight.copy_(torch.eye(c.width))
    obs=(torch.zeros(1,64,3),torch.zeros(1,64,3),torch.ones(1,64,dtype=torch.bool),torch.zeros(1,64,dtype=torch.long))
    with torch.autocast('cpu',dtype=torch.float16):
        bad=torch.ones(1,1,65)@torch.full((1,65,c.width),2000.);out=m(torch.zeros(1,c.nodes,c.width),torch.zeros(c.nodes,2),obs)
        parameter=torch.tensor([.37]);state=enforce(c,torch.ones(1,c.nodes+1,2,dtype=torch.float16),parameter,torch.tensor([True]),torch.tensor([0]))
    assert not torch.isfinite(bad).all();torch.testing.assert_close(out.float(),torch.full_like(out.float(),2000));assert state.dtype==torch.float32 and state[0,-1,0]==parameter[0] and state[0,:c.nodes,1].eq(0).all()
