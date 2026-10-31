"""Anchor success budgets and real GradScaler control flow; injected overflow != T4 evidence."""
from dataclasses import replace
from pathlib import Path
import json,os
import pytest
import torch
from pdeflow import batch_experiments as experiments
from pdeflow.colab import training_preset
from pdeflow.coordinate.training import read_checkpoint
from pdeflow.coordinate.optimization import optimizer_step,gradient_diagnostics
from pdeflow.runtime import write_json,read_json


def tiny():
    torch.set_num_threads(2)
    return training_preset('smoke','cpu','fp32',{'encoder_points':16,'query_points':16,
        'field_loss':{'analysis_resolution':8,'region_queries':16,'time_groups':1},
        'training_options':{'autoencoder':{'warmup':3}}})


def job(c,root,attempts=2,**extra):
    return {'config':c.to_dict(),'stage':'autoencoder','root':str(root),'stop_file':str(root.parent/'stop'),
            'warmup':0,'attempts':attempts,**extra}


def inject_startup(monkeypatch,number):
    # Real CPU GradScaler unscale/found_inf/backoff/step; only the bad gradients
    # and scaler device are controlled. Model forward is CPU FP32, not CUDA FP16.
    import pdeflow.coordinate.optimization as module
    factory=torch.amp.GradScaler;calls=[]
    monkeypatch.setattr(torch.amp,'GradScaler',lambda *a,**kw:factory('cpu',enabled=True,init_scale=65536.))
    def controlled(optimizer,scaler,parameters,clip,policy):
        before=scaler.get_scale()
        if len(calls)<number:next(p for p in parameters if p.grad is not None).grad.flatten()[0]=float('inf')
        result=optimizer_step(optimizer,scaler,parameters,clip,'fp16')
        calls.append((before,scaler.get_scale(),result[2]));return result
    monkeypatch.setattr(module,'optimizer_step',controlled)
    return calls


def save_evidence(name,value):
    if os.environ.get('PDEFLOW_ANCHOR_EVIDENCE'):
        p=Path(os.environ['PDEFLOW_ANCHOR_EVIDENCE']);p.mkdir(parents=True,exist_ok=True);write_json(p/name,value)


def test_original_two_attempt_stop_reproduced_and_bounded_anchor_recovers(tmp_path,monkeypatch):
    c=tiny();calls=inject_startup(monkeypatch,2)
    experiments.worker(job(c,tmp_path/'original'))
    old=read_json(tmp_path/'original/result.json')
    assert old['status']=='insufficient_measurements' and old['successful_updates']==0
    assert old['attempted_updates']==2 and old['scales']==[32768.,16384.]
    assert not (tmp_path/'original/autoencoder.pt').exists()
    assert old['numerically_finite'] is False and old['overflow_skips_handled'] is True
    calls.clear()
    experiments.worker(job(c,tmp_path/'fixed',2*8,role='anchor',successful_updates=2))
    result=read_json(tmp_path/'fixed/result.json');cp,_=read_checkpoint(tmp_path/'fixed/resume.pt')
    assert result['status']=='completed' and result['attempted_updates']==4 and result['successful_updates']==2
    assert result['scales']==[32768.,16384.,16384.,16384.]
    assert len(calls)==4 and result['nonfinite_gradient_attempts']==2
    assert cp['scheduler']['completed_updates']==2 and cp['scaler']['scale']==16384.
    assert [r['step'] for r in cp['history']]==[0,0,1,2]
    rates=[r['learning_rate'] for r in cp['history']];assert rates[0]==rates[1]==rates[2] and rates[3]>rates[2]
    assert result['history'][0]['overflow_diagnostics']['nonfinite_parameter_count']==1
    save_evidence('startup-reproduction-and-recovery.json',{'scope':'CPU FP32 network, real CPU GradScaler, two injected overflows; not native CUDA/T4 startup evidence','original':old,'fixed':result})


def test_persistent_invalid_gradients_fail_bounded_with_traceback(tmp_path,monkeypatch):
    c=tiny();calls=inject_startup(monkeypatch,100)
    experiments.worker(job(c,tmp_path/'persistent',16,role='anchor',successful_updates=2))
    r=read_json(tmp_path/'persistent/result.json');cp,_=read_checkpoint(tmp_path/'persistent/resume.pt')
    assert r['status']=='error' and r['error_type']=='FloatingPointError'
    assert cp['progress']==0 and cp['attempted_updates']==8 and len(calls)==8
    assert cp['scaler']['scale']==256. and cp['scheduler']['completed_updates']==0
    assert 'Repeated FP16 overflow' in r['error'] and 'engine.py' in r['traceback']
    assert Path(r['traceback_path']).read_text()==r['traceback']
    assert r['history'][-1]['overflow_diagnostics']['nonfinite_parameter_count']==1
    save_evidence('persistent-overflow.json',r)


def test_anchor_cancellation_and_runtime_cap(tmp_path):
    c=tiny();(tmp_path/'stop').touch()
    experiments.worker(job(c,tmp_path/'cancelled',16,role='anchor',successful_updates=2))
    r=read_json(tmp_path/'cancelled/result.json');assert r['status']=='insufficient_successful_updates'
    cp,_=read_checkpoint(tmp_path/'cancelled/resume.pt');assert cp['attempted_updates']==0
    (tmp_path/'stop').unlink()
    experiments.worker(job(c,tmp_path/'expired',16,role='anchor',successful_updates=2,wall_seconds=0))
    cp,_=read_checkpoint(tmp_path/'expired/resume.pt');assert cp['attempted_updates']==0


def test_optimizer_invalid_gradients_are_not_clipped_or_updated():
    p=torch.nn.Parameter(torch.ones(2));opt=torch.optim.Adam([p],lr=.1);scaler=torch.amp.GradScaler('cpu')
    scaler.scale(p.square().sum()).backward();p.grad[0]=float('inf')
    norm,finite,skipped=optimizer_step(opt,scaler,[p],1.,'fp16')
    assert not finite and skipped and torch.isinf(norm)
    torch.testing.assert_close(p,torch.ones(2));assert not opt.state
    # No inf*0 conversion into NaNs destroys the location of the first invalid values.
    assert torch.isinf(p.grad[0]) and torch.isfinite(p.grad[1])
    assert gradient_diagnostics([('weight',p)])['first_nonfinite_parameters'][0]['nonfinite_elements']==1


def test_finite_elements_but_norm_overflow_never_step():
    p=torch.nn.Parameter(torch.ones(2));opt=torch.optim.Adam([p]);scaler=torch.amp.GradScaler('cpu',enabled=False)
    p.grad=torch.full_like(p,3e38)
    with pytest.raises(FloatingPointError,match='finite gradient elements'):optimizer_step(opt,scaler,[p],1.,'fp16')
    torch.testing.assert_close(p,torch.ones(2));assert not opt.state


def test_parent_failure_retains_worker_trace_and_blocks_pending(tmp_path,monkeypatch):
    c=tiny();plan=experiments.plan(c,stages=['evolution'],candidates=[1],cache_conditions=['cold'])
    root=tmp_path/'experiment';root.mkdir();write_json(root/'experiment.json',{'plan':plan,'rows':plan['rows']})
    class Child:
        returncode=0
        def __init__(self,args,**kwargs):
            j=read_json(Path(args[-1]));assert j['role']=='anchor' and j['attempts']==16
            write_json(Path(j['root'])/'result.json',{'status':'error','error':'Persistent nonfinite gradients','traceback':'saved worker traceback','successful_updates':0,'attempted_updates':8})
        def poll(self):return 0
        def wait(self):return 0
    monkeypatch.setattr(experiments.subprocess,'Popen',Child)
    experiments.run(root);r=read_json(root/'experiment.json')
    assert r['state']=='failed' and all(row['status']=='blocked' for row in r['rows'])
    assert r['anchors']['autoencoder']['traceback']=='saved worker traceback'
    details=experiments.failure_details(r)
    assert all(s in details for s in ('Persistent nonfinite gradients','saved worker traceback','successful=0','ValueError','process.log'))
    assert (root/'error-traceback.txt').read_text()==r['traceback']
    save_evidence('parent-failure-report.json',r)


def test_setup_errors_and_notebook_display_expose_traceback(tmp_path):
    j=job(tiny(),tmp_path/'bad');j['config']['family_weights']={'invalid':1.}
    experiments.worker(j);r=read_json(tmp_path/'bad/result.json')
    assert r['status']=='error' and r['traceback'] and r['error_type']=='ValueError'
    notebook=json.loads(Path('notebooks/colab.ipynb').read_text())
    cell=next(''.join(c['source']) for c in notebook['cells'] if 'failure_details' in ''.join(c['source']))
    assert 'print(batch_experiments.failure_details(batch_result))' in cell


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Actual CUDA FP16 requires CUDA hardware')
def test_native_cuda_fp16_anchor(tmp_path):
    c=training_preset('smoke','cuda','fp16',{'encoder_points':32,'query_points':32,
        'field_loss':{'analysis_resolution':8,'region_queries':32,'time_groups':1}})
    experiments.worker(job(c,tmp_path/'cuda',16,role='anchor',successful_updates=2))
    r=read_json(tmp_path/'cuda/result.json')
    assert r['status']=='completed',r
    assert r['successful_updates']==2 and all(row['gradients_finite'] for row in r['history'] if not row['optimizer_update_skipped'])


def test_actual_parent_anchor_exports_and_frozen_stage_trials(tmp_path):
    c=tiny();result=experiments.start(c,tmp_path/'real-experiment',background=False,
        stages=['evolution','flow'],candidates=[1],cache_conditions=['cold'],warmup=1,measured=5,max_seconds=90)
    assert result['state']=='completed',experiments.failure_details(result)
    assert set(result['anchors'])=={'autoencoder','evolution'}
    for anchor in result['anchors'].values():
        assert anchor['status']=='completed' and anchor['successful_updates']==2 and anchor['attempted_updates']==2
        assert anchor['numerically_finite'] is True
    assert all(r['status']=='completed' and r['successful_updates']==6 for r in result['rows'])
    save_evidence('real-subprocess-anchors-and-frozen-trials.json',result)


def test_resume_restores_real_scaler_after_skipped_anchor_attempts(tmp_path,monkeypatch):
    from pdeflow.coordinate.training import train
    c=tiny();calls=inject_startup(monkeypatch,2)
    experiments.worker(job(c,tmp_path/'resume',2))
    before,_=read_checkpoint(tmp_path/'resume/resume.pt')
    assert before['scaler']['scale']==16384. and before['progress']==0
    train(c.to_dict(),tmp_path/'resume','autoencoder',resume=True)
    after,_=read_checkpoint(tmp_path/'resume/resume.pt')
    assert after['progress']==2 and after['attempted_updates']==4
    assert calls[2][0]==16384. and after['scaler']['scale']==16384.
    assert after['scheduler']['completed_updates']==2
    save_evidence('real-cpu-scaler-resume.json',{'before':{'scaler':before['scaler'],'progress':before['progress']},'after':{'scaler':after['scaler'],'progress':after['progress'],'attempted_updates':after['attempted_updates']},'scope':'CPU scaler with injected overflows, FP32 neural execution'})
