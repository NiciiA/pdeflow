"""Configure → disposable benchmark → APPLY → real training, including counters."""
from dataclasses import replace
from pathlib import Path
import json,os
import pytest
import torch
from fastapi.testclient import TestClient
from pdeflow.colab import Session,training_preset,apply_training_settings
from pdeflow.training_settings import resolve_batches,configure,batch_identity
from pdeflow.coordinate.optimization import options
from pdeflow.coordinate.training import train,read_checkpoint
from pdeflow.runtime import write_json,read_json,digest
from pdeflow import batch_experiments as experiments


def tiny():
    torch.set_num_threads(2)
    return training_preset('smoke','cpu','fp32',{'encoder_points':16,'query_points':16,'field_loss':{'analysis_resolution':8,'region_queries':16,'time_groups':1}})


def test_resolve_large_auto_overrides_and_remainder():
    c,_=configure(tiny(),{'effective_batch':1024,'microbatch':'auto','stages':{'A1':32,'A2':64,'B':128}})
    for stage,micro in [('autoencoder',32),('evolution',64),('flow',128)]:
        r=resolve_batches(c)[stage];assert r['microbatch']==micro and r['accumulation']==1024//micro;assert options(c,stage)['microbatch']==micro
    c,_=configure(c,{'effective_batch':7,'microbatch':3,'stages':{}});r=resolve_batches(c)['autoencoder'];assert (r['accumulation'],r['last_microbatch'])==(3,1)
    for setting in ({'effective_batch':0},{'effective_batch':True},{'effective_batch':8,'microbatch':16},{'microbatch':0},{'microbatch':'unknown'},{'stages':{'foo':4}}):
        with pytest.raises(ValueError):configure(tiny(),setting)


def test_budget_conversion_and_lr():
    c=replace(tiny(),ae_steps=10000,evolution_steps=10000,flow_steps=10000,batch_size=8)
    new,r=configure(c,{'effective_batch':1024,'microbatch':8},'keep_examples');assert new.ae_steps==79 and r['rounding']['autoencoder']['rounding_extra_examples']==896
    assert new.learning_rate==c.learning_rate
    unchanged,_=configure(c,{'effective_batch':1024,'microbatch':8},'keep_updates');assert unchanged.ae_steps*unchanged.batch_size==10_240_000


def test_application_replaces_real_session_preserves_identity_invalidates_preflight_and_stale_reference(tmp_path):
    c=tiny();original=Session(c,'same-name',tmp_path/'local',tmp_path/'persistent',True,sync_seconds=12,keep_snapshots=2)
    write_json(original.root/'profile.json',{'success':True,'old':'resource estimate'})
    new=apply_training_settings(original,{'effective_batch':5,'microbatch':2,'stages':{'A2':1}})
    assert new is not original and original.c.batch_size==2 and new.c.batch_size==5
    assert (new.name,new.root,new.persistent,new.sync_seconds,new.keep_snapshots)==(original.name,original.root,original.persistent,12,2)
    for key in ('width','tokens','layers','distribution','field_loss','family_weights','seed','learning_rate'):assert getattr(new.c,key)==getattr(c,key)
    assert not (new.root/'profile.json').exists() and list((new.root/'settings-history').glob('*profile.json'))
    assert read_json(new.root/'resolved-config.json')['batch_size']==5
    with pytest.raises(ValueError,match='stale settings'):original.run(stages='representation')
    result=new.run(stages='representation');cp,_=read_checkpoint(new.root/'autoencoder/autoencoder.pt');assert result['state']=='completed'
    assert cp['batch_settings']['microbatch']==2 and cp['config']['batch_size']==5
    assert cp['history'][0]['forward_microbatch_sizes']==[2,2,1]
    evidence=Path(os.environ.get('PDEFLOW_BATCH_SETTINGS_EVIDENCE',tmp_path/'evidence'));evidence.mkdir(parents=True,exist_ok=True)
    write_json(evidence/'applied-real-run.json',{'name':new.name,'path':str(new.root),'batch_settings':cp['batch_settings'],'history':cp['history'],'original_settings':batch_identity(c),'result':'Actual CPU encoder/decoder optimizer updates. No quality/gate improvement claim.'})


def test_active_run_application_rejected(tmp_path):
    from pdeflow.local import run_lock
    s=Session(tiny(),'owned',tmp_path/'local',tmp_path/'persistent',True)
    with run_lock(s.root):
        with pytest.raises(ValueError,match='active'):apply_training_settings(s,{'microbatch':1})


def test_benchmark_rng_isolation_apply_recommendation_and_actual_run(tmp_path):
    s=Session(tiny(),'trial',tmp_path/'local',tmp_path/'persistent',True);torch.manual_seed(913);before=torch.get_rng_state().clone();original=s.c.to_dict()
    r=s.benchmark_batch_sizes(background=False,stages=['autoencoder'],candidates=[1,2],effective_batch=2,cache_conditions=['cold'],warmup=1,measured=5,max_seconds=90)
    assert r['state']=='completed',r
    assert torch.equal(before,torch.get_rng_state()) and s.c.to_dict()==original
    assert not (s.root/'autoencoder').exists() and all(row['status']=='completed' for row in r['rows'])
    assert 'autoencoder' in r['recommendation']
    applied=s.apply_batch_result(r['id']);assert applied.name==s.name and applied.root==s.root
    result=applied.run(stages='representation');assert result['state']=='completed'
    cp,_=read_checkpoint(applied.root/'autoencoder/autoencoder.pt');assert cp['batch_settings']['microbatch']==r['recommendation']['autoencoder']['microbatch']
    evidence=Path(os.environ.get('PDEFLOW_BATCH_SETTINGS_EVIDENCE',tmp_path/'evidence'));evidence.mkdir(parents=True,exist_ok=True);write_json(evidence/'benchmark-apply-run.json',{'benchmark':r,'applied':read_json(applied.root/'applied-settings.json'),'actual_history':cp['history']})


def test_recommendations_exclude_oom_skips_low_headroom_and_robust_ties():
    base={'stage':'autoencoder','effective_batch':32,'status':'completed','numerically_finite':True,'std_seconds':.02,'mean_seconds':1.,'examples_per_second':100.,'peak_allocated':20,'gpu_total_bytes':100,'skipped_updates':0}
    rows=[{**base,'microbatch':8},{**base,'microbatch':16,'examples_per_second':103.,'peak_allocated':70},{**base,'microbatch':32,'examples_per_second':200.,'skipped_updates':1},{**base,'microbatch':64,'status':'oom'},{**base,'microbatch':128,'peak_allocated':90}]
    assert experiments.recommend(rows)['autoencoder']['microbatch']==8


def test_experiment_modes_candidates_and_budgets():
    c=tiny();p=experiments.plan(c,stages=['autoencoder'],candidates=[8,16,32,64,128,256,512,1024],effective_batch=1024)
    assert [r['accumulation'] for r in p['rows'][::2]]==[128,64,32,16,8,4,2,1]
    p=experiments.plan(c,mode='learning',stages=['autoencoder'],effective_batches=[2,4],budget_policy='examples',budget=28,warmup=2,measured=5)
    assert {r['planned_examples'] for r in p['rows']}=={28}
    with pytest.raises(ValueError):experiments.plan(c,candidates=[4],effective_batch=2)


def test_skip_counters_scheduler_resume_and_overflow_failure(tmp_path,monkeypatch):
    import pdeflow.coordinate.optimization as module
    c=tiny();method=module.optimizer_step;calls=[0]
    def skip_once(optimizer,scaler,parameters,clip,policy):
        calls[0]+=1
        if calls[0]==1:return torch.tensor(float('inf')),False,True
        return method(optimizer,scaler,parameters,clip,policy)
    monkeypatch.setattr(module,'optimizer_step',skip_once)
    train(c.to_dict(),tmp_path/'run','autoencoder',stop_after=1)
    cp,_=read_checkpoint(tmp_path/'run/resume.pt');assert (cp['progress'],cp['attempted_updates'],cp['effective_examples_processed'])==(0,1,2)
    assert cp['scheduler']['completed_updates']==0
    train(c.to_dict(),tmp_path/'run','autoencoder',resume=True)
    cp,_=read_checkpoint(tmp_path/'run/resume.pt');assert (cp['progress'],cp['attempted_updates'],cp['skipped_updates'])==(2,3,1)
    assert [r['learning_rate'] for r in cp['history']]==[c.learning_rate]*3
    monkeypatch.setattr(module,'optimizer_step',lambda *args:(torch.tensor(float('inf')),False,True))
    bad=replace(c,training_options={**c.training_options,'max_consecutive_overflows':2})
    with pytest.raises(FloatingPointError,match='Repeated FP16 overflow'):train(bad.to_dict(),tmp_path/'overflows','autoencoder')
    cp,_=read_checkpoint(tmp_path/'overflows/resume.pt');assert cp['attempted_updates']==2 and cp['progress']==0


def test_resume_batch_change_explicit_and_original_provenance(tmp_path):
    s=Session(tiny(),'continued',tmp_path/'local',tmp_path/'persistent',True)
    train(s.c.to_dict(),s.root/'autoencoder','autoencoder',stop_after=1)
    with pytest.raises(ValueError,match='explicit continuation'):apply_training_settings(s,{'microbatch':1})
    new=apply_training_settings(s,{'microbatch':1},continuation=True)
    train(new.c.to_dict(),new.root/'autoencoder','autoencoder',resume=True)
    cp,_=read_checkpoint(new.root/'autoencoder/resume.pt');assert cp['initial_config']['training_options']==s.c.training_options
    assert cp['continuation_history'] and cp['history'][-1]['actual_forward_microbatch']==1


def test_api_apply_preview_and_start_use_new_settings(tmp_path,monkeypatch):
    from pdeflow.workbench.api import create_app
    from pdeflow import training_workflow as workflow
    client=TestClient(create_app(workspace=tmp_path/'workspace',model_roots=[]));body={'problem':'heat-wave-2d','budget':'pipeline-check','device':'cpu','model':'resolution-flexible'}
    response=client.post('/api/training/batch/apply',json={'request':body,'settings':{'effective_batch':7,'microbatch':3,'stages':{'B':1}}});assert response.status_code==200,response.text
    value=response.json();assert value['resolved']['batch_settings']['flow']['accumulation']==7
    monkeypatch.setattr(workflow,'_spawn',lambda _:0)
    started=client.post('/api/training/start',json={**value['request'],'submission_token':'batch-applied'});assert started.status_code==202,started.text
    root=tmp_path/'workspace/training-workflows'/started.json()['id'];assert read_json(root/'resolved-config.json')['batch_size']==7


def test_worker_oom_real_errors_and_cancellation_are_distinct(tmp_path,monkeypatch):
    import pdeflow.coordinate.training as training
    c=tiny();job={'config':c.to_dict(),'root':str(tmp_path/'candidate'),'stage':'autoencoder','warmup':1,'attempts':6,'stop_file':str(tmp_path/'stop')}
    def oom(*args,**kwargs):raise torch.OutOfMemoryError('confirmed allocator failure')
    monkeypatch.setattr(training,'train',oom);experiments.worker(job);assert read_json(tmp_path/'candidate/result.json')['status']=='oom'
    def error(*args,**kwargs):raise RuntimeError('real numerical/kernel problem')
    monkeypatch.setattr(training,'train',error);experiments.worker(job);r=read_json(tmp_path/'candidate/result.json');assert r['status']=='error' and 'numerical' in r['error']
    root=tmp_path/'cancelled';root.mkdir();p=experiments.plan(c,stages=['autoencoder'],candidates=[1],cache_conditions=['cold'])
    write_json(root/'experiment.json',{'id':'cancelled','state':'queued','plan':p,'rows':p['rows']});experiments.stop(root);experiments.run(root)
    r=experiments.status(root);assert r['state']=='cancelled' and all(v['status']=='cancelled' for v in r['rows'])


def test_old_benchmark_cannot_apply_after_loss_change(tmp_path):
    c=tiny();record={'plan':{'science_identity':experiments.science_identity(c),'mode':'efficiency'},'recommendation':{'autoencoder':{'microbatch':1,'effective_batch':2}}}
    changed=replace(c,field_loss={**c.field_loss,'alpha':.75})
    with pytest.raises(ValueError,match='stale'):experiments.selected_settings(changed,record,recommended=True)


def test_bounded_effective_batch_learning_executes_same_example_budget(tmp_path):
    c=tiny();r=experiments.start(c,tmp_path/'learning',background=False,mode='learning',stages=['autoencoder'],effective_batches=[2,4],budget_policy='examples',budget=28,warmup=1,measured=5,cache_conditions=['cold'],max_seconds=90)
    assert r['state']=='completed',r
    assert r['recommendation']=={}
    assert {row['examples_processed'] for row in r['rows']}=={28}
    assert all('learning_validation' in row for row in r['rows'])


def test_learning_wall_budget_and_unequal_updates_are_explicit(tmp_path):
    c=tiny();p=experiments.plan(c,mode='learning',stages=['autoencoder'],effective_batches=[2,4],budget_policy='updates',budget=6,warmup=1,measured=5,cache_conditions=['cold'])
    assert {row['planned_examples'] for row in p['rows']}=={12,24}
    r=experiments.start(c,tmp_path/'wall',background=False,mode='learning',stages=['autoencoder'],effective_batches=[2],budget_policy='wall',budget=2.,warmup=1,measured=5,cache_conditions=['cold'],max_seconds=30)
    assert r['state']=='completed';row=r['rows'][0];assert row['status'] in ('completed','insufficient_measurements')
    assert 'learning_validation' in row and not r['recommendation']


def test_dead_benchmark_worker_keeps_partial_results(tmp_path,monkeypatch):
    root=tmp_path/'dead';root.mkdir();write_json(root/'experiment.json',{'state':'running','pid':999999,'rows':[{'status':'completed','stage':'autoencoder'},{'status':'running','stage':'evolution'}]})
    def dead(*args):raise ProcessLookupError()
    monkeypatch.setattr(os,'kill',dead);r=experiments.status(root);assert r['state']=='interrupted' and r['rows'][0]['status']=='completed'


def test_duration_estimate_uses_measured_baseline_not_memory():
    p=experiments.plan(tiny(),stages=['autoencoder'],candidates=[1,2],effective_batch=2,cache_conditions=['cold'],observed_seconds={'autoencoder':.2})
    assert p['estimate']['seconds']==pytest.approx(2.8) #2 candidates×7 attempts×.2s
    assert 'measured baseline' in p['estimate']['meaning']
