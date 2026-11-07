from dataclasses import replace
import pytest
import torch
from pdeflow.mesh.contracts import preset
from pdeflow.mesh.settings import configure,resolved,summary
from pdeflow.mesh.session import MeshSession
from pdeflow.mesh.provenance import read_checkpoint
from pdeflow.mesh.benchmark import profile,microbatch_benchmark
from pdeflow.runtime import file_digest


def small(name='mesh_m1_v1',**overrides):
    torch.set_num_threads(2)
    return preset('pipeline-check',name,'cpu','fp32',{'data':{'query_points':8},**overrides})


def test_explicit_policies_learning_rate_and_rounding():
    base=small();base=replace(base,budgets={s:11 for s in base.budgets})
    keep=configure(base,learning_rate=1e-4,effective_batch=3,microbatch=2)
    assert keep.budgets==base.budgets and keep.learning_rate==1e-4
    r=resolved(keep,'reconstruction');assert r['accumulation']==2 and r['last_microbatch']==1
    examples=configure(base,effective_batch=3,microbatch=2,budget_policy='keep_examples')
    assert examples.budgets['reconstruction']==8 and examples.learning_rate==base.learning_rate
    assert examples.batch_reference['reconstruction']['rounding_extra_examples']==2
    warm=replace(base,optimizer={'schedule':'cosine','warmup':8})
    with pytest.raises(ValueError,match='warmup'):configure(warm,effective_batch=8,microbatch=8,budget_policy='keep_examples')
    for changes in ({'learning_rate':0},{'microbatch':3},{'microbatch':0},{'effective_batch':0},{'stages':{'bogus':{}}},{'budget_policy':'auto'},{'stages':{'reconstruction':{'optimizer':{'name':'SGD'}}}}):
        with pytest.raises(ValueError):configure(base,**changes)


def test_real_stage_override_remainder_resume_and_immutability(tmp_path):
    c=configure(small(),learning_rate=1e-4,stages={'reconstruction':{'learning_rate':2e-4,'batch_size':3,'microbatch':2}})
    s=MeshSession(c,tmp_path/'run');assert s.run(stop_after=1)['state']=='interrupted'
    path=s.root/'stages/reconstruction/resume.pt';cp,_=read_checkpoint(path);sha=file_digest(path)
    r=cp['resolved_training_settings'];assert r['learning_rate']==2e-4 and r['effective_batch']==3 and r['accumulation']==2 and r['last_microbatch']==1
    assert cp['optimizer']['param_groups'][0]['lr']==2e-4
    assert cp['history'][-1]['actual_forward_microbatch']==2 and cp['effective_examples']==3
    for settings in ({'learning_rate':1e-4},{'effective_batch':4}):
        with pytest.raises(ValueError,match='NEW experiment'):s.configure_training(**settings)
    changed=replace(c,stage_settings={'reconstruction':{'learning_rate':1e-4,'batch_size':3,'microbatch':2}})
    with pytest.raises(ValueError,match='Incompatible'):MeshSession(changed,s.root).run()
    assert file_digest(path)==sha
    assert s.run()['state']=='completed'


def test_draft_settings_invalidate_preflight_without_starting(tmp_path):
    s=MeshSession(small(),tmp_path/'draft');(s.root/'profile.json').write_text('{}')
    report=s.configure_training(learning_rate=1e-4,effective_batch=4,microbatch=3,budget_policy='keep_examples')
    assert not (s.root/'profile.json').exists() and not (s.root/'session.json').exists()
    assert report['stages']['reconstruction']['learning_rate']==1e-4
    assert s.c.budgets['reconstruction']==1


def test_real_fgn_sweep_preserves_workload_schedule_and_members(tmp_path):
    c=small('mesh_m3_fgn_v1');c=replace(c,budgets={s:20 for s in c.budgets},optimizer={'schedule':'cosine','warmup':10})
    before=c.to_dict();rng=torch.get_rng_state().clone()
    result=microbatch_benchmark(c,tmp_path,[1,2],updates=7,warmup=2)
    assert c.to_dict()==before and torch.equal(rng,torch.get_rng_state())
    for candidate in result['candidates']:
        assert candidate['state']=='completed',candidate.get('traceback')
        report=candidate['report'];assert report['success']
        for stage,r in report['stages'].items():
            settings=r['resolved_training_settings'];assert settings['effective_batch']==2 and settings['schedule_horizon']==20 and settings['optimizer']['warmup']==10
            assert settings['learning_rate']==c.learning_rate and r['successful_updates']==7 and r['skipped_updates']==0
            assert r['measured_successful_updates']==5 and r['examples_per_second']>0 and r['estimated_total_training_seconds']>0
            assert r['memory']['cuda_peak_allocated'] is None # CPU verification, not CUDA
            assert r['fgn_members_per_example']==(2 if stage=='stochastic' else 0)
            assert r['warmup_rows'][0]['learning_rate']==pytest.approx(c.learning_rate/10)
    for stage in c.kind['stages']:
        a=result['candidates'][0]['report']['stages'][stage];b=result['candidates'][1]['report']['stages'][stage]
        assert a['warmup_rows'][0]['physical_target_times']==b['warmup_rows'][0]['physical_target_times']
        assert a['warmup_rows'][0]['training_loss']==pytest.approx(b['warmup_rows'][0]['training_loss'],rel=2e-5)
    assert list(tmp_path.glob('microbatch-benchmark-*/report.json'))
    with pytest.raises(ValueError,match='effective batch'):microbatch_benchmark(c,tmp_path,[4])


def test_fgn_actual_stage_batch_validation_and_monitor(tmp_path):
    from pdeflow.mesh.training import train_stage
    from pdeflow.mesh.monitoring import status
    # Legacy validation retains its stage-dependent batch cohort; balanced v2 has independent coverage tests.
    c=configure(small('mesh_m3_fgn_v1',validation_protocol='legacy_v1'),learning_rate=1e-4,stages={
        'central':{'learning_rate':2e-4,'batch_size':3,'microbatch':2},
        'stochastic':{'batch_size':4,'microbatch':3}})
    parent=None
    for stage in c.kind['stages']:
        record=train_stage(c,tmp_path,stage,parent)
        assert record['state']=='completed'
        cp,_=read_checkpoint(tmp_path/'stages'/stage/'resume.pt')
        settings=resolved(c,stage);assert cp['effective_examples']==2*settings['effective_batch']
        assert cp['optimizer']['param_groups'][0]['lr']==settings['learning_rate']
        progress=status(tmp_path)
        assert progress['batch_size']==settings['effective_batch'] and progress['microbatch']==settings['microbatch'] and progress['accumulation']==settings['accumulation']
        assert cp['validation']['records'] and len(cp['validation']['records'])==c.validation_cases*settings['effective_batch']
        parent=tmp_path/'stages'/stage/'resume.pt'


def test_empty_ui_optimizer_controls_are_rejected_not_defaulted():
    from pdeflow.mesh.workflow import plan
    for key in ('learning_rate','batch_size','microbatch'):
        with pytest.raises(ValueError,match='cannot be empty'):
            plan({'problem':'heat-wave-2d','model':'mesh_m1_v1','budget':'pipeline-check','device':'cpu','advanced':{key:None}})
