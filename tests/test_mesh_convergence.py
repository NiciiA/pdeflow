from dataclasses import replace
from pathlib import Path
import copy,json,math,random
import numpy as np
import pytest
import torch
from pdeflow.mesh.contracts import preset,MeshConfig
from pdeflow.mesh.convergence import family_contributions,training_summary,rolling_series,attach
from pdeflow.mesh.monitoring import status,html_summary,text_summary
from pdeflow.mesh.data import Data
from pdeflow.mesh.model import MeshModel
from pdeflow.mesh.validation import validate,fixed_batches
from pdeflow.mesh.session import MeshSession
from pdeflow.mesh.provenance import read_checkpoint
from pdeflow.runtime import read_json,file_digest


def small(name='mesh_m2_v1',**kw):
    torch.set_num_threads(2)
    return preset('pipeline-check',name,'cpu','fp32',{'data':{'query_points':8},**kw})


def test_default_and_recorded_cadence_and_protocol():
    c=small();assert c.validation_every==250 and c.validation_protocol=='balanced_v2'
    raw=c.to_dict();raw.pop('validation_protocol');raw.pop('training_smoothing_window');raw['validation_every']=500
    recorded=MeshConfig.from_record(raw);assert recorded.validation_every==500 and recorded.validation_protocol=='legacy_v1'
    from pdeflow.mesh.workflow import plan
    for budget in ('pipeline-check','development','extended'):
        p=plan({'budget':budget,'device':'cpu'});assert p['config']['validation_every']==250
        assert p['config']['checkpoint_every']==min(50,next(iter(p['updates'].values())))


def test_actual_objective_family_sums_unequal_microbatches_and_success_window():
    totals=None
    values=[1.,3.,9.,5.,7.];families=[0,1,1,0,1]
    for start in range(0,5,2):totals=family_contributions(values[start:start+2],families[start:start+2],totals)
    assert totals=={'heat':{'sum':6.,'examples':2},'wave':{'sum':19.,'examples':3}}
    rows=[{'step':1,'training_loss':5.,'family_objective':totals},
          {'step':1,'training_loss':1000.,'optimizer_update_skipped':True,'family_objective':totals},
          {'step':2,'training_loss':3.,'family_objective':{'heat':{'sum':6.,'examples':2},'wave':{'sum':0.,'examples':0}}},
          {'step':2,'training_loss':float('inf'),'optimizer_update_skipped':True},
          {'step':3,'training_loss':7.,'family_objective':totals}]
    summary=training_summary(rows,2);assert summary['mean']==5 and summary['count']==2
    assert summary['family_objective']['heat']['mean']==3 and summary['family_objective']['heat']['examples']==4
    assert summary['family_objective']['wave']['mean']==pytest.approx(19/3)
    assert rolling_series(rows,2)==[{'step':1,'mean':5.,'count':1},{'step':2,'mean':4.,'count':2},{'step':3,'mean':5.,'count':2}]
    old=training_summary([{'step':1,'training_loss':2,'family_example_counts':{'heat':5}}]);assert old['family_objective']['heat']['mean'] is None
    assert training_summary(rows[:1],200)['count']==1


def test_fixed_balanced_validation_fields_noise_rng_modes_and_physical_units(tmp_path,monkeypatch):
    from pdeflow.mesh import validation
    c=small('mesh_m3_fgn_v1',batch_size=1,microbatch=1);d=Data(c,tmp_path/'data');m=MeshModel(c)
    m.train();m.decoder.eval();m.evolution.requires_grad_(False)
    states={k:v.clone() for k,v in m.state_dict().items()};modes=[x.training for x in m.modules()];metadata=copy.deepcopy(d.dataset.metadata)
    torch.manual_seed(912);random.seed(44);np.random.seed(53)
    cpu=torch.get_rng_state().clone();py=random.getstate();npstate=np.random.get_state();calls=[];original=validation.generate
    def observed(*args,**kwargs):
        out=original(*args,**kwargs);calls.append(out[0].clone());return out
    monkeypatch.setattr(validation,'generate',observed)
    a=validate(m,d,'stochastic',preview_root=tmp_path/'preview',update=0);b=validate(m,d,'stochastic',preview_root=tmp_path/'preview',update=1)
    assert torch.equal(cpu,torch.get_rng_state()) and random.getstate()==py
    assert np.array_equal(npstate[1],np.random.get_state()[1]) and [x.training for x in m.modules()]==modes
    assert d.dataset.metadata==metadata and all(torch.equal(v,states[k]) for k,v in m.state_dict().items())
    assert a['validation_spec_sha256']==b['validation_spec_sha256'] and a['records']==b['records']
    assert all(torch.equal(calls[i],calls[i+len(calls)//2]) for i in range(len(calls)//2))
    assert set(a['groups'])=={'heat/u','wave/u','wave/v'} and a['case_count']==4 and a['generated_samples']==4
    for g in a['groups'].values():
        assert g['case_count']==2 and g['evaluated_points']==2*2*8
        assert g['physical_mse']==pytest.approx(g['normalized_mse']*g['normalization_scale']**2)
        assert g['physical_rmse']==pytest.approx(math.sqrt(g['physical_mse']))
    expected=.5*a['groups']['heat/u']['normalized_mse']+.25*(a['groups']['wave/u']['normalized_mse']+a['groups']['wave/v']['normalized_mse'])
    assert a['balanced_family_normalized_mse']==pytest.approx(expected)
    spec_path=tmp_path/'preview/validation-spec.json';spec=read_json(spec_path);assert len(spec['batches'])==2
    before_spec=spec_path.read_bytes()
    with pytest.raises(ValueError,match='specification changed'):validate(m,d,'stochastic',cases=1,preview_root=tmp_path/'preview',update=2)
    assert spec_path.read_bytes()==before_spec
    for batch in spec['batches']:
        assert batch['evaluation_seed'] and batch['observations']['coordinates_and_measurement_times']
        assert batch['known_parameters'] and batch['initial_queries'] and batch['physical_target_times']
        assert all(x['reference_initial_sha256'] and x['seed'] for x in batch['cases'])


def test_real_success_schedule_resume_no_duplicates_and_full_history(tmp_path,monkeypatch):
    from pdeflow.mesh import training
    c=small('mesh_m1_v1',budgets={s:5 for s in small().budgets},validation_every=2,checkpoint_every=2,training_smoothing_window=3)
    original=training.optimizer_step;attempts=[];events=[];evaluations=[];v=training.validate
    def skipped(*a,**kw):
        attempts.append(1)
        return (torch.tensor(float('inf')),False,True) if len(attempts) in (1,3) else original(*a,**kw)
    def evaluated(*a,**kw):evaluations.append(kw['update']);return v(*a,**kw)
    monkeypatch.setattr(training,'optimizer_step',skipped);monkeypatch.setattr(training,'validate',evaluated)
    s=MeshSession(c,tmp_path/'run',save_updates=2)
    assert s.run(stop_after=3,progress=lambda r:events.append(copy.deepcopy(r)))['state']=='interrupted'
    assert evaluations==[0,2]
    cp,_=read_checkpoint(s.root/'stages/reconstruction/resume.pt');assert cp['successful_updates']==3 and cp['skipped_updates']==2
    assert s.run()['state']=='completed' and evaluations==[0,2,4,5]
    history=read_json(s.root/'stages/reconstruction/validation-history.json');assert [r['update'] for r in history]==[0,2,4,5]
    assert len({r['validation_id'] for r in history})==4
    assert len({r['validation_spec_sha256'] for r in history})==1
    assert any(len(r.get('validation_history',[]))>=2 and r['successful_updates']<5 for r in events)
    record=status(s.root);assert record['training_summary']['count']==3
    values=[r['training_loss'] for r in read_checkpoint(s.root/'stages/reconstruction/resume.pt')[0]['history'] if not r['optimizer_update_skipped']][-3:]
    assert record['training_summary']['mean']==pytest.approx(sum(values)/3)
    document=html_summary(record);assert 'validation-marker' in document and 'Training objective' in document and 'wave/v' in document
    assert [r['update'] for r in record['validation_history']]==[0,2,4,5]
    before={str(p):file_digest(p) for p in s.root.rglob('*') if p.is_file()};rng=torch.get_rng_state().clone()
    status(s.root);html_summary(status(s.root));assert torch.equal(rng,torch.get_rng_state())
    assert before=={str(p):file_digest(p) for p in s.root.rglob('*') if p.is_file()}
    assert s.run()['state']=='completed' and evaluations==[0,2,4,5]
    with pytest.raises(ValueError,match='validation'):MeshSession(replace(c,validation_every=3),s.root).run()


def test_legacy_missing_breakdowns_and_full_stage_validation_outside_recent_window():
    legacy={'stage':'flow','history':[{'step':600,'training_loss':2.,'validation_loss':.4}],'validation_every':500,'successful_updates':600,'budget':10000}
    result=attach(legacy,[{'stage':'flow','update':0,'uniform_domain_normalized_mse':.7,'groups':{}},{'stage':'flow','update':500,'uniform_domain_normalized_mse':.4,'groups':{}}])
    document=html_summary(result);assert 'validation-marker' in document and '>0</text>' in document
    assert result['best_validation']['update']==500 and result['training_summary']['family_objective']['wave']['mean'] is None
    empty=html_summary({'stage':'flow','history':[],'validation_every':250,'successful_updates':101,'budget':10000})
    assert 'No validation has been recorded' in empty and '250' in empty and 'validation-marker' not in empty


def test_live_ipython_markers_before_training_returns(tmp_path,monkeypatch):
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    from pdeflow.mesh.monitoring import run
    from pdeflow.mesh import training
    import time
    original=training.loss
    def slow(*a,**kw):time.sleep(.06);return original(*a,**kw)
    monkeypatch.setattr(training,'loss',slow)
    was_initialized=InteractiveShell.initialized();InteractiveShell.instance();s=MeshSession(small('mesh_m1_v1',validation_every=1),tmp_path/'live')
    with capture_output(display=True) as published:result=run(s,refresh_seconds=.05,plots=True)
    assert result['state']=='completed'
    htmls=[r.data.get('text/html','') for r in published.outputs]
    if not was_initialized:InteractiveShell.clear_instance()
    assert 'Initializing' in htmls[0]
    assert any('validation-marker' in h and '· training' in h and 'Latest fixed validation update 1' in h for h in htmls)


def test_validation_frequency_and_metrics_leave_real_training_identical(tmp_path):
    from pdeflow.mesh.training import train_stage
    c=small('mesh_m2_v1',batch_size=5,microbatch=2,budgets={s:3 for s in small().budgets})
    checkpoints=[]
    for every in (1,250):
        root=tmp_path/str(every);train_stage(replace(c,validation_every=every),root,'reconstruction')
        checkpoints.append(read_checkpoint(root/'stages/reconstruction/resume.pt')[0])
    a,b=checkpoints
    assert all(torch.equal(v,b['model'][k]) for k,v in a['model'].items())
    assert torch.equal(a['generator'],b['generator']) and torch.equal(a['torch_rng'],b['torch_rng'])
    assert [r['training_loss'] for r in a['history']]==[r['training_loss'] for r in b['history']]
    for row in a['history']:
        parts=row['family_objective'];assert sum(p['examples'] for p in parts.values())==5
        assert sum(p['sum'] for p in parts.values())/5==pytest.approx(row['training_loss'],rel=2e-6)
    assert [r['update'] for r in a['validation_history']]==[0,1,2,3]
    assert [r['update'] for r in b['validation_history']]==[0,3]


def test_suite_metrics_stay_on_the_actual_candidate_and_stage(tmp_path,monkeypatch):
    from pdeflow.mesh.suite import Suite,suite_plan
    from pdeflow.mesh import monitoring,training
    import time
    plan=suite_plan('pipeline-check','cpu','fp32',candidates=['mesh_m1_v1','mesh_m3_fgn_v1'],overrides={'data':{'query_points':8},'validation_every':1})
    suite=Suite.new(plan,tmp_path/'suite');original=training.loss;seen=[];real_status=monitoring.status
    def slow(*a,**kw):time.sleep(.03);return original(*a,**kw)
    def observed(root):
        r=real_status(root)
        if Path(root)==suite.root and r.get('active_model',{}).get('validation_history'):seen.append(copy.deepcopy(r))
        return r
    monkeypatch.setattr(training,'loss',slow);monkeypatch.setattr(monitoring,'status',observed)
    assert monitoring.run(suite,refresh_seconds=.05,plots=False)['state']=='completed'
    assert {r['active'] for r in seen}=={'mesh_m1_v1','mesh_m3_fgn_v1'}
    for r in seen:
        active=r['active_model'];assert active['architecture']==r['active']
        assert all(v['architecture']==r['active'] and v['stage']==active['stage'] for v in active['validation_history'])
    for architecture,config in plan['configs'].items():
        c=MeshConfig(**config)
        for stage in c.kind['stages']:
            rows=read_json(suite.root/'runs'/architecture/'stages'/stage/'validation-history.json')
            assert all(r['architecture']==architecture and r['stage']==stage for r in rows)


def test_old_saved_validation_without_update_is_shown_without_invented_marker(tmp_path):
    from pdeflow.runtime import write_json
    write_json(tmp_path/'session.json',{'stage':'reconstruction','state':'interrupted','config':{'validation_every':500}})
    path=tmp_path/'stages/reconstruction';path.mkdir(parents=True)
    write_json(path/'run.json',{'stage':'reconstruction','progress':321,'history':[{'step':321,'training_loss':2.}]})
    write_json(path/'validation.json',{'uniform_domain_normalized_mse':.7,'records':[]})
    before={str(p):file_digest(p) for p in tmp_path.rglob('*') if p.is_file()}
    record=status(tmp_path);assert record['latest_validation']['uniform_domain_normalized_mse']==.7
    assert record['latest_validation'].get('update') is None
    document=html_summary(record);assert 'evaluation update is unavailable' in document and 'validation-marker' not in document
    assert 'not available' in text_summary(record)
    assert before=={str(p):file_digest(p) for p in tmp_path.rglob('*') if p.is_file()}
