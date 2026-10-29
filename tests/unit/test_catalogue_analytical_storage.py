"""Behavioral contracts for catalogue completeness, formulas and safe retention."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import json,os,time
import numpy as np
import pytest
from pdeflow import local,suites,storage
from pdeflow.catalogue import catalogue,signature,coverage
from pdeflow.symbolic import analyze
from pdeflow.reference.examples import example
from pdeflow.reference.analytical import templates,verify,normalize_expression
from pdeflow.core import ProblemSpec,DomainSpec
from pdeflow.inference import compatibility_report
from pdeflow.runtime import write_json


def test_full_catalogue_is_inspectable_and_quickstart_is_distinct(tmp_path):
    rows=catalogue();assert len(rows)==11
    assert {r['id'] for r in rows}==set(suites.SUITES['full'])
    full=suites.plan(workspace=tmp_path);assert len(full['entries'])==11
    assert any(r['grid']==[512,512] for r in full['entries']) and any(r['grid']==[1024,1024] for r in full['entries'])
    assert sum(len(r['grid'])==3 for r in full['entries'])==3
    assert len(suites.plan('quickstart',workspace=tmp_path)['entries'])==6
    assert not tmp_path.joinpath('runs').exists()
    target=next(r for r in rows if r['id']=='heat-2d-512');assert target['architecture']['parameters']==73939
    assert target['grid']['times']==(0.,.05,.1) and target['data_mode']=='online'


@pytest.mark.parametrize('inline',[False,True])
def test_supported_chemistry_arrangements_and_different_kinetics(inline):
    r=example('chemistry-2d').request
    if inline:
        r['equations']=['0=a_τ-D_A*lap(a)+b*k_f*a-c*k_r','b_τ+k_f*a*b-k_r*c=D_B*lap(b)','c_τ-k_f*b*a+k_r*c=D_C*lap(c)']
        r['declarations']=[d for d in r['declarations'] if d['name']!='r']
    else:r['equations']=['0=a_τ-D_A*lap(a)+r','b_τ+r=D_B*lap(b)','D_C*lap(c)+r=c_τ','r=b*k_f*a-c*k_r']
    assert analyze(r)['recognized_family']=='chemistry' and analyze(r)['problem']
    r['equations']=[e.replace('k_f*a*b','k_f*a*a').replace('b*k_f*a','a*k_f*a').replace('k_f*b*a','k_f*a*a') for e in r['equations']]
    assert not analyze(r)['problem']


def test_structured_compatibility_does_not_require_one_model_per_condition():
    spec=example('chemistry-2d');p=ProblemSpec.from_dict(analyze(spec.request)['problem']);c=local.preset_config('chemistry-2d')
    r=compatibility_report(p,c);assert not r['compatible'];assert 'grid.shape' in [m['path'] for m in r['mismatches']]
    q=replace(p,grid=c.grid,known_parameters={**p.known_parameters,'k_f':.8})
    assert compatibility_report(q,c)['compatible']
    wrong=replace(q,operator_spec={**q.operator_spec,'rate':'k_f*a*a-k_r*c'})
    assert 'operator_spec' in [m['path'] for m in compatibility_report(wrong,c)['mismatches']]


@pytest.mark.parametrize('dimension',[2,3])
@pytest.mark.parametrize('nonunit',[False,True])
def test_heat_formulas_independent_PDE_initial_boundary_and_domain_scaling(dimension,nonunit):
    spec=example(f'heat-{dimension}d');p=ProblemSpec.from_dict(analyze(spec.request)['problem'])
    if nonunit:p=replace(p,grid=replace(p.grid,domain=DomainSpec(tuple(1.5+i*.2 for i in range(dimension)),tuple(.1*i for i in range(dimension)))))
    for t in templates(p):
        values,report=verify(p,{'expression':t['initial_expression']},t['reference_expression'])
        assert report['passed'],report
        assert values.shape==(4,*p.grid.shape)
        assert report['physical_time_reference_norms'][-1]<=report['physical_time_reference_norms'][0]+1e-12
    t=templates(p)[0];_,bad=verify(p,{'expression':t['initial_expression']},t['initial_expression'])
    assert not bad['passed'] and bad['measured']['PDE_max_abs']>.001
    with pytest.raises(ValueError):verify(p,0,'__import__("os").system("echo unsafe")')
    assert normalize_expression(r'\exp(-8\pi^2 D t)')=='exp(-8*pi**2*D*t)'


def test_inventory_preview_pinning_active_dependencies_and_containment(tmp_path):
    root=tmp_path/'outputs';cache=root/'cache';cache.mkdir(parents=True)
    disposable=cache/'fontlist-v330.json';disposable.write_text('{}')
    user=cache/'my-notes.txt';user.write_text('preserve')
    scientific=root/'accepted/reference.npz';scientific.parent.mkdir();scientific.write_bytes(b'accepted-data')
    outside=tmp_path/'outside';outside.mkdir();(outside/'fontlist-other.json').write_text('{}')
    (root/'link').symlink_to(outside,target_is_directory=True)
    assert not storage.preview(root)['entries'] # Names alone never grant ownership of user files.
    os.utime(disposable,(time.time()-172800,time.time()-172800));storage.register_cache(root,'cache/fontlist-v330.json')
    first=storage.inventory(root,True);assert first['files']==3
    plan=storage.preview(root);assert [r['path'] for r in plan['entries']]==['cache/fontlist-v330.json']
    assert disposable.exists() and scientific.read_bytes()==b'accepted-data'
    storage.pin(root,'./cache',True);assert not storage.preview(root)['entries']
    storage.pin(root,'cache',False)
    active=root/'runs/active';(active/'cache').mkdir(parents=True);(active/'cache/fontlist-v330.json').write_text('{}')
    write_json(active/'run.json',{'kind':'pdeflow_local_run','id':'active'})
    suite=root/'suites/example';suite.mkdir(parents=True);write_json(suite/'suite.json',{'id':'example-suite','entries':[{'run_path':str(active)}]})
    item=next(r for r in storage.inventory(root,True)['entries'] if r['path']=='runs/active/run.json')
    assert item['run_id']=='active' and item['suite_id']=='example-suite'
    assert item['created_at_unix'] is None or item['created_at_unix']>0
    with local.run_lock(active):
        assert all(not r['path'].startswith('runs/active') for r in storage.preview(root)['entries'])
    for relative in ['../outside','link/fontlist-other.json','/etc/passwd']:
        with pytest.raises(ValueError):storage.pin(root,relative)
    assert not any(r['path'].startswith('link/') for r in storage.inventory(root,True)['entries'])


def test_only_reviewed_unchanged_fixture_cache_can_be_quarantined(tmp_path):
    root=tmp_path/'outputs';p=root/'cache/fontlist-v330.json';p.parent.mkdir(parents=True);p.write_text('{}')
    os.utime(p,(time.time()-172800,time.time()-172800));storage.register_cache(root,'cache/fontlist-v330.json')
    plan=storage.preview(root)
    with pytest.raises(ValueError,match='digest'):storage.apply(root,plan['id'],'unreviewed')
    p.write_text('changed')
    with pytest.raises(ValueError,match='stale'):storage.apply(root,plan['id'],plan['review_digest'])
    os.utime(p,(time.time()-172800,time.time()-172800));storage.register_cache(root,'cache/fontlist-v330.json')
    current=storage.preview(root);r=storage.apply(root,current['id'],current['review_digest'])
    assert not p.exists() and not r['permanent_deletion']
    assert json.loads(Path(r['recovery_manifest']).read_text())['moved'][0]['original']==str(p)
    assert (Path(r['quarantine'])/'files/cache/fontlist-v330.json').read_text()=='changed'


def test_more_trained_partial_export_is_not_hidden_by_completed_smoke():
    from pdeflow.inference import selection_key
    c=local.preset_config('heat-2d-512')
    partial={'checkpoint':{'progress':9000},'config':c,'run_state':'paused','export_kinds':['latest']}
    smoke={'checkpoint':{'progress':2},'config':replace(c,training=replace(c.training,steps=2)),'run_state':'completed','export_kinds':['latest']}
    assert selection_key('b',partial)<selection_key('a',smoke)


def test_controller_native_grid_budget_requires_explicit_review():
    from pdeflow.inference import validate_compatibility
    from pdeflow.runtime import InferenceConfig
    from pdeflow.specialists import get_specialist
    c=local.preset_config('heat-2d-512');p=get_specialist('heat').default_problem(c.grid)
    with pytest.raises(ValueError,match='Feedback controller budget'):
        validate_compatibility(p,c,InferenceConfig(controller={}))
    validate_compatibility(p,c,InferenceConfig(controller={'max_grid_points':512*512}))
    validate_compatibility(p,c,InferenceConfig(controller=None))


def test_storage_and_catalogue_cli_default_to_compact_views(tmp_path,monkeypatch,capsys):
    from pdeflow.cli import main
    root=tmp_path/'outputs';root.mkdir();(root/'notes.txt').write_text('preserve user files')
    monkeypatch.setattr('sys.argv',['pdeflow','storage','--root',str(root),'--refresh']);main()
    text=capsys.readouterr().out;assert 'Storage ' in text and 'notes.txt' in text and len(text.splitlines())<30
    monkeypatch.setattr('sys.argv',['pdeflow','storage','--root',str(root),'--json']);main()
    assert json.loads(capsys.readouterr().out)['files']==1
    monkeypatch.setattr('sys.argv',['pdeflow','catalogue']);main()
    text=capsys.readouterr().out;assert '11 finite targets' in text and 'heat-2d-512' in text
    monkeypatch.setattr('sys.argv',['pdeflow','cleanup','--root',str(root),'--preview','--apply','a'*32,'--confirm','unreviewed'])
    with pytest.raises(SystemExit):main()
    assert (root/'notes.txt').read_text()=='preserve user files'
    cache=root/'cache/fontlist-v330.json';cache.parent.mkdir();cache.write_text('{}')
    monkeypatch.setattr('sys.argv',['pdeflow','cleanup','--root',str(root),'--register-cache','./cache/fontlist-v330.json']);main()
    registered=json.loads(capsys.readouterr().out);assert registered['registered'] and registered['files_moved']==0
    assert cache.exists() and (root/'notes.txt').exists()



def test_pin_updates_serialize_with_cleanup_ownership(tmp_path):
    root=tmp_path/'outputs';root.mkdir();(root/'cache').mkdir();meta=root/'.storage';meta.mkdir()
    with local.run_lock(meta):
        with pytest.raises(ValueError,match='Another process'):storage.pin(root,'./cache',True)
    assert storage.pin(root,'./cache',True)['path']=='cache'


def test_training_completion_identity_excludes_inference_controls():
    c=local.preset_config('heat-2d')
    changed=replace(c,inference=replace(c.inference,steps=7,particles=1,observation_strength=.7))
    assert signature(c)==signature(changed)
    assert signature(c)!=signature(replace(c,training=replace(c.training,learning_rate=.00001)))
