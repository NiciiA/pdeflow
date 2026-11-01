"""Real local filesystem persistence; not Google Drive/CUDA execution evidence."""
from pathlib import Path
from dataclasses import replace
import json,sys,subprocess,os,shutil
import pytest
import torch
from pdeflow import persistence
from pdeflow.runtime import write_json,read_json,file_digest
from pdeflow.colab import Session
from pdeflow.coordinate.training import train,read_checkpoint
from pdeflow.training_artifacts import build,verify_view,load_diagnostic,download_archive
from pdeflow.source_bundle import package,verify
from test_observation_aligned_objectives import tiny


def test_store_pointer_corruption_partial_retention_and_conflict_preflight(tmp_path):
    local=tmp_path/'fast';local.mkdir();remote=tmp_path/'drive';(local/'encoder.pt').write_bytes(b'exact frozen E')
    (local/'resume.pt').write_bytes(b'one');a=persistence.publish(local,remote,keep=1,tag='best')
    (local/'resume.pt').write_bytes(b'two');b=persistence.publish(local,remote,keep=1)
    (remote/'latest.json').write_text('{broken');(remote/'objects/.upload-incomplete').write_bytes(b'partial')
    write_json(remote/'snapshots'/('f'*32+'.json'),{'files':{},'created_unix':1e30,'id':'f'*32})
    assert persistence.latest(remote)['id']==b['id']
    assert {r['id'] for r in persistence.snapshots(remote)}=={a['id'],b['id']}
    target=tmp_path/'restored';persistence.restore(remote,target,a['id']);assert (target/'resume.pt').read_bytes()==b'one'
    # All conflicts are found before adding even the nonconflicting encoder file.
    other=tmp_path/'conflict';other.mkdir();(other/'resume.pt').write_bytes(b'user work')
    with pytest.raises(ValueError,match='new empty'):persistence.restore(remote,other)
    assert not (other/'encoder.pt').exists()
    (remote/'objects'/b['files']['resume.pt']['sha256']).write_bytes(b'corrupt')
    assert persistence.latest(remote)['id']==a['id']
    assert persistence.latest(remote)['files']['encoder.pt']==a['files']['encoder.pt']


def test_visible_bounded_failure_preserves_last_good_snapshot(tmp_path,monkeypatch,capsys):
    session=Session.new(tiny(),'same display',tmp_path/'fast',tmp_path/'drive',allow_cpu_check=True,save_retries=1)
    write_json(session.root/'session.json',{'config':session.c.to_dict()});good=session.sync(True)
    def broken(*a,**k):raise OSError('simulated Drive disconnect')
    monkeypatch.setattr(persistence,'publish',broken)
    with pytest.raises(OSError,match='disconnect'):session.sync(True)
    status=read_json(session.root/'sync-status.json');assert status['state']=='failed' and status['attempt']==2 and 'traceback' in status
    assert session.snapshot['id']==good['id'] and persistence.latest(session.persistent)['id']==good['id']
    assert 'PERSISTENT SAVE FAILED' in capsys.readouterr().out


def test_unique_run_ids_and_non_destructive_probe(tmp_path):
    a=Session.new(tiny(),'Same name',tmp_path/'fast',tmp_path/'drive',allow_cpu_check=True)
    b=Session.new(tiny(),'Same name',tmp_path/'fast',tmp_path/'drive',allow_cpu_check=True)
    assert a.name!=b.name and a.display_name==b.display_name
    assert persistence.probe(a.persistent)['verified'];assert not list((tmp_path/'drive').glob('.pdeflow-probe-*'))


def test_real_stage_artifacts_fresh_process_restore_and_idempotent_finalize(tmp_path,monkeypatch):
    c=tiny({'beta':.0001,'minimum_s':0,'sensor_budget':2,'discrepancy_relative':.05})
    c=replace(c,evolution={**c.evolution,'physics':{'weight':.0001,'query_points':1}}).validate()
    session=Session.new(c,'bounded integration',tmp_path/'fast',tmp_path/'drive',allow_cpu_check=True)
    # Explicit low-level objective plumbing check, NOT a gate-passing model.
    # Production coordinate_stages still enforces the representation gate.
    write_json(session.root/'session.json',{'config':c.to_dict(),'scientific_source':__import__('pdeflow.coordinate.optimization',fromlist=['scientific_source']).scientific_source(),'state':'objective integration only; accuracy not assessed','display_name':session.display_name})
    source=package(session.root/'source/source.zip');source['scientific_source']=read_json(session.root/'session.json')['scientific_source'];write_json(session.root/'source/identity.json',source)
    dep=None
    for stage in ('autoencoder','evolution','flow'):
        r=train(c.to_dict(),session.root/stage,stage,dep)
        assert r['state']=='completed';dep=r['export']
        cp=read_checkpoint(session.root/stage/'resume.pt')[0]
        assert all(row['family_example_counts']=={'heat':1,'wave':1} for row in cp['history'])
        assert all(row['gradients_finite'] for row in cp['history'])
        assert cp['progress']==2 and 'ambient_rng' in cp
        if stage=='flow':
            assert all(row['actual_forward_microbatch']==2 for row in cp['history'])
            assert cp['history'][0]['components']['observation_endpoint_loss']>0
        if stage=='evolution':assert 'physics_heat' in cp['history'][0]['components'] or 'physics_wave_u' in cp['history'][0]['components']
    result=session.finalize();first=result['verified_snapshot'];second=session.finalize()['verified_snapshot'];assert first==second
    bundle=load_diagnostic(session.root);assert bundle.ae.encoder is not None
    assert verify_view(session.root,'resumable')['stages']['flow']['successful_updates']==2
    zip_path=download_archive(session.root,'diagnostic');assert download_archive(session.root,'diagnostic')==zip_path
    script='''from pdeflow.colab import Session
from pdeflow.training_artifacts import load_diagnostic,verify_view
from pdeflow.coordinate.training import train,read_checkpoint
from dataclasses import replace
import sys
s=Session.from_store(sys.argv[1],sys.argv[2],sys.argv[3],allow_cpu_check=True)
b=load_diagnostic(s.root);assert b.ae.encoder is not None
cp=read_checkpoint(s.root/'flow/resume.pt')[0]
assert cp['progress']==2 and cp['optimizer'] and cp['scheduler']['completed_updates']==2
c=replace(s.c,flow_steps=3)
r=train(c.to_dict(),s.root/'flow','flow',resume=True,extend=True)
assert r['progress']==3
print('FRESH PROCESS RESTORE/EXTEND OK')
'''
    outcome=subprocess.run([sys.executable,'-c',script,str(session.persistent.parent),session.name,str(tmp_path/'fresh')],capture_output=True,text=True,timeout=90)
    assert outcome.returncode==0,outcome.stdout+outcome.stderr
    assert 'FRESH PROCESS' in outcome.stdout
    # Entire prior run, source and all checkpoints survived unchanged.
    verify_view(session.root,'resumable')
    if os.environ.get('PDEFLOW_COMPLETE_EVIDENCE'):
        evidence=Path(os.environ['PDEFLOW_COMPLETE_EVIDENCE']);evidence.mkdir(parents=True,exist_ok=True)
        shutil.copytree(session.root,evidence/'complete-fixture')
        shutil.copytree(session.persistent,evidence/'complete-store')
        write_json(evidence/'complete-result.json',{'finalize':result,'fresh_process':outcome.stdout,'first_and_second_finalization_snapshot_identical':first==second,'stages':verify_view(session.root,'resumable')['stages'],'qualification':'Low-level two-update objective plumbing fixture; no accuracy or gate-passing claim'})
    # Explicit continue may keep learning/local artifacts, but the final handoff
    # must never label NEW unsaved bundles as persistently verified.
    write_json(session.root/'new-note.json',{'review':'after last verified save'})
    session.save_failure='continue';session.save_retries=0
    def failed(*a,**k):raise OSError('simulated final upload failure')
    monkeypatch.setattr(persistence,'publish',failed)
    failed_final=session.finalize(standalone_check=False)
    assert failed_final['complete_bundle_persisted'] is False and failed_final['final_bundle_snapshot'] is None
    assert persistence.latest(session.persistent)['id']==first


def test_safe_stop_first_checkpoint_and_fresh_process_exact_cpu_resume(tmp_path):
    c=tiny();s=Session.new(c,'partial representation',tmp_path/'fast',tmp_path/'drive',allow_cpu_check=True,save_updates=1000)
    outcome=s.run(stages='representation',stop_after=1);assert outcome['state']=='interrupted'
    latest=persistence.latest(s.persistent);assert 'autoencoder/resume.pt' in latest['files'] and 'source/source.zip' in latest['files']
    views=s.finalize()['artifact_views'];assert not views['inference']['available']
    assert not verify_view(s.root,'diagnostic')['generative_model_available']
    restored=Session.from_store(s.persistent.parent,s.name,tmp_path/'fresh',allow_cpu_check=True)
    train(c.to_dict(),restored.root/'autoencoder','autoencoder',resume=True)
    train(c.to_dict(),tmp_path/'control','autoencoder')
    a=read_checkpoint(restored.root/'autoencoder/resume.pt')[0];b=read_checkpoint(tmp_path/'control/resume.pt')[0]
    assert a['progress']==b['progress']==2 and torch.equal(a['rng'],b['rng'])
    for name in a['autoencoder']:assert torch.equal(a['autoencoder'][name],b['autoencoder'][name]),name
