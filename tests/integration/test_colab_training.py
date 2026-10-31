"""Operator-v2 regressions; superseded trajectory tests preserved in recovery evidence."""
from pathlib import Path


from dataclasses import replace


import json,os,shutil,subprocess,sys,zipfile


import numpy as np


import pytest


import torch


from fastapi.testclient import TestClient


from pdeflow.colab import preset,summary,guard,Session,profile,gallery


from pdeflow.coordinate.training import train,read_checkpoint


from pdeflow.coordinate.dataset import Dataset,example,identity


from pdeflow.coordinate.optimization import options,precision,learning_rate,scientific_source


from pdeflow.runtime import read_json,write_json,file_digest


from pdeflow import persistence


def tiny(**kwargs):
    return replace(preset('pipeline-check','cpu','fp32'),runtime={'device':'cpu','threads':1},**kwargs)


def test_large_exact_extra_large_validation_and_no_cpu_capacity_allocation():
    c=preset('large');assert (c.train_count,c.validation_count,c.test_count,c.ae_steps,c.flow_steps)==(65536,2048,2048,100000,100000)
    assert (c.width,c.tokens,c.layers,c.heads,c.encoder_points,c.query_points,c.batch_size,c.learning_rate)==(768,384,16,12,1024,512,8,.0003)
    assert c.unconditional_probability==0 and c.observation_counts==(1,128) and c.checkpoint_every==500
    assert summary(c)['parameters']['inference'] < summary(c)['parameters']['total_training']
    x=preset('extra-large');assert x.width>c.width and x.tokens>c.tokens and x.layers>c.layers
    with pytest.raises(ValueError,match='divisible by heads'):replace(c,width=513).validate()
    with pytest.raises(ValueError):preset('large',overrides={'training_options':{'microbatch':3,'accumulation':1}})
    with pytest.raises(ValueError,match='Substantial'):guard(replace(c,runtime={'device':'cpu'}),True)
    assert precision('auto',torch.device('cpu'))[0]=='fp32'
    with pytest.raises(ValueError,match='CUDA'):precision('fp16',torch.device('cpu'))


def test_optimizer_accumulation_checkpointing_resume_rng_and_extensions(tmp_path):
    c=tiny();full=tmp_path/'full';partial=tmp_path/'partial'
    train(c.to_dict(),full,'autoencoder')
    train(c.to_dict(),partial,'autoencoder',stop_after=1)
    first=read_checkpoint(partial/'resume.pt')[0];assert first['progress']==1 and first['optimizer'] and first['scheduler'] and 'scaler' in first
    train(c.to_dict(),partial,'autoencoder',resume=True)
    a=read_checkpoint(full/'autoencoder.pt')[0];b=read_checkpoint(partial/'autoencoder.pt')[0]
    for key in a['autoencoder']:assert torch.equal(a['autoencoder'][key],b['autoencoder'][key]),key
    assert torch.equal(read_checkpoint(full/'resume.pt')[0]['rng'],read_checkpoint(partial/'resume.pt')[0]['rng'])
    mc=replace(c,training_options={**c.training_options,'microbatch':1,'accumulation':2,'activation_checkpointing':True})
    train(mc.to_dict(),tmp_path/'micro','autoencoder');m=read_checkpoint(tmp_path/'micro/autoencoder.pt')[0]
    for key in a['autoencoder']:torch.testing.assert_close(a['autoencoder'][key],m['autoencoder'][key],atol=2e-6,rtol=2e-5)
    extended=replace(c,ae_steps=3)
    with pytest.raises(ValueError,match='extend'):train(extended.to_dict(),partial,'autoencoder',resume=True)
    train(extended.to_dict(),partial,'autoencoder',resume=True,extend=True)
    cp=read_checkpoint(partial/'resume.pt')[0];assert cp['progress']==3 and cp['scheduler']['settings']['horizon']==2
    before=(partial/'resume.pt').read_bytes()
    with pytest.raises(ValueError,match='Incompatible model/data'):train(replace(extended,width=32).to_dict(),partial,'autoencoder',resume=True)
    assert (partial/'resume.pt').read_bytes()==before


def test_persistent_integrity_interrupted_upload_retention_and_dependency(tmp_path):
    local=tmp_path/'local';remote=tmp_path/'persistent';local.mkdir()
    (local/'representation.pt').write_bytes(b'immutable frozen dependency');(local/'resume.pt').write_bytes(b'first boundary')
    first=persistence.publish(local,remote,keep=2)
    # Incomplete newest uploads and malformed manifests are ignored, never chosen.
    (remote/'snapshots/incomplete.json').write_text('{bad')
    write_json(remote/'snapshots/half.json',{'created_unix':1e30,'id':'half','files':{'resume.pt':{'sha256':'0'*64,'bytes':10}}})
    (remote/'snapshots/half.commit').write_text('not its digest')
    assert persistence.latest(remote)['id']==first['id']
    for i in range(3):
        (local/'resume.pt').write_bytes(f'boundary{i}'.encode());persistence.publish(local,remote,keep=2)
    latest=persistence.latest(remote);assert latest['files']['representation.pt']['sha256']==first['files']['representation.pt']['sha256']
    target=tmp_path/'restored';persistence.restore(remote,target);assert (target/'resume.pt').read_bytes()==b'boundary2'
    commits=list((remote/'snapshots').glob('*.commit'));valid=[p for p in commits if p.stem not in ('half','incomplete')];assert len(valid)==2
    # Corrupt latest object: recovery must fall back to prior complete snapshot.
    bad=remote/'objects'/latest['files']['resume.pt']['sha256'];bad.write_bytes(b'partial upload')
    assert persistence.latest(remote)['id']!=latest['id']
    write_json(remote/'snapshots/traversal.json',{'created_unix':1e30,'id':'bad','files':{'../escape':first['files']['resume.pt']}})
    with pytest.raises(ValueError):persistence._safe(local,'../escape')


def test_profile_is_disposable_and_restores_rng(tmp_path):
    torch.manual_seed(328);before=torch.get_rng_state().clone();r=profile(tiny(),tmp_path,True)
    assert r['success'] and set(r['stages'])=={'autoencoder','evolution','flow'} and r['stages']['flow']['step_seconds']>0
    assert torch.equal(before,torch.get_rng_state()) and not list(tmp_path.glob('pdeflow-profile-*'))
    assert r['stages']['autoencoder']['memory']['cuda_peak_allocated'] is None


def test_store_ownership_and_arbitrary_files_protected(tmp_path):
    local=tmp_path/'local';local.mkdir();(local/'resume.pt').write_bytes(b'owned')
    remote=tmp_path/'unrelated';remote.mkdir();(remote/'personal.txt').write_text('keep')
    with pytest.raises(ValueError,match='not an owned'):persistence.publish(local,remote)
    assert (remote/'personal.txt').read_text()=='keep'


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Requires actual CUDA GPU; policy unit tests are not GPU verification')
def test_actual_cuda_amp_checkpoint(tmp_path):
    c=replace(tiny(),runtime={'device':'cuda'},training_options={**tiny().training_options,'precision':'auto'})
    r=train(c.to_dict(),tmp_path/'cuda','autoencoder');assert r['state']=='completed' and r['memory']['cuda_peak_allocated']>0
