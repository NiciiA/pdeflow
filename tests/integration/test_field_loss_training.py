"""Tiny real optimizer execution, not accuracy evidence."""
from dataclasses import replace
import os,json
from pathlib import Path
import pytest
import numpy as np
import torch
from pdeflow.colab import training_preset,preset,Session
from pdeflow.coordinate.training import train,read_checkpoint
from pdeflow.coordinate.model import Autoencoder
from pdeflow.coordinate.validation import representation
from pdeflow.runtime import file_digest,read_json
from pdeflow.field_losses import settings


def tiny(kind='region_balanced_mse',device='cpu'):
    return replace(training_preset('pipeline-check',device,'fp32'),width=8,tokens=2,heads=2,encoder_points=16,query_points=32,
       runtime={'device':device,'threads':1},field_loss={'type':kind,'analysis_resolution':8,'time_groups':2,'region_queries':32})


@pytest.mark.parametrize('kind',['mse','weighted_mse','region_balanced_mse','mae'])
def test_actual_encoder_decoder_updates_and_loss_provenance(tmp_path,kind):
    c=tiny(kind);root=Path(os.environ.get('PDEFLOW_LOSS_EVIDENCE',tmp_path))/kind
    torch.manual_seed(c.seed);initial=Autoencoder(c).state_dict()
    record=train(c.to_dict(),root,'autoencoder')
    cp,_=read_checkpoint(root/'autoencoder.pt')
    assert cp['progress']==2 and cp['field_loss']['type']==kind
    assert not torch.equal(initial['encoder.in_proj_weight'],cp['autoencoder']['encoder.in_proj_weight'])
    assert not torch.equal(initial['decoder.in_proj_weight'],cp['autoencoder']['decoder.in_proj_weight'])
    assert not torch.equal(initial['reference.0.weight'],cp['autoencoder']['reference.0.weight'])
    assert all(r['gradient_norm']>0 for r in cp['history'])
    assert record['config']['field_loss']['type']==kind
    report=representation(root/'autoencoder.pt',root/'validation.json',subset=2)
    assert report['field_metrics']['mse']>=0
    assert report['field_metrics']['rmse']**2==pytest.approx(report['field_metrics']['mse'])
    assert report['records'][0]['training_weighted_objective_normalized']>=0
    assert all('background_mse' in r for r in report['records'][0]['field_metrics']['per_time'])


def test_resume_same_objective_exact_and_change_rejected(tmp_path):
    c=tiny();train(c.to_dict(),tmp_path/'full','autoencoder')
    train(c.to_dict(),tmp_path/'part','autoencoder',stop_after=1)
    old=file_digest(tmp_path/'part/resume.pt')
    with pytest.raises(ValueError,match='Incompatible'):
        train(replace(c,field_loss={**c.field_loss,'alpha':.3}).to_dict(),tmp_path/'part','autoencoder',resume=True)
    assert file_digest(tmp_path/'part/resume.pt')==old
    train(c.to_dict(),tmp_path/'part','autoencoder',resume=True)
    a,_=read_checkpoint(tmp_path/'full/autoencoder.pt');b,_=read_checkpoint(tmp_path/'part/autoencoder.pt')
    for key in a['autoencoder']:assert torch.equal(a['autoencoder'][key],b['autoencoder'][key])
    assert torch.equal(read_checkpoint(tmp_path/'full/resume.pt')[0]['rng'],read_checkpoint(tmp_path/'part/resume.pt')[0]['rng'])


def test_shared_colab_path_defaults_preserved_legacy_and_frozen_flow(tmp_path):
    assert settings(preset('large').field_loss).type=='mse'
    assert training_preset('large').field_loss['alpha']==.5
    c=tiny();session=Session(c,'check',tmp_path/'local',tmp_path/'persistent',allow_cpu_check=True)
    result=session.run(override_gate=True);assert result['state']=='completed'
    cp,_=read_checkpoint(session.root/'flow/coordinate.pt')
    assert cp['field_loss']['type']=='region_balanced_mse'
    assert cp['latent_flow_loss']=='unchanged mean squared CFM velocity error'
    assert cp['autoencoder'] and cp['velocity']
    exported=session.export();assert exported.is_file()
    wrong=Session(replace(c,field_loss={'type':'mae'}),'check',tmp_path/'local',tmp_path/'persistent',allow_cpu_check=True)
    with pytest.raises(ValueError,match='recorded run'):wrong.run(mode='resume')


def test_actual_mps_representation_if_available(tmp_path):
    if not torch.backends.mps.is_available():pytest.skip('MPS unavailable')
    c=tiny(device='mps');root=Path(os.environ.get('PDEFLOW_LOSS_EVIDENCE',tmp_path))/'mps';train(c.to_dict(),root,'autoencoder')
    cp,_=read_checkpoint(root/'autoencoder.pt');assert cp['environment']['device']=='mps' if 'device' in cp['environment'] else cp['config']['runtime']['device']=='mps'


def test_alpha_changes_real_encoder_and_decoder_optimizer_updates(tmp_path):
    c=tiny()
    states=[]
    for alpha in (0.,1.):
        config=replace(c,field_loss={**c.field_loss,'alpha':alpha})
        train(config.to_dict(),tmp_path/str(alpha),'autoencoder')
        states.append(read_checkpoint(tmp_path/str(alpha)/'autoencoder.pt')[0])
    assert states[0]['dataset_identity']==states[1]['dataset_identity']
    for key in ('encoder.in_proj_weight','decoder.in_proj_weight'):
        assert not torch.allclose(states[0]['autoencoder'][key],states[1]['autoencoder'][key],atol=1e-7,rtol=0),key


def test_cuda_region_loss_amp_if_available(tmp_path):
    if not torch.cuda.is_available():pytest.skip('Actual CUDA unavailable')
    c=tiny(device='cuda');c=replace(c,training_options={**c.training_options,'precision':'auto'})
    train(c.to_dict(),tmp_path/'cuda','autoencoder')
    cp,_=read_checkpoint(tmp_path/'cuda/autoencoder.pt');assert cp['actual_precision'] in ('fp16','bf16')
    assert all(np.isfinite(r['training_loss']) for r in cp['history'])
