"""New default, unchanged experimental estimator and immutable recorded semantics."""
from dataclasses import replace
from pathlib import Path
import json,subprocess,sys
import numpy as np
import pytest
import torch
from pdeflow.field_losses import settings,training_summary,grouped_queries,differentiable_loss,AnalysisCache
from pdeflow.coordinate.contracts import CoordinateConfig
from pdeflow.coordinate.training import train,read_checkpoint,checkpoint_config,SCHEMA
from pdeflow.colab import training_preset
from pdeflow.training_workflow import plan
from pdeflow.runtime import file_digest,read_json


def tiny(kind='mse'):
    torch.set_num_threads(2)
    return training_preset('pipeline-check','cpu','fp32',{'encoder_points':16,'query_points':16,
        'field_loss':{'type':kind,'time_groups':1,'analysis_resolution':8,'region_queries':16}})


def test_all_new_defaults_and_explicit_experiment():
    assert CoordinateConfig().field_loss['type']=='mse'
    for name in ('pipeline-check','xsmall','small','medium','large','xlarge'):
        c=training_preset(name);summary=training_summary(c)
        assert c.field_loss['type']=='mse' and summary['additional_region_queries_per_quantity']==0
        assert summary['description']=='Field reconstruction loss: Standard MSE' and 'alpha' not in summary['settings']
    p=plan({'device':'cpu'});assert p['config']['field_loss']['type']=='mse'
    assert p['field_reconstruction_loss']['settings']['type']=='mse'
    rb=training_preset('pipeline-check','cpu','fp32',{'field_loss':{'type':'region_balanced_mse'}})
    assert settings(rb.field_loss).alpha==.5 and 'alpha=0.5' in training_summary(rb)['description']
    for base in ('configs/models','src/pdeflow/coordinate/profiles'):
        for path in Path(base).glob('*.json'):
            raw=json.loads(path.read_text())
            if 'field_loss' in raw:assert raw['field_loss']['type']=='mse',path


def test_hand_calculated_normalized_masked_per_example_mse():
    # Normalized prediction/reference; per-example valid-point quadrature then
    # equal cases. Physical squared errors would be multiplied by scale².
    pred=torch.tensor([[1.,3.,float('nan')],[2.,float('nan'),float('nan')]],requires_grad=True)
    target=torch.zeros_like(pred);weights=torch.tensor([[.5,.5,0.],[1.,0.,0.]])
    value=differentiable_loss(pred,target,weights,'mse');assert value.item()==pytest.approx(4.5)
    value.backward();torch.testing.assert_close(pred.grad,torch.tensor([[.5,1.5,0.],[2.,0.,0.]]))
    assert value.item()*4==18.  # global physical quantity scale=2


def test_standard_skips_all_region_work_with_identical_domain_queries(monkeypatch):
    import pdeflow.field_losses as losses
    from pdeflow.heat_continuous import FourierHeat
    u=np.zeros((8,8));u[1:3,1:3]=3;u[5:7,5:7]=3
    solver=FourierHeat(u,(1,1),(0,0));item={'solver':solver,'id':'train:fixed','D':.03};c=tiny('region_balanced_mse')
    q,t,w,stats=grouped_queries(replace(c,field_loss={**c.field_loss,'alpha':0.}),[item],torch.Generator().manual_seed(99),{'scale':1},AnalysisCache(),[0.])
    def fail(*args,**kwargs):raise AssertionError('Standard MSE did region-specific training work')
    monkeypatch.setattr(losses,'analyze',fail);monkeypatch.setattr(solver,'frame',fail)
    standard=replace(c,field_loss={**c.field_loss,'type':'mse'})
    sq,st,sw,ss=grouped_queries(standard,[item],torch.Generator().manual_seed(99),{'scale':1},None,[0.])
    torch.testing.assert_close(q[:,:c.query_points],sq,atol=0,rtol=0);torch.testing.assert_close(t[:,:c.query_points],st,atol=0,rtol=0)
    assert ss[0]['additional_region_queries']==0 and sq.shape[1]==c.query_points
    pred=torch.sin(q[...,0]);sp=torch.sin(sq[...,0])
    torch.testing.assert_close(differentiable_loss(pred,t,w),differentiable_loss(sp,st,sw),atol=0,rtol=0)


@pytest.mark.parametrize('kind',['mse','region_balanced_mse'])
def test_actual_a1_a2_loss_config_and_recorded_resume(tmp_path,monkeypatch,kind):
    import pdeflow.coordinate.engine as engine
    import pdeflow.field_losses as losses
    c=tiny(kind)
    if kind=='mse':
        def fail(*args,**kwargs):raise AssertionError('Standard MSE allocated region cache/detection')
        monkeypatch.setattr(engine,'AnalysisCache',fail);monkeypatch.setattr(losses,'analyze',fail)
    a=train(c.to_dict(),tmp_path/'a1','autoencoder',stop_after=1)
    cp,_=read_checkpoint(tmp_path/'a1/resume.pt');assert cp['field_loss']['type']==kind and cp['resolved_field_loss']['settings']['type']==kind
    before=file_digest(tmp_path/'a1/resume.pt');other='region_balanced_mse' if kind=='mse' else 'mse'
    with pytest.raises(ValueError,match='Incompatible'):train(replace(c,field_loss={**c.field_loss,'type':other}).to_dict(),tmp_path/'a1','autoencoder',resume=True)
    assert file_digest(tmp_path/'a1/resume.pt')==before
    a=train(c.to_dict(),tmp_path/'a1','autoencoder',resume=True)
    b=train(c.to_dict(),tmp_path/'a2','evolution',a['export'])
    for r in (a,b):
        cp,_=read_checkpoint(r['export']);assert cp['field_loss']['type']==kind and cp['progress']==2
        assert all(row['family_example_counts']=={'heat':1,'wave':1} for row in cp['history'])
        assert cp['resolved_field_loss']['additional_region_queries_per_quantity']==(16 if kind=='region_balanced_mse' else 0)
        assert read_json(Path(r['export']).parent/'run.json')['resolved_field_loss']==cp['resolved_field_loss']
    # Changing this field choice never changes these separately configured terms.
    otherc=replace(c,field_loss={**c.field_loss,'type':other})
    assert c.joint==otherc.joint and c.evolution==otherc.evolution


def test_legacy_versioned_absence_and_ambiguous_conflict():
    raw=tiny().to_dict();raw.pop('field_loss')
    legacy={'schema':SCHEMA,'engine_version':4,'config':raw}
    c=checkpoint_config(legacy);assert c.field_loss=={} and settings(c.field_loss).estimator=='legacy_uniform_points'
    recorded={**legacy,'field_loss':{'type':'region_balanced_mse','alpha':.73}}
    assert checkpoint_config(recorded).field_loss['alpha']==.73
    with pytest.raises(ValueError,match='Ambiguous resume'):checkpoint_config({**legacy,'engine_version':5})
    with pytest.raises(ValueError,match='Conflicting recorded'):checkpoint_config({**recorded,'config':{**raw,'field_loss':{'type':'mse'}}})


def test_cli_explicit_choices_plan_without_training(tmp_path):
    for kind in ('mse','region_balanced_mse'):
        r=subprocess.run([sys.executable,'-m','pdeflow.cli','training','start','--plan','--device','cpu','--field-loss',kind],capture_output=True,text=True,timeout=30)
        assert r.returncode==0,r.stderr
        p=json.loads(r.stdout);assert p['config']['field_loss']['type']==kind
        r=subprocess.run([sys.executable,'scripts/joint_model.py','config','--preset','smoke','--field-loss',kind,'--output',str(tmp_path/(kind+'.json'))],capture_output=True,text=True,timeout=30)
        assert r.returncode==0,r.stderr
        assert read_json(tmp_path/(kind+'.json'))['field_loss']['type']==kind
