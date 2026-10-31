"""Training selection → real resolved physics/data/losses → saved metadata."""
from pathlib import Path
from dataclasses import replace
import os
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from pdeflow import training_workflow as workflow
from pdeflow.coordinate.contracts import CoordinateConfig
from pdeflow.coordinate.dataset import Dataset,example
from pdeflow.coordinate.training import train,read_checkpoint
from pdeflow.coordinate.model import Autoencoder
from pdeflow.runtime import write_json,read_json,file_digest

CHOICES={'heat-2d':{'heat':1.},'wave-2d':{'wave':1.},'heat-wave-2d':{'heat':.5,'wave':.5}}

def request(choice):
    return {'problem':choice,'model':'resolution-flexible','budget':'pipeline-check','device':'cpu','advanced':{'encoder_points':16,'query_points':32,'width':8,'heads':2,'tokens':8,'layers':1,'field_loss':{'type':'region_balanced_mse','alpha':.5,'analysis_resolution':16,'region_queries':32,'time_groups':1},'time_sampling':{'encoder':{'initial':1e-8,'early':0.,'uniform':1-1e-8,'final':0.},'query':{'initial':1e-8,'early':0.,'uniform':1-1e-8,'final':0.}}}}

@pytest.fixture(scope='module')
def evidence(tmp_path_factory):
    p=Path(os.environ.get('PDEFLOW_WIRING_EVIDENCE',tmp_path_factory.mktemp('training-equations')));p.mkdir(parents=True,exist_ok=True);return p

@pytest.mark.parametrize('choice',CHOICES)
def test_real_api_choice_dataset_and_persisted_workflow(choice,evidence,monkeypatch,tmp_path):
    from pdeflow.workbench.api import create_app
    body=request(choice)
    with TestClient(create_app(workspace=evidence/'api-workspace',model_roots=[])) as client:
        response=client.post('/api/training/plan',json=body);assert response.status_code==200,response.text;resolved=response.json()
    assert resolved['config']['family_weights']==CHOICES[choice]
    assert resolved['training_physics']['family_weights']==CHOICES[choice]
    c=CoordinateConfig(**resolved['config']).validate();data=Dataset(c,evidence/choice/'data');families={data.get('train',i)['family'] for i in range(c.train_count)};assert families==set(CHOICES[choice]);assert set(data.metadata['normalizations'])==families
    assert resolved['config']['field_loss']['type']=='region_balanced_mse' and resolved['config']['field_loss']['alpha']==.5
    assert set(resolved['training_physics']['state_schemas'])==families
    if 'wave' in families:assert resolved['training_physics']['state_schemas']['wave']==['u','v']
    if choice=='heat-2d':assert set(resolved['training_physics']['parameter_ranges'])=={'diffusivity'}
    if choice=='wave-2d':assert set(resolved['training_physics']['parameter_ranges'])=={'wave_speed'}
    # Control-plane persistence only: don't launch three extra training processes.
    monkeypatch.setattr(workflow,'_spawn',lambda _:0)
    row=workflow.start(body,tmp_path/'workspace',token=choice)
    persisted=read_json(Path(row['path'])/'workflow.json') if 'path' in row else read_json(tmp_path/'workspace/training-workflows'/row['id']/'workflow.json')
    assert persisted['plan']['training_physics']==resolved['training_physics']
    assert persisted['plan']['config']['family_weights']==CHOICES[choice]
    write_json(evidence/(choice+'-request-resolved.json'),{'request':body,'resolved':resolved,'persisted_plan':persisted['plan'],'generated_training_families':sorted(families)})


def test_defaults_weights_and_conflicts():
    assert workflow.plan({'device':'cpu'})['config']['family_weights']==CHOICES['heat-wave-2d']
    body=request('heat-wave-2d');body['advanced']['family_weights']={'heat':2.,'wave':1.};result=workflow.plan(body)
    assert result['training_physics']['family_weights']=={'heat':2/3,'wave':1/3}
    for choice,weights in [('heat-2d',{'heat':.5,'wave':.5}),('heat-wave-2d',{'heat':1.}),('wave-2d',{'heat':1.})]:
        body=request(choice);body['advanced']['family_weights']=weights
        with pytest.raises(ValueError,match='conflict'):workflow.plan(body)


def test_joint_stages_loss_gradients_and_resume(evidence,monkeypatch):
    import pdeflow.coordinate.engine as engine
    torch.set_num_threads(2);resolved=workflow.plan(request('heat-wave-2d'));c=CoordinateConfig(**resolved['config']).validate();root=evidence/'joint';write_json(root/'resolved.json',resolved)
    stage=['autoencoder'];context={};trace=[];item_quantity=engine.item_quantity;field_loss=engine.differentiable_loss
    def view(item,quantity):context.update(family=item['family'],quantity=quantity);return item_quantity(item,quantity)
    def loss(pred,target,weights,kind,**kwargs):
        value=field_loss(pred,target,weights,kind,**kwargs)
        if torch.is_grad_enabled() and pred.requires_grad:
            gradient=torch.autograd.grad(value,pred,retain_graph=True)[0];trace.append({'stage':stage[0],**context,'type':kind,'loss':float(value.detach()),'prediction_gradient_norm':float(gradient.norm()),'weight_sum':float(weights.sum()),'nonuniform_weights':bool(weights.max()-weights.min()>1e-8)})
        return value
    monkeypatch.setattr(engine,'item_quantity',view);monkeypatch.setattr(engine,'differentiable_loss',loss)
    torch.manual_seed(c.seed);initial=Autoencoder(c).state_dict();reports={}
    for name,dep in [('autoencoder',None),('evolution',root/'autoencoder/autoencoder.pt'),('flow',root/'evolution/evolution.pt')]:
        stage[0]=name;record=train(c.to_dict(),root/name,name,dep);cp,_=read_checkpoint(Path(record['export']),name)
        assert cp['config']['family_weights']==c.family_weights and cp['field_loss']['alpha']==.5
        assert cp['training_physics']==resolved['training_physics'] and record['training_physics']==resolved['training_physics']
        assert cp['capabilities']['state_schemas']=={'heat':['u'],'wave':['u','v']}
        for step in cp['history']:
            assert step['family_example_counts']=={'heat':1,'wave':1}
            assert set(step['per_family_loss'])=={'heat','wave'} and all(np.isfinite(v) for v in step['per_family_loss'].values())
        reports[name]={'history':cp['history'],'field_loss':cp['field_loss'],'training_physics':cp['training_physics']}
    a1,_=read_checkpoint(root/'autoencoder/autoencoder.pt')
    for key in ['encoder.fields.0.0.weight','encoder.fields.1.0.weight','decoder.output.0.weight','decoder.output.1.weight']:
        assert not torch.equal(initial[key],a1['autoencoder'][key]),key
    for name in ('autoencoder','evolution'):
        calls=[t for t in trace if t['stage']==name];assert {(t['family'],t['quantity']) for t in calls}=={('heat','u'),('wave','u'),('wave','v')};assert all(t['prediction_gradient_norm']>0 for t in calls);assert all(t['type']=='region_balanced_mse' for t in calls)
    assert not any(t['stage']=='flow' for t in trace) # B remains latent CFM, no spatial weighting.
    # Same model/data/seed/query budget: alpha0 vs .5 changes actual A1 loss and encoder/decoder updates.
    stage[0]='alpha0-control';other=replace(c,field_loss={**c.field_loss,'alpha':0.});train(other.to_dict(),root/'alpha0','autoencoder');control,_=read_checkpoint(root/'alpha0/autoencoder.pt')
    differences={}
    assert control['dataset_identity']==a1['dataset_identity']
    assert control['history'][0]['training_loss']!=a1['history'][0]['training_loss']
    for key in ['encoder.fields.0.0.weight','encoder.fields.1.0.weight','decoder.output.0.weight','decoder.output.1.weight']:
        difference=float((control['autoencoder'][key]-a1['autoencoder'][key]).abs().max());assert difference>0;differences[key]=difference
    before=file_digest(root/'flow/resume.pt')
    for changed in (replace(c,family_weights={'heat':1.}),replace(c,field_loss={**c.field_loss,'alpha':.8})):
        with pytest.raises(ValueError,match='Incompatible'):train(changed.to_dict(),root/'flow','flow',resume=True)
    assert file_digest(root/'flow/resume.pt')==before
    write_json(evidence/'joint-stage-loss-evidence.json',{'stages':reports,'actual_decoded_loss_calls':trace,'alpha0_vs_alpha05_parameter_differences':differences,'alpha0_first_loss':control['history'][0]['training_loss'],'alpha05_first_loss':a1['history'][0]['training_loss'],'changed_family_or_loss_resume_rejected':True,'meaning':'Tiny CPU execution only; no accuracy or representation-gate claim.'})
