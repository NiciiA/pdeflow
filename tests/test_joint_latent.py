"""Actual shared A1/A2/B execution; tiny budgets do not establish accuracy."""
from dataclasses import replace
from pathlib import Path
import os,shutil
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from pdeflow.colab import training_preset,Session
from pdeflow.coordinate.training import train,read_checkpoint
from pdeflow.coordinate.inference import CoordinateBundle,reconstruct
from pdeflow.coordinate.physics import problem
from pdeflow.coordinate.joint import schema,layout
from pdeflow.coordinate.presets import LADDER,capacity,counts
from pdeflow.coordinate.registry import register_checkpoint
from pdeflow.core import GridSpec,DomainSpec
from pdeflow.runtime import file_digest,write_json

@pytest.fixture(scope='module')
def executed(tmp_path_factory):
    torch.set_num_threads(2);root=Path(os.environ.get('PDEFLOW_OPERATOR_EVIDENCE',tmp_path_factory.mktemp('operator')));root.mkdir(parents=True,exist_ok=True)
    c=training_preset('smoke','cpu','fp32',{'encoder_points':16,'query_points':16,'field_loss':{'analysis_resolution':8,'region_queries':16,'time_groups':1}})
    write_json(root/'config.json',c.to_dict());train(c.to_dict(),root/'uninterrupted-a1','autoencoder')
    for stage,dep in [('autoencoder',None),('evolution',root/'autoencoder/autoencoder.pt'),('flow',root/'evolution/evolution.pt')]:
        result=train(c.to_dict(),root/stage,stage,dep,stop_after=1);assert result['state']=='interrupted'
        train(c.to_dict(),root/stage,stage,dep,resume=True)
    a,_=read_checkpoint(root/'uninterrupted-a1/autoencoder.pt');b,_=read_checkpoint(root/'autoencoder/autoencoder.pt')
    for key in a['autoencoder']:assert torch.equal(a['autoencoder'][key],b['autoencoder'][key]),key
    # Frozen canonical encoders/decoders do not move in A2.
    e,_=read_checkpoint(root/'evolution/evolution.pt')
    for key,value in b['autoencoder'].items():
        if not key.startswith(('evolution.','graph_encoder.')):assert torch.equal(value,e['autoencoder'][key]),key
    for prefix in ('evolution.','graph_encoder.'):
        assert any(not torch.equal(value,e['autoencoder'][key]) for key,value in b['autoencoder'].items() if key.startswith(prefix))
    return root,c

@pytest.mark.parametrize('name',list(LADDER))
def test_capacity_counts(name):
    c=training_preset(name,'cpu','fp32');n=counts(c);assert n['inference']==n['total_training']-n['encoder'];assert n['trainable_A1']+n['trainable_A2']+n['trainable_B']==n['total_training'];assert sum(g.stop-g.start for g in layout(c))==c.tokens


def test_shared_pipeline_resume_portable_and_known_unknown_modes(executed):
    root,c=executed;cp,_=read_checkpoint(root/'flow/coordinate.pt');assert cp['architecture']==schema(c)['architecture'] and cp['autoencoder'] is None and cp['inference_state'];assert cp['latent_scale'].min()>=.01
    standalone=root/'standalone/coordinate.pt';standalone.parent.mkdir();shutil.copyfile(root/'flow/coordinate.pt',standalone)
    # Dependency files removed from the portable inference path, not deleted.
    b=CoordinateBundle(standalone);assert b.ae.encoder is None
    q=torch.tensor([[.1,.2,0.],[.2,.3,.237]]);noise=torch.randn(1,c.tokens,c.width);evidence=[]
    for family in ('heat','wave'):
        parameter='diffusivity' if family=='heat' else 'wave_speed';value=.07 if family=='heat' else .4;rows=[dict(field='u',x=.1,y=.2,time=.237,value=.4,sigma=.03)];grid=GridSpec((8,8),DomainSpec((1.,1.)),(0.,.237,1.))
        for known in (True,False):
            p=problem(grid,family,value if known else None);a,r=b.sample(p,rows,q,steps=2,particles=1,initial_noise=noise);v=b.velocity_samples.copy() if family=='wave' else None
            again,r2=b.sample(p,list(reversed(rows)),q,steps=2,particles=1,initial_noise=noise);np.testing.assert_array_equal(a,again)
            if known:assert r['inferred_parameters'][parameter]==[value]
            if family=='wave':np.testing.assert_array_equal(v,b.velocity_samples);assert r['observation_consistency']['quantities']==['u'] and 'u_t_minus_v' in r['physics']['per_sample'][0]
            assert r['no_reference_input'];evidence.append(r)
    write_json(root/'four-modes.json',evidence)
    from pdeflow.coordinate.validation import representation,evolution,flow
    representation(root/'autoencoder/autoencoder.pt',root/'representation-validation.json',subset=2)
    evolution(root/'evolution/evolution.pt',root/'evolution-validation.json',subset=2)
    flow(root/'flow/coordinate.pt',root/'flow-validation.json',subset=2,steps=2,particles=1,resolution=8)


def test_api_independent_validation_and_reference_isolation(executed,tmp_path):
    from pdeflow.workbench.api import create_app
    root,c=executed;entry=register_checkpoint(root/'flow/coordinate.pt',tmp_path);before=file_digest(root/'flow/coordinate.pt')
    with TestClient(create_app(workspace=tmp_path,model_roots=[])) as client:
        for family in ('heat','wave'):
            p=problem(GridSpec((8,8),DomainSpec((1.,1.)),(0.,.237,1.)),family,None);rows=[dict(field='u',x=.1,y=.2,time=.237,value=.4,sigma=.03)];body={'model_id':entry['registered_sha256'][:32],'problem':p.to_dict(),'rows':rows,'steps':2,'particles':1}
            assert client.post('/api/coordinate/validate',json=body).status_code==200
            for hidden in ('reference','initial_velocity','true_parameter'):
                assert client.post('/api/coordinate/validate',json={**body,hidden:[[1.]]}).status_code==422
            assert client.post('/api/coordinate/validate',json={**body,'problem':{**p.to_dict(),'known_fields':{'initial_u':[[42.]]}}}).status_code==422
            wrong='wave_speed' if family=='heat' else 'diffusivity'
            assert client.post('/api/coordinate/validate',json={**body,'problem':{**p.to_dict(),'known_parameters':{wrong:.4},'unknown_parameters':[]}}).status_code==422
    assert before==file_digest(root/'flow/coordinate.pt')


def test_loss_disclosure_dependency_and_legacy_rejection(executed,tmp_path):
    root,c=executed;before=file_digest(root/'flow/resume.pt')
    for changed in (replace(c,joint={**c.joint,'known_probability':.2}),replace(c,field_loss={**c.field_loss,'alpha':.2})):
        with pytest.raises(ValueError,match='Incompatible'):train(changed.to_dict(),root/'flow','flow',resume=True)
    assert file_digest(root/'flow/resume.pt')==before
    with pytest.raises(ValueError,match='evolution'):train(c.to_dict(),tmp_path/'wrong','flow',root/'autoencoder/autoencoder.pt')
    path=tmp_path/'old.pt';torch.save({'schema':'joint_heat_v1','architecture':'physics_conditioned_joint_latent_v1'},path)
    with pytest.raises(ValueError,match='Legacy architecture'):read_checkpoint(path)


def test_decoded_auxiliary_and_correction_real_gradients(executed):
    from pdeflow.coordinate.consistency import terms,refine,Correction
    root,c=executed;b=CoordinateBundle(root/'flow/coordinate.pt');f=torch.ones(1,dtype=torch.long);k=torch.tensor([False]);p=torch.zeros(1);z=torch.randn(1,c.tokens,c.width,requires_grad=True);q=torch.tensor([[[.1,.2,.237]]]);features=torch.tensor([[[.2,.03,1.]]]);obs=(q,features,torch.ones(1,1,dtype=torch.bool),torch.zeros(1,1,dtype=torch.long));ns=b.cp['normalizations']['wave']
    loss=terms(b.ae,c,z,p,k,f,obs,ns,True);sum(v.mean() for v in loss).backward();assert torch.isfinite(z.grad).all() and z.grad.abs().sum()>0
    corrected,report=refine(b.ae,c,z.detach(),p,k,f,obs,torch.zeros_like(z),torch.ones_like(z),ns,Correction(steps=1));assert report['steps']==1 and torch.isfinite(corrected).all()


def test_shared_colab_local_driver_and_persistent_export(tmp_path):
    c=training_preset('smoke','cpu','fp32',{'encoder_points':16,'query_points':16,'field_loss':{'analysis_resolution':8,'region_queries':16,'time_groups':1}});s=Session(c,'shared',tmp_path/'local',tmp_path/'persistent',True);result=s.run(override_gate=True);assert result['state']=='completed';assert (s.root/'evolution/evolution.pt').exists();assert s.export().exists()
    from pdeflow.persistence import latest
    assert latest(s.persistent)
