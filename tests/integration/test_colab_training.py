"""Actual CPU execution. CUDA policy tests are not evidence of GPU execution."""
from pathlib import Path
from dataclasses import replace
import json,os,shutil,subprocess,sys,zipfile
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from pdeflow.colab import preset,summary,guard,Session,profile,gallery
from pdeflow.coordinate.training import train,read_checkpoint
from pdeflow.coordinate.dataset import Dataset,example,identity,query_subset
from pdeflow.coordinate.optimization import options,precision,learning_rate,scientific_source
from pdeflow.runtime import read_json,write_json,file_digest
from pdeflow import persistence


def tiny(**kwargs):
    return replace(preset('pipeline-check','cpu','fp32'),runtime={'device':'cpu','threads':1},**kwargs)


def test_large_exact_extra_large_validation_and_no_cpu_capacity_allocation():
    c=preset('large');assert (c.train_count,c.validation_count,c.test_count,c.ae_steps,c.flow_steps)==(65536,2048,2048,100000,100000)
    assert (c.width,c.tokens,c.layers,c.heads,c.encoder_points,c.query_points,c.batch_size,c.learning_rate)==(512,128,8,8,1024,512,8,.0003)
    assert c.unconditional_probability==0 and c.observation_counts==(1,128) and c.checkpoint_every==500
    assert summary(c)['parameters']=={'representation':3235329,'flow':18675200}
    x=preset('extra-large');assert x.width>c.width and x.tokens>c.tokens and x.layers>c.layers
    with pytest.raises(ValueError,match='divide heads'):replace(c,width=513).validate()
    with pytest.raises(ValueError):preset('large',overrides={'training_options':{'microbatch':3,'accumulation':1}})
    with pytest.raises(ValueError,match='Substantial'):guard(replace(c,runtime={'device':'cpu'}),True)
    assert precision('auto',torch.device('cpu'))[0]=='fp32'
    with pytest.raises(ValueError,match='CUDA'):precision('fp16',torch.device('cpu'))


def test_population_identity_split_normalization_resolution_and_unique_points(tmp_path):
    c=replace(tiny(),temporal_mode='snapshots');data=Dataset(c,tmp_path)
    assert not (tmp_path/'dataset.npz').exists()
    assert set(data.metadata['normalization']['fitted_ids'])<=set(data.metadata['ids'][:c.train_count])
    assert not set(data.metadata['ids'][:c.train_count])&set(data.metadata['ids'][c.train_count:])
    a=example(c,'train',0);b=example(c,'validation',0)
    assert a['id']!=b['id'] and not np.array_equal(a['values'],b['values'])
    assert identity(c)!=identity(replace(c,distribution={**c.distribution,'amplitude':[1,2]}))
    assert np.array_equal(a['values'],example(c,'train',0)['values'])
    qi,w=query_subset(c,torch.Generator().manual_seed(4));assert len(qi.unique())==len(qi)
    for t in range(len(c.times)):assert abs(float(w[(qi//64)==t].sum())-1/len(c.times))<1e-7
    cosine=next(i for i in range(6) if example(c,'train',i)['family']=='cosine')
    q=example(c,'train',cosine,(16,16))['values'].reshape(4,16,16)[:,::2,::2]
    np.testing.assert_allclose(q,example(c,'train',cosine)['values'].reshape(4,8,8),atol=2e-7)
    custom=replace(c,distribution={**c.distribution,'families':{'cosine':1.,'brush':2.}})
    roundtrip=replace(custom,distribution=json.loads(json.dumps(custom.distribution,sort_keys=True)))
    assert identity(custom)==identity(roundtrip) and np.array_equal(example(custom,'train',0)['values'],example(roundtrip,'train',0)['values'])
    assert {example(c,'train',i)['family'] for i in range(6)}==set(c.distribution.get('families',{})) or len({example(c,'train',i)['family'] for i in range(6)})==6
    gallery(c,tmp_path/'gallery.png');assert (tmp_path/'gallery.json').is_file()


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


def test_gate_stop_restore_flow_portable_export_and_real_import(tmp_path):
    c=tiny();session=Session(c,'roundtrip',tmp_path/'fast',tmp_path/'persistent',True)
    r=session.run(gate_ratio=1e-12);assert r['state']=='needs_review' and not (session.root/'flow').exists()
    r=session.run(mode='resume',gate_ratio=1e-12,override_gate=True,stop_after=1);assert r['state']=='interrupted'
    partial=session.export();cp,_=read_checkpoint(partial);assert cp['training_status'].startswith('partial') and not cp['completed'] and cp['inference_ready']
    # Runtime loss: only complete committed files are restored, not a random restart.
    shutil.rmtree(session.root)
    session.restore();r=session.run(mode='resume',gate_ratio=1e-12,override_gate=True);assert r['state']=='completed'
    exported=session.export();isolated=tmp_path/'standalone';isolated.mkdir();shutil.copyfile(exported,isolated/'coordinate.pt')
    inference='''import json,numpy as np
from pdeflow.coordinate.inference import CoordinateBundle
from pdeflow.coordinate.contracts import grid_queries
from pdeflow.equations import heat_problem
from pdeflow.core import GridSpec,DomainSpec
b=CoordinateBundle("coordinate.pt","cpu")
p=heat_problem(GridSpec((4,4),DomainSpec((1.,1.)),(0.,.05,.1,.15)),.07)
rows=[dict(field="u",x=.2,y=.3,time=.05,value=.1,sigma=.01)]
a,r=b.sample(p,rows,grid_queries(p.grid),steps=1,particles=1)
assert np.isfinite(a).all() and r["no_reference_input"]
print(json.dumps({"shape":list(a.shape),"no_reference_input":r["no_reference_input"]}))
'''
    result=subprocess.run([sys.executable,'-c',inference],cwd=isolated,capture_output=True,text=True);assert result.returncode==0,result.stderr
    assert 'true' in result.stdout
    from pdeflow.workbench.api import create_app
    with TestClient(create_app(workspace=tmp_path/'import-workspace',model_roots=[])) as client:
        response=client.post('/api/playground/import',content=exported.read_bytes(),headers={'Content-Type':'application/octet-stream'});assert response.status_code==200,response.text
        from pdeflow.equations import heat_problem
        from pdeflow.core import GridSpec,DomainSpec
        p=heat_problem(GridSpec((4,4),DomainSpec((1.,1.)),c.times),.07)
        entries=client.post('/api/playground/models',json={'problem':p.to_dict()}).json();assert entries and entries[0]['compatible']
        wrong=client.post('/api/coordinate/validate',json={'model_id':entries[0]['id'],'problem':p.to_dict(),'rows':[]});assert wrong.status_code==422
    assert not any(p.name=='dataset.npz' for p in session.root.rglob('*'))
    assert read_json(session.root/'session.json')['attempts'][-1]['gate_override']


def test_profile_is_disposable_and_restores_rng(tmp_path):
    torch.manual_seed(328);before=torch.get_rng_state().clone();r=profile(tiny(),tmp_path,True)
    assert r['success'] and set(r['stages'])=={'autoencoder','flow'} and r['stages']['flow']['step_seconds']>0
    assert torch.equal(before,torch.get_rng_state()) and not list(tmp_path.glob('pdeflow-profile-*'))
    assert r['stages']['autoencoder']['memory']['cuda_peak_allocated'] is None


def test_safe_runtime_changes_source_override_and_completed_run_validation(tmp_path):
    c=tiny();root=tmp_path/'ae';train(c.to_dict(),root,'autoencoder',stop_after=1)
    cp,_=read_checkpoint(root/'resume.pt');cp['scientific_source']['sha256']='different scientific source'
    torch.save(cp,root/'resume.pt')
    with pytest.raises(ValueError,match='Scientific training source'):train(c.to_dict(),root,'autoencoder',resume=True)
    changed=replace(c,batch_size=3,training_options={**c.training_options,'microbatch':2,'accumulation':2,'allow_source_change':True})
    train(changed.to_dict(),root,'autoencoder',resume=True)
    result,_=read_checkpoint(root/'resume.pt')
    assert result['source_change_override'] and result['effective_examples_processed']==5
    from pdeflow.training_workflow import coordinate_stages
    # Completed stages still validate requests; an equal update count is not permission to change architecture.
    parent=tmp_path/'driver';parent.mkdir();shutil.copytree(root,parent/'autoencoder')
    with pytest.raises(ValueError,match='Incompatible model/data'):coordinate_stages(replace(changed,width=32).to_dict(),parent,stages='representation',mode='resume')


def test_store_ownership_and_arbitrary_files_protected(tmp_path):
    local=tmp_path/'local';local.mkdir();(local/'resume.pt').write_bytes(b'owned')
    remote=tmp_path/'unrelated';remote.mkdir();(remote/'personal.txt').write_text('keep')
    with pytest.raises(ValueError,match='not an owned'):persistence.publish(local,remote)
    assert (remote/'personal.txt').read_text()=='keep'


def test_partial_microbatch_and_live_physical_validation(tmp_path):
    c=tiny(batch_size=3);a=tmp_path/'whole';b=tmp_path/'partial'
    train(c.to_dict(),a,'autoencoder')
    micro=replace(c,training_options={**c.training_options,'microbatch':2,'accumulation':2})
    train(micro.to_dict(),b,'autoencoder')
    x=read_checkpoint(a/'autoencoder.pt')[0]['autoencoder'];y=read_checkpoint(b/'autoencoder.pt')[0]['autoencoder']
    for key in x:torch.testing.assert_close(x[key],y[key],atol=2e-6,rtol=2e-5)
    live=replace(tiny(),training_options={**tiny().training_options,'broad_every':1,'broad_subset':1})
    train(live.to_dict(),tmp_path/'live-ae','autoencoder')
    assert read_json(tmp_path/'live-ae/history.json')[-1]['broad_initial_rmse']>=0
    train(live.to_dict(),tmp_path/'live-flow','flow',tmp_path/'live-ae/autoencoder.pt')
    assert (tmp_path/'live-flow/validation-preview.npz').exists()
    assert read_json(tmp_path/'live-flow/history.json')[-1]['broad_ensemble_mean_rmse']>=0


def test_completed_flow_extension_preserves_export_and_schedule(tmp_path):
    c=tiny();session=Session(c,'extend',tmp_path/'local',tmp_path/'remote',True)
    session.run(override_gate=True);old=file_digest(session.export())
    newer=Session(replace(c,flow_steps=4),'extend',tmp_path/'local',tmp_path/'remote',True)
    result=newer.run(stages='flow',mode='extend',override_gate=True,stop_after=1);assert result['state']=='interrupted'
    partial,_=read_checkpoint(newer.export());assert partial['progress']==3 and not partial['completed'] and partial['training_status'].startswith('partial')
    result=newer.run(stages='flow',mode='resume',override_gate=True);assert result['state']=='completed'
    cp,_=read_checkpoint(newer.root/'flow/resume.pt');assert cp['progress']==4 and cp['scheduler']['settings']['horizon']==2
    # Reopening a completed extension must also retain its original schedule.
    assert newer.run(stages='flow',mode='resume',override_gate=True)['state']=='completed'
    assert any(file_digest(p)==old for p in (newer.root/'budgets').glob('*/export/coordinate.pt'))
    assert list((newer.root/'evaluations').glob('*/flow-validation.json'))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Requires actual CUDA GPU; policy unit tests are not GPU verification')
def test_actual_cuda_amp_checkpoint(tmp_path):
    c=replace(tiny(),runtime={'device':'cuda'},training_options={**tiny().training_options,'precision':'auto'})
    r=train(c.to_dict(),tmp_path/'cuda','autoencoder');assert r['state']=='completed' and r['memory']['cuda_peak_allocated']>0
