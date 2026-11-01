"""Staged oracle diagnostics and production conditioning: mathematical checks."""
from pathlib import Path
from dataclasses import replace
import json
import numpy as np
import pytest
import torch
from pdeflow.colab import training_preset
from pdeflow.coordinate.training import train,read_checkpoint
from pdeflow.coordinate.diagnostic_cases import capture,load_case,sensor_truth
from pdeflow.coordinate.diagnostic_metrics import sensor_metrics,spectral_error,field_metrics
from pdeflow.coordinate.diagnostics import diagnose,exact_sensors
from pdeflow.coordinate.diagnostic_inventory import oracle_model
from pdeflow.coordinate.inference import CoordinateBundle
from pdeflow.coordinate.model import ObservationEncoder
from pdeflow.coordinate.joint import path,active_mask
from pdeflow.coordinate.diagnostic_baseline import train_baseline
from pdeflow.runtime import read_json,write_json,file_digest


@pytest.fixture(scope='module')
def trained(tmp_path_factory):
    torch.set_num_threads(2);root=tmp_path_factory.mktemp('diagnostic-model')
    c=training_preset('smoke','cpu','fp32',{'encoder_points':16,'query_points':16,
        'field_loss':{'analysis_resolution':8,'region_queries':16,'time_groups':1}})
    a=train(c.to_dict(),root/'a1','autoencoder');b=train(c.to_dict(),root/'a2','evolution',a['export']);f=train(c.to_dict(),root/'b','flow',b['export'])
    return c,Path(a['export']),Path(b['export']),Path(f['export'])


def test_exact_sensor_coordinates_times_and_quantities():
    class Bundle:
        device=torch.device('cpu')
        def decode(self,z,q,p,k,f,name):return (q[:,0]+2*q[:,1]+3*q[:,2]+(10 if name=='v' else 0))[None].expand(len(z),-1)
    rows=[{'x':.2,'y':.3,'time':.17,'field':'v'},{'x':.8,'y':.1,'time':.93,'field':'u'}]
    a=exact_sensors(Bundle(),torch.zeros(2,1,1),None,None,torch.tensor([1,1]),rows)
    np.testing.assert_allclose(a,[[11.31,3.79]]*2,atol=1e-6)


def test_sensor_zero_noise_and_standardized_noise():
    rows=[{'value':1.,'sigma':0.,'time':0.,'field':'u'},{'value':2.,'sigma':.5,'time':.237,'field':'v'}]
    r=sensor_metrics([[1.,3.],[1.000001,1.]],rows,[1.,2.])
    assert r['per_sample'][0]['standardized_noisy']['rms']==2.
    assert r['ensemble_mean']['standardized_noisy']['rms']==0.
    assert r['per_sample'][1]['noiseless']['within_tolerance_fraction']==1.
    assert r['by_quantity']['v']['per_sample'][0]['mse']==1.
    assert sensor_metrics(np.empty((2,0)),[])['ensemble_mean'] is None


def test_parseval_and_physical_normalized_metrics(trained):
    c,*_=trained;x=np.arange(16)/16;a=np.sin(2*np.pi*x)[:,None]*np.ones((1,16));b=np.zeros_like(a)
    s=spectral_error(a,b);assert s['low_mode_mse']==pytest.approx(.5) and s['high_mode_mse']<1e-20
    r=field_metrics(a[None,None],b[None],[0.],c,'u',{'mean':2.,'scale':10.})
    assert r['ensemble_mean']['uniform_physical']['mse']==pytest.approx(.5)
    assert r['ensemble_mean']['uniform_physical']['relative_l2'] is None
    assert r['ensemble_mean']['constant_training_mean']['mse']==4.


def test_clean_endpoint_for_actual_straight_path(trained):
    c,*_=trained;g=torch.Generator().manual_seed(5);source=torch.randn(2,c.tokens,c.width,generator=g);target=torch.randn(source.shape,generator=g)
    s=torch.tensor([.1,.87]);known=torch.tensor([False,True]);f=torch.tensor([0,1]);state,v,mask=path(c,target,source,s,known,f)
    torch.testing.assert_close(state+(1-s[:,None,None])*v,target*mask)


def test_masked_padding_with_explicit_quantity_ids(trained):
    c,*_=trained;encoder=ObservationEncoder(c).eval();q=torch.rand(1,4,3);features=torch.rand(1,4,3);mask=torch.ones(1,4,dtype=torch.bool);ids=torch.tensor([[0,1,0,1]])
    with torch.no_grad():
        a=encoder((q,features,mask),quantity_ids=ids)
        padded=(torch.cat([q,torch.full((1,2,3),float('nan'))],1),torch.cat([features,torch.full((1,2,3),float('nan'))],1),torch.cat([mask,torch.zeros(1,2,dtype=torch.bool)],1))
        b=encoder(padded,quantity_ids=torch.cat([ids,torch.zeros(1,2,dtype=torch.long)],1))
    assert torch.isfinite(b).all();torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)


def test_capture_replay_and_stage_dependency_checks(trained,tmp_path):
    c,a1,a2,flow=trained;case=capture(flow,tmp_path/'case',validation_index=3,steps=2,particles=2,resolution=8)
    saved,p,solver,context,noise=load_case(tmp_path/'case');assert p.family=='wave' and np.any(solver.velocity)
    assert saved['observations']==case['observations'] and not set(map(str,case['heldout_observations_evaluation_only']))&set(map(str,case['observations']))
    bundle=CoordinateBundle(flow);bundle.checkpoint_path=str(flow)
    ae,lineage=oracle_model(bundle,a2,a1);assert ae is not None and lineage['explicit_a1_verified']
    with pytest.raises(ValueError,match='hash differs'):oracle_model(bundle,a1)
    r=diagnose(flow,tmp_path/'case/case.json',tmp_path/'diagnosis',stages=('a1','a2','b'),evolution=a2,representation=a1)
    assert set(r['stages'])=={'a1','a2','b'}
    assert r['stages']['a1']['quantities']['v']['per_sample'][0]['frames'][0]['time']==0.
    assert r['stages']['b']['heldout_sensor_metrics']['count']==64
    assert (tmp_path/'diagnosis/oracle-initial/initial-state-v.png').exists()
    repeat=diagnose(flow,tmp_path/'case',tmp_path/'repeat',stages=('b',))
    with np.load(tmp_path/'diagnosis/fields.npz') as aa,np.load(tmp_path/'repeat/fields.npz') as bb:np.testing.assert_array_equal(aa['b_u'],bb['b_u'])
    with pytest.raises(FileExistsError):capture(flow,tmp_path/'case',validation_index=0)


def test_missing_encoder_is_reported_not_randomly_replaced(trained,tmp_path):
    c,a1,a2,flow=trained;copied=tmp_path/'isolated.pt';copied.write_bytes(flow.read_bytes());capture(copied,tmp_path/'case',validation_index=0,steps=1,particles=1,resolution=8)
    r=diagnose(copied,tmp_path/'case',tmp_path/'report',stages=('a1','a2'))
    assert r['stages']['a1']['status']=='unavailable' and r['stages']['a2']['status']=='unavailable'
    with pytest.raises(ValueError,match='cannot be invented'):train_baseline(copied,tmp_path/'baseline',steps=1)


def test_baseline_head_updates_frozen_modules_unchanged(trained,tmp_path):
    c,a1,a2,flow=trained;before=[file_digest(p) for p in (a1,a2,flow)]
    r=train_baseline(flow,tmp_path/'baseline',a2,a1,steps=5,train_cases=2,validation_cases=2)
    assert r['successful_updates']==5 and all(np.isfinite(x['gradient_norm']) and x['gradient_norm']>0 for x in r['history'])
    assert before==[file_digest(p) for p in (a1,a2,flow)]
    assert not set(r['training_ids'])&set(r['heldout_ids'])
    assert all('fields' in x and 'sensors' in x for x in r['after'])
    assert r['history'][-1]['latent_loss']<r['history'][0]['latent_loss']


def test_case_tamper_rejected_and_sensor_truth_separate(trained,tmp_path):
    c,*_,flow=trained;capture(flow,tmp_path/'case',validation_index=0,steps=1,particles=1,resolution=8)
    case,p,solver,_,_=load_case(tmp_path/'case');truth=sensor_truth(solver,case['observations'],case['true_parameter_evaluation_only'],c.interval)
    np.testing.assert_allclose(truth,[r['value'] for r in case['observations']])
    (tmp_path/'case/noise.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='checksum'):load_case(tmp_path/'case')


def test_stage_provenance_and_colab_gate_export(trained,tmp_path):
    import shutil
    from pdeflow.colab import Session
    c,a1,a2,flow=trained;cp,_=read_checkpoint(flow)
    assert set(cp['training_provenance']['stages'])=={'autoencoder','evolution','flow'}
    assert all(s['successful_updates']==2 and s['attempted_updates']==2 for s in cp['training_provenance']['stages'].values())
    session=Session(c,'export-provenance',tmp_path/'fast',tmp_path/'persistent',True)
    (session.root/'a2').mkdir();shutil.copyfile(a2,session.root/'a2/evolution.pt')
    (session.root/'flow').mkdir();shutil.copyfile(flow.parent/'resume.pt',session.root/'flow/resume.pt')
    write_json(session.root/'representation-quality.json',{'passed':False,'mse_ratio':1.5})
    write_json(session.root/'session.json',{'gate_override':True})
    exported=session.export(persist=False);out,_=read_checkpoint(exported)
    assert out['training_provenance']['representation_gate']['failed_gate_bypassed'] is True
    assert out['diagnostic_requirements']['matching_evolution_checkpoint_sha256']==file_digest(a2)
    assert not any(k.startswith('encoder.') for k in out['inference_state'])


@pytest.mark.parametrize('index',[0,3])
def test_production_reference_isolation_with_immutable_case_comparison(trained,tmp_path,index):
    import shutil
    from pdeflow.runtime import digest
    c,a1,a2,flow=trained;capture(flow,tmp_path/'original',validation_index=index,steps=1,particles=1,resolution=8)
    shutil.copytree(tmp_path/'original',tmp_path/'changed');case=read_json(tmp_path/'changed/case.json')
    with np.load(tmp_path/'changed/reference.npz') as a:initial=a['initial'].copy();context=a['encoder_coordinates'].copy()
    initial[...,0]=initial[...,0]*2+.2
    np.savez_compressed(tmp_path/'changed/reference.npz',initial=initial,encoder_coordinates=context)
    case['files']['reference.npz']=file_digest(tmp_path/'changed/reference.npz');case['case_identity']=digest({k:v for k,v in case.items() if k!='case_identity'});write_json(tmp_path/'changed/case.json',case)
    for name in ('original','changed'):diagnose(flow,tmp_path/name,tmp_path/(name+'-report'),stages=('b',))
    with np.load(tmp_path/'original-report/fields.npz') as a,np.load(tmp_path/'changed-report/fields.npz') as b:
        np.testing.assert_array_equal(a['b_u'],b['b_u']);
        if 'b_v' in a.files:np.testing.assert_array_equal(a['b_v'],b['b_v'])
        assert not np.allclose(a['reference_u'],b['reference_u'])


def test_unknown_known_masks_sigma_units_and_permutation(trained):
    from pdeflow.coordinate.conditioning import prepare
    from pdeflow.coordinate.physics import problem
    from pdeflow.core import GridSpec,DomainSpec
    c,a1,a2,flow=trained;bundle=CoordinateBundle(flow);grid=GridSpec((8,8),DomainSpec(c.lengths,c.origins),(0.,1.))
    rows=[{'field':'v','x':.2,'y':.3,'time':.19,'value':2.,'sigma':.05},{'field':'u','x':.7,'y':.4,'time':.73,'value':-.3,'sigma':0.}]
    p=problem(grid,'wave',None);a=prepare(c,bundle.cp['normalizations'],p,rows,1,'cpu');b=prepare(c,bundle.cp['normalizations'],p,list(reversed(rows)),1,'cpu')
    assert not a.known.any() and not a.parameter.any()
    norm=bundle.cp['normalizations']['wave'][1];assert float(a.observations[1][0,0,1])==pytest.approx(.05/norm['scale'])
    with torch.no_grad():
        graph=bundle.ae.graph_encoder(a.families);ca=bundle.velocity.condition(a.parameter,a.observations,a.known,a.families,graph);cb=bundle.velocity.condition(b.parameter,b.observations,b.known,b.families,graph)
    torch.testing.assert_close(ca[0],cb[0],atol=2e-6,rtol=2e-5)
    known=prepare(c,bundle.cp['normalizations'],problem(grid,'wave',.4),rows,1,'cpu')
    assert known.known.all() and not active_mask(c,known.known,known.families)[:,-1].any()


def test_missing_or_unexpected_trained_tensors_are_not_substituted(trained,tmp_path):
    c,a1,a2,flow=trained;cp,_=read_checkpoint(flow);cp['inference_state']=dict(cp['inference_state']);cp['inference_state'].pop(next(iter(cp['inference_state'])))
    torch.save(cp,tmp_path/'missing.pt')
    with pytest.raises(RuntimeError,match='Missing key'):CoordinateBundle(tmp_path/'missing.pt')


def test_initial_information_control_uses_measurements_not_hidden_truth(trained):
    from pdeflow.coordinate.diagnostic_information import initial_fourier_check
    from pdeflow.coordinate.physics import problem
    from pdeflow.numerics.dynamic import SpectralState
    from pdeflow.core import GridSpec,DomainSpec
    c,*_=trained;grid=GridSpec((8,8),DomainSpec(c.lengths,c.origins),(0.,1.));x,y=np.meshgrid(np.arange(8)/8,np.arange(8)/8,indexing='ij');u=.2+.8*np.cos(2*np.pi*x);solver=SpectralState(u);rng=np.random.default_rng(4);rows=[]
    for x,y in rng.random((24,2)):rows.append({'field':'u','x':float(x),'y':float(y),'time':0.,'value':float(.2+.8*np.cos(2*np.pi*x)),'sigma':0.})
    a,r=initial_fourier_check(c,problem(grid,'heat',None),rows,solver,.07)
    assert r['full_column_rank'] and r['sensor_mse']<1e-24
    np.testing.assert_allclose(a,u,atol=1e-12)
    other=SpectralState(np.ones((8,8))*5);b,s=initial_fourier_check(c,problem(grid,'heat',None),rows,other,.07)
    np.testing.assert_array_equal(a,b);assert s['initial_field_mse_evaluation_only']>1.


def test_playground_sensor_view_uses_saved_predictions_without_new_inference(trained,tmp_path):
    from fastapi.testclient import TestClient
    from pdeflow.workbench.api import create_app
    from pdeflow.coordinate.inference import reconstruct
    c,a1,a2,flow=trained;case=capture(flow,tmp_path/'case',validation_index=3,steps=1,particles=2,resolution=8)
    root=tmp_path/'jobs'/('a'*32);root.mkdir(parents=True);result=root/'coordinate-result'
    reconstruct(flow,case['problem'],case['observations'],result,steps=1,particles=2)
    write_json(root/'job.json',{'id':'a'*32,'status':'succeeded','created_at':0.})
    before=file_digest(result/'metrics.json')
    app=create_app(workspace=tmp_path/'workspace',output_root=tmp_path/'jobs',data_root=tmp_path/'data')
    with TestClient(app) as client:
        response=client.get('/api/playground/results/'+('a'*32));assert response.status_code==200,response.text
        r=response.json();sensor=r['metrics']['sensor_diagnostics'];original=r['metrics']['observation_consistency']['predictions_per_sample'];y=np.array([r['value'] for r in case['observations']])
        assert sensor['ensemble_mean']['mse']==pytest.approx(np.mean((np.asarray(original).mean(0)-y)**2))
        assert r['comparison'] is None and set(sensor['by_quantity'])=={'u','v'}
    assert file_digest(result/'metrics.json')==before


def test_observation_only_correction_gradients_masks_and_reference_isolation(trained,tmp_path):
    import shutil
    from pdeflow.coordinate.diagnostic_correction import correct
    from pdeflow.runtime import digest
    c,a1,a2,flow=trained;capture(flow,tmp_path/'case',validation_index=0,steps=1,particles=1,resolution=8)
    case=read_json(tmp_path/'case/case.json');case['observations']=[r for r in case['observations'] if r['time']==0.];case['problem']['known_parameters']={};case['problem']['unknown_parameters']=['diffusivity'];case['case_identity']=digest({k:v for k,v in case.items() if k!='case_identity'});write_json(tmp_path/'case/case.json',case)
    shutil.copytree(tmp_path/'case',tmp_path/'changed')
    with np.load(tmp_path/'changed/reference.npz') as a:initial=a['initial'].copy();context=a['encoder_coordinates'].copy()
    initial[...,0]*=-2
    np.savez_compressed(tmp_path/'changed/reference.npz',initial=initial,encoder_coordinates=context)
    changed=read_json(tmp_path/'changed/case.json');changed['files']['reference.npz']=file_digest(tmp_path/'changed/reference.npz');changed['case_identity']=digest({k:v for k,v in changed.items() if k!='case_identity'});write_json(tmp_path/'changed/case.json',changed)
    outputs=[]
    for name in ('case','changed'):
        r=correct(flow,tmp_path/name,tmp_path/(name+'-corrected'),steps=2)
        assert r['frozen_state_sha256_before']==r['frozen_state_sha256_after'] and not r['reference_used_in_optimization']
        assert r['before']['parameters']==r['after']['parameters']  # t=0 cannot inform D
        assert all(x['gradient_norm']>0 and np.isfinite(x['gradient_norm']) for x in r['history'])
        outputs.append(tmp_path/(name+'-corrected')/'fields.npz')
    with np.load(outputs[0]) as a,np.load(outputs[1]) as b:
        np.testing.assert_array_equal(a['corrected_standardized_latent'],b['corrected_standardized_latent'])
        assert np.max(abs(a['corrected_standardized_latent']-a['initial_standardized_latent']))>0
    with pytest.raises(ValueError,match='1..40'):correct(flow,tmp_path/'case',tmp_path/'too-long',steps=41)


def test_explicit_noiseless_training_mass_and_gradients(trained,tmp_path):
    from pdeflow.coordinate.engine import observation_batch
    from pdeflow.coordinate.dataset import example
    from pdeflow.coordinate.optimization import representation_contract
    c,a1,a2,flow=trained;cp,_=read_checkpoint(flow);item=example(c,'train',0);positive=replace(c,joint={**c.joint,'noiseless_probability':0.}).validate();zero=replace(c,joint={**c.joint,'noiseless_probability':1.}).validate()
    assert representation_contract(positive)==representation_contract(zero)
    for value,config in [(False,positive),(True,zero)]:
        obs=observation_batch(config,[item],torch.Generator().manual_seed(194),cp['normalizations'],torch.device('cpu'));valid=obs[2]
        assert bool((obs[1][...,1][valid]==0).all()) is value
        encoder=ObservationEncoder(config);encoder(obs).square().mean().backward();grad=encoder.input[0].weight.grad[:,-1]
        assert bool((grad.abs().sum()>0)) is value
    mixed=replace(c,joint={**c.joint,'noiseless_probability':.5}).validate();seen=[]
    g=torch.Generator().manual_seed(9)
    for _ in range(16):
        obs=observation_batch(mixed,[item],g,cp['normalizations'],torch.device('cpu'));seen.extend(obs[1][...,1][obs[2]].eq(0).tolist())
    assert any(seen) and not all(seen)
    # Existing absent/explicit0 policies keep EXACT RNG draws.
    absent=replace(c,joint={k:v for k,v in c.joint.items() if k!='noiseless_probability'})
    a=observation_batch(absent,[item],torch.Generator().manual_seed(5),cp['normalizations'],torch.device('cpu'));b=observation_batch(positive,[item],torch.Generator().manual_seed(5),cp['normalizations'],torch.device('cpu'))
    for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=0,atol=0)
    ga=torch.Generator().manual_seed(7);gb=torch.Generator().manual_seed(7)
    a=observation_batch(positive,[item],ga,cp['normalizations'],torch.device('cpu'));b=observation_batch(mixed,[item],gb,cp['normalizations'],torch.device('cpu'))
    torch.testing.assert_close(ga.get_state(),gb.get_state(),rtol=0,atol=0)
    for index in (0,2,3):torch.testing.assert_close(a[index],b[index],rtol=0,atol=0)
    with pytest.raises(ValueError,match='including zero'):replace(c,noise_range=(.01,.05),joint={**c.joint,'noiseless_probability':.1}).validate()
    with pytest.raises(ValueError,match='Incompatible model/data changes'):train(mixed.to_dict(),flow.parent,'flow',resume=True)


def test_wave_value_shuffle_preserves_quantities_and_coordinates():
    from pdeflow.coordinate.diagnostics import shuffle_within_quantity
    rows=[{'field':name,'x':i/8,'y':.3,'time':.237,'value':float(i+(100 if name=='v' else 0)),'sigma':0.} for i in range(8) for name in ('u','v')]
    shuffled=shuffle_within_quantity(rows,np.random.default_rng(194))
    for name in ('u','v'):assert sorted(r['value'] for r in rows if r['field']==name)==sorted(r['value'] for r in shuffled if r['field']==name)
    assert [(r['field'],r['x'],r['y'],r['time']) for r in rows]==[(r['field'],r['x'],r['y'],r['time']) for r in shuffled]


def test_exact_training_observation_capture_roundtrips_shared_packing(trained,tmp_path):
    from pdeflow.coordinate.engine import observation_batch
    from pdeflow.coordinate.conditioning import prepare
    from pdeflow.coordinate.dataset import example
    from pdeflow.core import ProblemSpec
    c,a1,a2,flow=trained;cp,_=read_checkpoint(flow)
    case=capture(flow,tmp_path/'training-policy',validation_index=3,steps=1,particles=1,resolution=8,training_observations=True)
    expected=observation_batch(c,[example(c,'validation',3)],torch.Generator().manual_seed(91),cp['normalizations'],torch.device('cpu'))
    actual=prepare(c,cp['normalizations'],ProblemSpec.from_dict(case['problem']),case['observations'],1,torch.device('cpu')).observations
    for a,b in zip(expected,actual):torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-7)
    assert case['observation_policy']=='production_training_sampler'
    with pytest.raises(ValueError,match='never regenerated'):capture(flow,tmp_path/'invalid',job='not-opened',training_observations=True)
