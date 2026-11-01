"""Known continuous sensor times, independent of A2 targets and display frames."""
from dataclasses import replace
import numpy as np
import pytest
import torch
from pdeflow.observation_times import continuous,explicit,settings,focused_positions
from pdeflow.colab import training_preset
from pdeflow.coordinate.engine import observation_batch
from pdeflow.coordinate.dataset import example
from pdeflow.coordinate.training import checkpoint_config,read_checkpoint,train
from pdeflow.coordinate.contracts import validate_rows
from pdeflow.coordinate.temporal import sample_times,evolution_time
from pdeflow.coordinate.model import Autoencoder,Velocity
from pdeflow.coordinate.conditioning import prepare,physical_rows
from pdeflow.coordinate.observation_objective import decode_points
from pdeflow.coordinate.physics import problem
from pdeflow.core import GridSpec,DomainSpec
from pdeflow.workbench.playground import Heat,Measurements,reference,measurements,_reference
from pdeflow.runtime import read_json,write_json


def tiny(**kw):
    torch.set_num_threads(2)
    return training_preset('pipeline-check','cpu','fp32',{'encoder_points':16,'query_points':16,**kw})


def test_continuous_strata_coverage_replay_small_counts_and_configured_interval():
    t=continuous(200,(.2,.9),np.random.default_rng(7))
    np.testing.assert_array_equal(t,continuous(200,(.2,.9),np.random.default_rng(7)))
    assert len(set(t))==200 and t.min()<.21 and t.max()>.89
    assert len(set(np.floor((t-.2)/.7*200)))==200
    assert not np.allclose(t,np.sort(t)) and np.all((t>.2)&(t<.9))
    small=np.array([continuous(1,(.2,.9),np.random.default_rng(k))[0] for k in range(200)])
    assert small.min()<.22 and small.max()>.88
    p=settings({'policy':'continuous_mixture','whole':.6,'early':.3,'initial':.1})
    with pytest.raises(ValueError,match='starting at zero'):continuous(5,(.2,.9),np.random.default_rng(1),p)
    a=continuous(1000,(0,2),np.random.default_rng(12),p)
    assert (a==0).sum()>50 and (a>1).sum()>200 and ((a>0)&(a<.001)).any()
    for raw in ({'policy':'continuous_mixture','whole':0.,'early':1.},{'early':.1},{'placement_groups':0}):
        with pytest.raises(ValueError):settings(raw)


def test_explicit_total_vs_per_time_validation():
    assert list(explicit([.02,.2,.47,.83],7,(0,1)))==[.02,.2,.47,.83,.02,.2,.47]
    assert list(explicit([.02,.2,.47],6,(0,1),[1,2,3]))==[.02,.2,.2,.47,.47,.47]
    for t,n,counts in (([],3,[]),([-.1],2,[]),([1.1],2,[]),([.2,.2],2,[]),([.2],2,[3]),([.2],2,[2.0])):
        with pytest.raises(ValueError):explicit(t,n,(0,1),counts)


def test_training_wave_heat_actual_values_and_independent_policies():
    c=tiny(observation_counts=(48,48),noise_range=(0.,0.),unconditional_probability=0.,observation_placement={'heat':1.})
    items=[example(c,'train',i) for i in (0,1)];norms={'heat':[{'mean':.2,'scale':2.}], 'wave':[{'mean':-.1,'scale':3.},{'mean':.4,'scale':4.}]}
    components={};obs=observation_batch(c,items,torch.Generator().manual_seed(9),norms,torch.device('cpu'),components)
    for b,item in enumerate(items):
        rows=physical_rows(obs,norms,item['family'],b);q=obs[0][b].numpy()
        assert len(rows)==48 and q[:,2].min()<.03 and q[:,2].max()>.97 and len(set(q[:,2]))==48
        assert not np.isin(q[:,2],c.times).any()
        for name in ('u','v')[:1+(item['family']=='wave')]:
            ix=np.array([r['field']==name for r in rows]);target=item['solver'].evaluate(q[ix],item['parameter'],c.interval,name)
            np.testing.assert_allclose([rows[i]['value'] for i in np.flatnonzero(ix)],target,atol=2e-6)
        assert any(abs(r['value']-item['solver'].evaluate([[r['x'],r['y'],0]],item['parameter'],c.interval,r['field'])[0])>1e-3 for r in rows)
    assert set(obs[3][1].tolist())=={0,1}
    assert 'b_observation_time_bin_7_fraction' in components['_measurements']
    # A2 still independently retains exact identity + early + continuous coverage.
    t=np.array([evolution_time(c,i,torch.Generator().manual_seed(i+100)) for i in range(600)])
    assert (t==0).sum()>80 and (t>.8).sum()>40 and ((t>0)&(t<.001)).sum()>30 and len(set(t))>300


def test_legacy_checkpoint_sampling_preserved_and_new_policy_not_inherited():
    c=tiny();raw=c.to_dict();raw.pop('observation_time')
    old=checkpoint_config({'config':raw,'engine_version':5})
    assert old.observation_time['policy']=='legacy_mixture_v1'
    # Historical default has exact 15% initial, 5% final atoms.
    t=sample_times(old,1000,torch.Generator().manual_seed(4),'observation')
    assert (t==0).sum()==150 and (t==1).sum()==50
    with pytest.raises(ValueError,match='Missing recorded'):checkpoint_config({'config':raw,'engine_version':6})
    assert c.observation_time['policy']=='continuous_uniform'


def test_bounded_time_grouping_records_approximation_not_sensor_rounding():
    t=np.array([.021,.031,.29,.31,.91]);before=t.copy();calls=[]
    def at(n,time,k):calls.append(time);return np.zeros((n,2)),{'background_count':max(1,n//4)}
    xy,meta=focused_positions(t,(0,1),4,at)
    np.testing.assert_array_equal(t,before);assert len(calls)==3 and calls[0]==pytest.approx(.026)
    assert meta['max_groups']==4 and sum(r['count'] for r in meta['time_designs'])==5


@pytest.mark.parametrize('family,quantity',[('heat','u'),('wave','u'),('wave','v')])
def test_playground_continuous_explicit_manual_roundtrip_and_exact_reference(tmp_path,family,quantity):
    r=reference(Heat(family=family,preset='travelling' if family=='wave' else 'cosine'),tmp_path)
    p,solver=_reference(str(tmp_path),r['id'])
    for policy in ('continuous','chosen','timestamped_manual'):
        n=19 if policy!='timestamped_manual' else 2
        request=Measurements(reference_id=r['id'],quantity=quantity,count=n,pattern=policy,placement='heat',times=[.02,.2,.47,.83],positions=[[.2,.3],[.2,.3]],position_times=[.12,.77])
        a=measurements(request,tmp_path);b=measurements(request,tmp_path)
        assert a['rows']==b['rows'] and len(a['rows'])==n
        times=[row['time'] for row in a['rows']]
        if policy=='continuous':assert len(set(times))==n and min(times)<.06 and max(times)>.94
        elif policy=='chosen':assert set(times)=={.02,.2,.47,.83}
        else:assert times==[.12,.77]
        q=np.array([[row['x'],row['y'],row['time']] for row in a['rows']]);expected=solver.evaluate(q,next(iter(p.known_parameters.values())),(0,1),quantity)
        np.testing.assert_allclose([row['value'] for row in a['rows']],expected,atol=1e-14)
        stored=read_json(tmp_path/r['id']/'samplings'/a['id']/'sampling.json');assert stored==a
        validate_rows(tiny(),stored['rows'],p)
    with pytest.raises(ValueError,match='own known timestamp'):measurements(Measurements(reference_id=r['id'],pattern='timestamped_manual',count=2,positions=[[.1,.2],[.2,.3]]),tmp_path)
    with pytest.raises(ValueError,match='time'):validate_rows(tiny(),[{'field':'u','x':.2,'y':.3,'value':1}])
    with pytest.raises(ValueError,match='requires an explicit physical time'):measurements(Measurements(reference_id=r['id'],pattern='current'),tmp_path)


def test_imported_static_or_trajectory_not_silently_re_solved(tmp_path):
    r=reference(Heat(),tmp_path);path=tmp_path/r['id']/'reference.json';m=read_json(path);m['diagnostics']['solver']='external';write_json(path,m)
    with pytest.raises(ValueError,match='no declared continuous-time'):measurements(Measurements(reference_id=r['id']),tmp_path)


def test_timestamp_feature_permutation_padding_and_exact_decoder_gradients():
    c=tiny();torch.manual_seed(19);ae=Autoencoder(c).eval().requires_grad_(False);flow=Velocity(c).eval();f=torch.tensor([1]);p=torch.tensor([.4]);known=torch.tensor([True]);z=torch.randn(1,c.tokens,c.width,requires_grad=True)
    q=torch.tensor([[[.2,.3,.02],[.4,.7,.47],[.6,.1,.83]]]);feat=torch.tensor([[[.1,.1,1.],[.2,.1,1.],[.3,.1,1.]]]);mask=torch.ones(1,3,dtype=torch.bool);ids=torch.tensor([[0,1,0]]);obs=(q,feat,mask,ids)
    s=torch.tensor([.5]);graph=ae.graph_encoder(f)
    original=flow(z,s,p,obs,known,f,graph);permutation=torch.tensor([2,0,1]);other=flow(z,s,p,tuple(x[:,permutation] for x in obs),known,f,graph)
    torch.testing.assert_close(original,other,atol=2e-6,rtol=2e-6)
    moved=q.clone();moved[:,:,2]=0;changed=flow(z,s,p,(moved,feat,mask,ids),known,f,graph);assert (changed-original).abs().max()>1e-7
    padded=tuple(torch.cat([x,x[:,:1]],1) for x in obs);padded[0][:,-1]=float('nan');padded[1][:,-1]=float('nan');padded[2][:,-1]=False;padded[3][:,-1]=99
    torch.testing.assert_close(original,flow(z,s,p,padded,known,f,graph),atol=2e-6,rtol=2e-6)
    decoded=decode_points(ae,z,q,ids,p,known,f)
    expected=torch.cat([ae.decode(z,q[0,i:i+1],p,known,('u','v')[int(ids[0,i])],f) for i in range(3)],1)
    torch.testing.assert_close(decoded,expected,atol=1e-6,rtol=1e-6)
    decoded.square().sum().backward();assert z.grad.abs().sum()>0 and all(x.grad is None for x in ae.parameters())


def test_tiny_joint_stages_log_time_policies_resume_compatibility(tmp_path):
    c=tiny(observation_counts=(4,4),unconditional_probability=0.)
    a=train(c.to_dict(),tmp_path/'a1','autoencoder');e=train(c.to_dict(),tmp_path/'a2','evolution',a['export']);b=train(c.to_dict(),tmp_path/'b','flow',e['export']);cp,_=read_checkpoint(b['export'])
    assert cp['engine_version']==6 and cp['observation_time_policy']['policy']=='continuous_uniform'
    assert cp['a2_target_time_policy']['mixture']['uniform']==.35
    assert cp['capabilities']['temporal']['observation_time_policy']==c.observation_time
    assert all('b_observation_time_mean' in row['components'] for row in cp['history'])
    assert all('a2_tb_early_fraction' in row['components'] for row in read_checkpoint(e['export'])[0]['history'])
    changed=replace(c,observation_time={'policy':'continuous_mixture','whole':.6,'early':.3,'initial':.1}).validate()
    with pytest.raises(ValueError,match='observation_time'):train(changed.to_dict(),tmp_path/'b','flow',resume=True)
