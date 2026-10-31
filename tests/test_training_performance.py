"""Numerical equivalence, cache identities and native precision; not GPU claims."""
from dataclasses import replace
import time
import numpy as np
import pytest
import torch
from pdeflow.colab import training_preset
from pdeflow.coordinate.dataset import example,state_samples,Dataset,family_for
from pdeflow.coordinate.optimization import precision
from pdeflow.coordinate.profiling import Timings
from pdeflow.coordinate.graph import GraphEncoder
from pdeflow.field_losses import grouped_queries,AnalysisCache,differentiable_loss
from pdeflow.coordinate.training import train,read_checkpoint


def cfg(**kwargs):
    return replace(training_preset('smoke','cpu','fp32'),reference_shape=(32,32),encoder_points=64,query_points=64,
                   distribution={'max_wave':3,'width':[.1,.2]},field_loss={'type':'region_balanced_mse','alpha':.5,'analysis_resolution':16,'region_queries':64,'time_groups':1},**kwargs)

@pytest.mark.parametrize('index',range(12))
def test_reference_frames_arbitrary_times_and_state(index):
    c=cfg();old=example(replace(c,training_options={**c.training_options,'performance_mode':'legacy'}),'train',index);new=example(c,'train',index)
    assert len(new['solver'].modes)==49 and len(old['solver'].modes)==1089
    assert family_for(c,'train',index)==new['family']
    q=np.random.default_rng(83).random((61,3));q[:7,2]=0;q[7:14,2]=.237;q[14:21,2]=1
    for quantity in ('u','v') if new['family']=='wave' else ('u',):
        np.testing.assert_allclose(new['solver'].evaluate(q,new['parameter'],c.interval,quantity),old['solver'].evaluate(q,old['parameter'],c.interval,quantity),atol=3e-13,rtol=2e-12)
        for t in (0.,.137329,1.):
            np.testing.assert_allclose(new['solver'].frame(t,new['parameter'],c.interval,(17,23),quantity),old['solver'].frame(t,old['parameter'],c.interval,(17,23),quantity),atol=3e-13,rtol=2e-12)
    torch.testing.assert_close(state_samples(old,old['context'],.237,c),state_samples(new,new['context'],.237,c),atol=1e-7,rtol=1e-7)

@pytest.mark.parametrize('index',[0,1,4,5,10,11])
def test_region_queries_loss_and_gradients_equal(index):
    c=cfg();old=example(replace(c,training_options={**c.training_options,'performance_mode':'legacy'}),'train',index);new=example(c,'train',index)
    from pdeflow.coordinate.dataset import item_quantity
    for quantity in ('u','v') if new['family']=='wave' else ('u',):
        results=[]
        for item in (old,new):
            q,y,w,_=grouped_queries(c,[item_quantity(item,quantity)],torch.Generator().manual_seed(891),{'mean':.1,'scale':.7},AnalysisCache(),[.237])
            pred=torch.linspace(-.2,.5,y.numel()).reshape_as(y).requires_grad_();loss=differentiable_loss(pred,y,w,'region_balanced_mse');grad=torch.autograd.grad(loss,pred)[0];results.append((q,y,w,loss,grad))
        for a,b in zip(*results):torch.testing.assert_close(a,b,atol=2e-7,rtol=2e-6)


def test_no_invalid_declared_bandlimit_and_zero_wave_modes():
    from pdeflow.numerics.dynamic import SpectralState
    n=16;x=np.arange(n)[:,None]/n;u=np.broadcast_to(np.cos(8*np.pi*x),(n,n))
    with pytest.raises(ValueError,match='discard physical signal'):SpectralState(u,mode_limit=1)
    s=SpectralState(np.ones((n,n))*2,np.ones((n,n))*3,family='wave',mode_limit=2)
    q=np.array([[.23,.71,.237],[.64,.31,1.]])
    np.testing.assert_allclose(s.evaluate_state(q,.4),np.c_[2+3*q[:,2],np.full(2,3.)],atol=1e-13)


def test_precision_native_policy(monkeypatch):
    from contextlib import nullcontext
    monkeypatch.setattr(torch.cuda,'device',lambda _:nullcontext())
    monkeypatch.setattr(torch.cuda,'is_bf16_supported',lambda including_emulation=True:including_emulation)
    assert precision('auto',torch.device('cuda'))==('fp16',torch.float16)
    with pytest.raises(ValueError,match='native BF16'):precision('bf16',torch.device('cuda'))
    monkeypatch.setattr(torch.cuda,'is_bf16_supported',lambda including_emulation=True:True)
    assert precision('auto',torch.device('cuda'))==('bf16',torch.bfloat16)


def test_graph_autocast_dtype_and_finite_gradient():
    torch.manual_seed(42);m=GraphEncoder(16)
    with torch.autocast('cpu',dtype=torch.bfloat16):out=m(torch.tensor([0,1]));loss=out.float().square().mean()
    loss.backward();assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_step_timing_not_cumulative_or_double_counted():
    t=Timings(torch.device('cpu'),True)
    for _ in range(2):
        t.reset()
        with t.measure('parent',host=True):
            with t.measure('child',host=True):time.sleep(.002)
        a=t.report();b=t.report()
        assert a['cumulative_host_inclusive_seconds']==b['cumulative_host_inclusive_seconds']
        assert sum(a['host_exclusive_seconds'].values())==pytest.approx(a['host_inclusive_seconds']['parent'],abs=.0001)
    assert a['cumulative_host_inclusive_seconds']['child']>a['host_inclusive_seconds']['child']


def test_frozen_cache_references_and_conditioning_fresh(tmp_path):
    c=cfg(ae_steps=2,evolution_steps=2,flow_steps=2)
    # all initial: A2 repeatedly requests identical complete states; observations
    # retain their own non-degenerate continuous-time sampling and fresh noise.
    c=replace(c,time_sampling={'encoder':{'initial':1.-1e-8,'early':0.,'uniform':1e-8,'final':0.},'query':{'initial':1.-1e-8,'early':0.,'uniform':1e-8,'final':0.}},training_options={**c.training_options,'profile_components':True,'frozen_cache_items':2})
    torch.set_num_threads(2)
    for stage,dep in [('autoencoder',None),('evolution',tmp_path/'autoencoder/autoencoder.pt'),('flow',tmp_path/'evolution/evolution.pt')]:
        r=train(c.to_dict(),tmp_path/stage,stage,dep);cp,_=read_checkpoint(r['export'])
        assert all(x['gradients_finite'] for x in cp['history'])
        assert all(x['end_to_end_seconds']>=x['step_seconds'] for x in cp['history'])
        if stage=='evolution':assert cp['history'][-1]['frozen_cache']['hits_cumulative']>0
        assert cp['history'][-1]['frozen_cache']['items']<=2
    assert len(list((tmp_path/'flow/latent-targets').rglob('*.pt')))<=2
    from pdeflow.coordinate.engine import observation_batch
    item=example(c,'train',1);ns=Dataset(c,tmp_path/'autoencoder',False).metadata['normalizations'];g=torch.Generator().manual_seed(91)
    a=observation_batch(c,[item],g,ns,torch.device('cpu'));b=observation_batch(c,[item],g,ns,torch.device('cpu'))
    assert a[0].shape!=b[0].shape or not torch.equal(a[0],b[0])


def test_batched_heat_wave_states_and_arbitrary_time():
    from pdeflow.coordinate.dataset import state_samples_batch
    c=cfg();items=[example(c,'train',i) for i in range(8)];times=[.237,.127,.3,0.,1.,.001,.71,.85321]
    actual=state_samples_batch(items,times,c)
    expected=torch.stack([state_samples(item,item['context'],t,c) for item,t in zip(items,times)])
    torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-7)


def test_real_flow_microbatch_loss_and_updates(tmp_path):
    torch.set_num_threads(2);c=cfg(ae_steps=2,evolution_steps=2,flow_steps=2)
    train(c.to_dict(),tmp_path/'a1','autoencoder');train(c.to_dict(),tmp_path/'a2','evolution',tmp_path/'a1/autoencoder.pt')
    cp=[]
    for micro in (1,2):
        other=replace(c,training_options={**c.training_options,'microbatch':micro,'accumulation':2//micro})
        r=train(other.to_dict(),tmp_path/str(micro),'flow',tmp_path/'a2/evolution.pt');cp.append(read_checkpoint(r['export'])[0])
        assert all(x['actual_forward_microbatch']==micro for x in cp[-1]['history'])
    for a,b in zip(cp[0]['history'],cp[1]['history']):assert a['training_loss']==pytest.approx(b['training_loss'],rel=2e-6,abs=2e-6)
    for key in cp[0]['velocity']:torch.testing.assert_close(cp[0]['velocity'][key],cp[1]['velocity'][key],atol=1e-5,rtol=1e-4)


def test_disposable_target_cache_not_uploaded(tmp_path):
    from pdeflow.persistence import publish
    local=tmp_path/'local';(local/'latent-targets').mkdir(parents=True);(local/'resume.pt').write_bytes(b'optimizer checkpoint');(local/'latent-targets/cached.pt').write_bytes(b'recomputable target')
    r=publish(local,tmp_path/'persistent');assert 'resume.pt' in r['files'];assert 'latent-targets/cached.pt' not in r['files'];assert (local/'latent-targets/cached.pt').is_file()
