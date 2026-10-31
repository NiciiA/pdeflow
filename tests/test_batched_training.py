"""Actual optimizer/gradient and arbitrary-time frozen-cache equivalence checks."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pytest
import torch
from pdeflow.colab import training_preset
from pdeflow.coordinate.training import train,read_checkpoint
from pdeflow.coordinate.optimization import options
from pdeflow.coordinate.frozen import FrozenTargets
from pdeflow.coordinate.profiling import Timings


def config():
    torch.set_num_threads(2)
    return training_preset('smoke','cpu','fp32',{'batch_size':5,'encoder_points':24,'query_points':32,'family_weights':{'heat':.3,'wave':.7},'field_loss':{'type':'region_balanced_mse','alpha':.5,'analysis_resolution':8,'region_queries':32,'time_groups':1},'training_options':{'microbatch':3,'accumulation':2,'frozen_prefetch':1,'profile_components':True}})


@pytest.mark.parametrize('kind',['mse','weighted_mse','region_balanced_mse','mae'])
def test_batched_a1_per_example_per_quantity_losses_and_gradients(tmp_path,monkeypatch,kind):
    c=replace(config(),field_loss={**config().field_loss,'type':kind});compare(tmp_path,monkeypatch,c,'autoencoder',None)


def compare(root,monkeypatch,c,stage,dep):
    results=[];gradients=[];method=torch.optim.Adam.step
    def capture(opt,*args,**kwargs):
        gradients.append([p.grad.detach().cpu().clone() if p.grad is not None else None for group in opt.param_groups for p in group['params']]);return method(opt,*args,**kwargs)
    monkeypatch.setattr(torch.optim.Adam,'step',capture)
    traces=[]
    for execution in ('accumulated','batched'):
        start=len(gradients);cc=replace(c,training_options={**c.training_options,'execution':execution});r=train(cc.to_dict(),root/execution,stage,dep)
        results.append(read_checkpoint(r['export'])[0]);traces.append(gradients[start:])
        assert all(row['actual_forward_microbatch']==(3 if execution=='batched' else 1) for row in results[-1]['history'])
        if execution=='batched':assert all(row['forward_microbatch_sizes']==[3,2] for row in results[-1]['history'])
    for a,b in zip(results[0]['history'],results[1]['history']):
        assert a['family_example_counts']==b['family_example_counts'];assert a['training_loss']==pytest.approx(b['training_loss'],abs=2e-6,rel=3e-6)
        assert a['per_family_loss']==pytest.approx(b['per_family_loss'],abs=2e-6,rel=3e-6)
    max_gradient=0.
    for a,b in zip(*traces):
        for ga,gb in zip(a,b):
            assert (ga is None)==(gb is None)
            if ga is not None:
                torch.testing.assert_close(ga,gb,atol=2e-6,rtol=2e-4);max_gradient=max(max_gradient,float((ga-gb).abs().max()))
    key='velocity' if stage=='flow' else 'autoencoder';diff=0.
    for k in results[0][key]:
        a=results[0][key][k];b=results[1][key][k]
        if a.is_floating_point():torch.testing.assert_close(a,b,atol=2e-5,rtol=2e-4);diff=max(diff,float((a-b).abs().max()))
    evidence=Path(os.environ.get('PDEFLOW_BATCH_EVIDENCE',root/'evidence'));evidence.mkdir(parents=True,exist_ok=True)
    (evidence/(stage+'-'+c.field_loss['type']+'.json')).write_text(json.dumps({'maximum_clipped_gradient_difference':max_gradient,'maximum_parameter_difference':diff,'history':{mode:r['history'] for mode,r in zip(('accumulated','batched'),results)},'meaning':'Two-update CPU equivalence only; no accuracy claim'},indent=2))


@pytest.mark.parametrize('stage',['evolution','flow'])
def test_frozen_stages_batched_match_accumulated(tmp_path,monkeypatch,stage):
    c=config();a=train(c.to_dict(),tmp_path/'a1','autoencoder');dep=Path(a['export'])
    if stage=='flow':dep=Path(train(c.to_dict(),tmp_path/'a2','evolution',dep)['export'])
    compare(tmp_path,monkeypatch,c,stage,dep)


def test_frozen_cache_hash_once_disk_and_exact_time_keys(tmp_path,monkeypatch):
    import pdeflow.coordinate.frozen as module
    calls=[];old=module.digest
    def traced(x):calls.append(x);return old(x)
    monkeypatch.setattr(module,'digest',traced);t=Timings(torch.device('cpu'),True)
    cache=FrozenTargets({'encoder':'checkpoint-hash','normalization':list(range(1024))},8,t,tmp_path,1)
    initial_calls=len(calls);assert initial_calls==1
    item={'id':'train:1:dataset'};requests=[(item,0.),(item,.237),(item,.2370001),(item,0.)];seen=[]
    def prep(rows):seen.extend([time for _,time in rows]);return torch.tensor([[time,time+1] for _,time in rows])
    def enc(rows,values):return values[:,None,:]
    a=cache.resolve(requests,False,2,prep,enc);b=cache.resolve(requests,False,2,prep,enc)
    torch.testing.assert_close(a,b);assert seen==[0.,.237,.2370001];assert cache.stats['memory_hits']>0
    # Only primitive lookup keys are hashed after setup, never large identity.
    assert all(isinstance(x,tuple) for x in calls[initial_calls:]);cache.close()
    cache=FrozenTargets({'encoder':'checkpoint-hash','normalization':list(range(1024))},8,t,tmp_path,0)
    value=cache.resolve([(item,0.)],False,2,lambda _:pytest.fail('Unexpected reference evaluation on disk hit'),enc)
    torch.testing.assert_close(value,a[:1]);assert cache.stats['disk_hits']==1;cache.close()
    changed=FrozenTargets({'encoder':'different','normalization':list(range(1024))},8,t,None,0)
    assert changed.prefix!=cache.prefix;changed.close()


def test_microbatch_8_16_32_same_effective_batch(tmp_path):
    # Real forwards/backwards at all requested sizes, bounded smoke capacity.
    c=replace(config(),batch_size=32,ae_steps=2,evolution_steps=2,flow_steps=2,training_options={**config().training_options,'validation_every':100,'microbatch':32,'accumulation':1})
    a=train(c.to_dict(),tmp_path/'a1','autoencoder');b=train(c.to_dict(),tmp_path/'a2','evolution',a['export'])
    for stage,dep in [('autoencoder',None),('evolution',a['export']),('flow',b['export'])]:
        rows=[]
        for micro in (8,16,32):
            cc=replace(c,training_options={**c.training_options,'microbatch':micro,'accumulation':32//micro})
            r=train(cc.to_dict(),tmp_path/(stage+str(micro)),stage,dep);cp=read_checkpoint(r['export'])[0];row=cp['history'][0];rows.append(row)
            assert row['actual_forward_microbatch']==micro and row['effective_examples']==32
        for row in rows[1:]:assert row['training_loss']==pytest.approx(rows[0]['training_loss'],rel=4e-6,abs=4e-6)


def test_checkpoint_loading_and_config_hashing_stay_outside_updates(tmp_path,monkeypatch):
    import pdeflow.coordinate.engine as engine
    c=config();dep=train(c.to_dict(),tmp_path/'a1','autoencoder')['export']
    phase={'updating':False};counts={'checkpoint_load':0,'checkpoint_hash':0,'large_identity_digest':0}
    load=engine.read_checkpoint;hash_file=engine.file_digest;hash_value=engine.digest
    def read(*a,**k):assert not phase['updating'];counts['checkpoint_load']+=1;return load(*a,**k)
    def file(*a,**k):assert not phase['updating'];counts['checkpoint_hash']+=1;return hash_file(*a,**k)
    def digest(x):
        if isinstance(x,dict) and 'latent_mean' in x:assert not phase['updating'];counts['large_identity_digest']+=1
        return hash_value(x)
    monkeypatch.setattr(engine,'read_checkpoint',read);monkeypatch.setattr(engine,'file_digest',file);monkeypatch.setattr(engine,'digest',digest)
    original=engine.batching.forward
    def forward(*args,**kwargs):
        phase['updating']=True
        try:return original(*args,**kwargs)
        finally:phase['updating']=False
    monkeypatch.setattr(engine.batching,'forward',forward)
    train(c.to_dict(),tmp_path/'a2','evolution',dep)
    assert counts=={'checkpoint_load':1,'checkpoint_hash':1,'large_identity_digest':1}


@pytest.mark.parametrize('backend',['mps','cuda'])
def test_actual_accelerator_three_stages_if_available(tmp_path,backend):
    available=torch.backends.mps.is_available() if backend=='mps' else torch.cuda.is_available()
    if not available:pytest.skip(backend+' unavailable; policy tests are not execution evidence')
    c=replace(config(),runtime={'device':backend,'threads':2},training_options={**config().training_options,'precision':'fp16' if backend=='cuda' else 'fp32'})
    dep=None
    for stage in ('autoencoder','evolution','flow'):
        r=train(c.to_dict(),tmp_path/stage,stage,dep);cp=read_checkpoint(r['export'])[0];dep=r['export']
        assert all(np.isfinite(x['training_loss']) for x in cp['history'])
        assert all(x['gradients_finite'] or x['optimizer_update_skipped'] for x in cp['history'])
        assert all(x['actual_forward_microbatch']==3 for x in cp['history'])
