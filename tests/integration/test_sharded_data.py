from dataclasses import replace
from copy import deepcopy
import json
import numpy as np
import pytest
import torch

from pdeflow.core import TransformBundle
from pdeflow.data import generate,load_dataset
from pdeflow.data.shards import plan
from pdeflow.runtime import Config,read_json
from pdeflow.training import train,load_inference


def config():
    c=Config.load('configs/experiments/cpu_2d.json')
    return replace(c,data=replace(c.data,count=12,split_counts=(8,2,2),storage='sharded_npz',shard_size=3,bandwidth_range=(.6,2.4),amplitude_range=(.3,2),offset_range=(-1,1)),training=replace(c.training,steps=3,batch_size=4),model=replace(c.model,width=4,blocks=1))


def test_restart_split_isolation_independence_and_lazy_cache(tmp_path,monkeypatch):
    import pdeflow.data.shards as shards
    c=config();original=shards.instance;calls=[]
    def interrupted(c,seed):
        calls.append(seed)
        if len(calls)==5:raise RuntimeError('power loss simulation')
        return original(c,seed)
    monkeypatch.setattr(shards,'instance',interrupted)
    with pytest.raises(RuntimeError,match='power loss'):generate(c,tmp_path/'a')
    held=(tmp_path/'a/test_manifest.json').read_bytes();first=(tmp_path/'a/shard-000000.npz').read_bytes()
    monkeypatch.setattr(shards,'instance',original);generate(c,tmp_path/'a');generate(c,tmp_path/'b')
    assert (tmp_path/'a/test_manifest.json').read_bytes()==held
    assert (tmp_path/'a/shard-000000.npz').read_bytes()==first
    a,b=load_dataset(tmp_path/'a',c),load_dataset(tmp_path/'b',c)
    ix=list(range(12));values=a.values[ix]
    assert torch.equal(values,b.values[ix]);assert len(a.values.cache)<=2
    assert all(not torch.equal(values[i],values[j]) for i in ix for j in ix if i!=j)
    assert len(set(plan(c)['record_seeds']))==12
    flat=sum([a.indices(s) for s in ['train','validation','test']],[]);assert sorted(flat)==ix
    with pytest.raises(FileExistsError):generate(c,tmp_path/'a')
    # Only training members determine the streaming transform, independently checked.
    expected=TransformBundle.fit(values[a.indices('train')],[a.metadata['ids'][i] for i in a.indices('train')])
    actual=a.transform();assert actual.mean==pytest.approx(expected.mean,abs=1e-14);assert actual.scale==pytest.approx(expected.scale,abs=1e-14)
    assert actual.fitted_ids==expected.fitted_ids
    heldout=set(a.metadata['ids'][i] for i in a.indices('test'));assert heldout.isdisjoint(actual.fitted_ids)


def test_sharded_real_training_exact_resume_and_cpu_export(tmp_path):
    c=config();generate(c,tmp_path/'data')
    train(c,tmp_path/'data',tmp_path/'full');train(c,tmp_path/'data',tmp_path/'resumed',stop_after=1)
    train(c,tmp_path/'data',tmp_path/'resumed',resume=tmp_path/'resumed/checkpoint.pt')
    a=load_inference(tmp_path/'full/inference.pt');b=load_inference(tmp_path/'resumed/inference.pt')
    assert all(torch.equal(v,b.model.state_dict()[k]) for k,v in a.model.state_dict().items())
    assert len(read_json(tmp_path/'full/history.json'))==3


def test_immutable_test_manifest_and_shard_corruption(tmp_path):
    c=config();generate(c,tmp_path/'data');p=tmp_path/'data/test_manifest.json';p.write_text('{}')
    with pytest.raises(ValueError,match='Held-out'):load_dataset(tmp_path/'data',c)
    p.write_text(json.dumps({'ids':[plan(c)['ids'][i] for i in plan(c)['splits']['test']], 'record_seeds':[plan(c)['record_seeds'][i] for i in plan(c)['splits']['test']]}))
    (tmp_path/'data/shard-000000.npz').write_bytes(b'broken')
    data=load_dataset(tmp_path/'data',c)
    with pytest.raises(ValueError,match='integrity'):data.values[[0]]


def test_cost_probe_cannot_silently_use_heat_for_darcy():
    from pdeflow.data.shards import estimate
    c=Config.load('configs/experiments/darcy_cpu_2d.json')
    with pytest.raises(ValueError,match='heat reference probe only'):estimate(c,1)
    report=estimate(config(),1)
    assert report['channels']==1 and report['samples']==1 and report['raw_bytes']>0
