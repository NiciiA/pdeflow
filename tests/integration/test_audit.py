from dataclasses import replace
import numpy as np
import pytest
import torch

from pdeflow.cli import reconstruct
from pdeflow.core import TransformBundle
from pdeflow.equations import HeatPhysics
from pdeflow.data import generate
from pdeflow.runtime import atomic_write, Config
from pdeflow.training import train, _read_checkpoint


def test_reference_cannot_change_samples(tmp_path):
    config=Config.load('configs/experiments/cpu_2d.json')
    config=replace(config,training=replace(config.training,steps=2),inference=replace(config.inference,steps=2))
    generate(config,tmp_path/'data');train(config,tmp_path/'data',tmp_path/'train')
    wrong=tmp_path/'wrong.npz';atomic_write(wrong,lambda f:np.savez(f,values=np.ones((1,1,4,8,8))*500))
    references=[None,tmp_path/'data/test_reference.npz',wrong]
    for j,reference in enumerate(references):
        reconstruct(config,tmp_path/'train/inference.pt',tmp_path/'data/test_problem.json',tmp_path/'data/test_observations.json',tmp_path/f'run{j}',reference)
    for mode in ('unguided','observation','observation_pde','fourier_inverse'):
        arrays=[np.load(tmp_path/f'run{j}'/mode/'result.npz')['samples'] for j in range(3)]
        assert np.array_equal(arrays[0],arrays[1]) and np.array_equal(arrays[1],arrays[2])
    import json
    a=json.loads((tmp_path/'run1/observation_pde/metrics.json').read_text());b=json.loads((tmp_path/'run2/observation_pde/metrics.json').read_text())
    assert a['relative_l2_unobserved_mean']!=b['relative_l2_unobserved_mean']


def test_physical_residual_decoding_has_correct_scale():
    config=Config.load('configs/experiments/cpu_2d.json');g=config.grid
    t=torch.tensor(g.times,dtype=torch.float64)
    q=t.square().reshape(1,1,4,1,1).expand(1,1,4,8,8).clone().requires_grad_()
    transform=TransformBundle((5.,),(3.,),('training-only',))
    r=HeatPhysics(g,torch.tensor([.07])).evaluate(transform.decode(q))
    expected=(6*t[1:-1]).reshape(1,1,2,1,1).expand_as(r.pointwise['pde'])
    assert torch.allclose(r.pointwise['pde'],expected,atol=1e-12)
    grad,=torch.autograd.grad(r.mean_square(),q);assert grad.norm()>0


def test_incomplete_checkpoint_is_rejected(tmp_path):
    path=tmp_path/'incomplete.pt';path.write_bytes(b'PK\x03\x04partial')
    with pytest.raises(ValueError,match='incomplete'):_read_checkpoint(path)
