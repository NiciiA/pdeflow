"""Physical/geometry/sampling contracts; no reference prediction inside inference."""
from dataclasses import replace
import pytest
import torch
from pdeflow.mesh.contracts import preset,CANDIDATES
from pdeflow.mesh.model import MeshModel
from pdeflow.mesh.geometry import Geometry,regular,displacement
from pdeflow.mesh.sampling import flow_path,flow_endpoint,diffusion_path,diffusion_endpoint,diffusion_schedule,generate,fgn_noise
from pdeflow.mesh.objectives import fair_crps,fair_energy,loss
from pdeflow.mesh.state import enforce,active_mask,parameter_physical
from pdeflow.mesh.data import Data
from pdeflow.mesh.training import draw,slice_noise,ownership,train_stage
from pdeflow.mesh.provenance import read_checkpoint,counts


def config(name='mesh_m2_v1',**overrides):
    torch.set_num_threads(2);return preset('pipeline-check',name,'cpu','fp32',{'data':{'query_points':12},**overrides})


def test_periodic_neighbors_cell_quadrature_and_cross_scale():
    c=config();g=Geometry(c);corner=g.neighbors[0].tolist();assert 3 in corner and 12 in corner
    assert 0 not in corner and len(set(corner))==c.neighbors
    x=torch.ones(2,c.nodes,c.width,requires_grad=True);out=g.restrict(x);torch.testing.assert_close(out,torch.ones_like(out));torch.testing.assert_close(g.prolong(out),x);out.sum().backward();assert torch.isfinite(x.grad).all()
    torch.testing.assert_close(displacement(torch.tensor([.99,.99]),torch.tensor([.01,.01])),torch.tensor([.02,.02]),atol=1e-7,rtol=1e-5)


@pytest.mark.parametrize('name',CANDIDATES)
def test_real_candidate_gradients_masks_and_initial_identity(tmp_path,name):
    c=config(name);m=MeshModel(c);d=Data(c,tmp_path/name);batch=d.prepare(0);g=torch.Generator().manual_seed(17);noise,s,k,q=draw(c,c.batch_size,g,'cpu')
    for stage in c.kind['stages']:
        owned=ownership(m,stage);values,parts=loss(m,batch,stage,d.norms,noise,s,k,q);values.mean().backward();assert torch.isfinite(values).all();assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.parameters() if p.requires_grad)
        assert all(p.grad is None for p in m.parameters() if not p.requires_grad);m.zero_grad(set_to_none=True)
    state=enforce(c,torch.randn(2,c.nodes+1,2),batch.parameter,batch.known,batch.families);torch.testing.assert_close(m.evolve(state,0.,batch.families),state,atol=0,rtol=0)
    assert torch.equal(state[batch.known,-1,0],batch.parameter[batch.known]);assert torch.all(state[batch.families.eq(0),:c.nodes,1]==0)
    if not c.kind['trajectory']:
        assert m.evolution is None
        with pytest.raises(ValueError,match='initial'):m.evolve(state,.2,batch.families)


def test_tuple_permutation_padding_empty_time_and_pairing(tmp_path):
    c=config();m=MeshModel(c).eval();data=Data(c,tmp_path);b=data.prepare(1);obs=b.observations
    def context(o):return m.condition(o,b.parameter,b.known,b.families)[0]
    y=context(obs);perm=torch.arange(obs[0].shape[1]-1,-1,-1);torch.testing.assert_close(y,context(tuple(x[:,perm] for x in obs)),atol=2e-6,rtol=2e-5)
    padded=[]
    for x in obs:
        extra=torch.full_like(x[:,:2],False if x.dtype==torch.bool else 99 if x.dtype==torch.long else float('nan'));padded.append(torch.cat([x,extra],1))
    torch.testing.assert_close(y,context(tuple(padded)),atol=2e-6,rtol=2e-5)
    assert torch.isfinite(context(tuple(x[:,:0] for x in obs))).all()
    moved=[x.clone() for x in obs];moved[0][...,-1]=1-moved[0][...,-1];assert (y-context(tuple(moved))).abs().max()>1e-5
    swapped=[x.clone() for x in obs];swapped[1][...,0]=swapped[1][:,perm,0];assert (y-context(tuple(swapped))).abs().max()>1e-5


def test_physical_query_chunk_permutation_and_derivatives(tmp_path):
    c=config();m=MeshModel(c).eval();d=Data(c,tmp_path);b=d.prepare(0);state=b.target_state;xy=torch.rand(2,9,2,requires_grad=True);a=m.decoder(state,xy,b.families,chunk=2);z=m.decoder(state,xy,b.families,chunk=9);torch.testing.assert_close(a,z,atol=2e-6,rtol=2e-5);perm=torch.randperm(9);torch.testing.assert_close(a[:,perm],m.decoder(state,xy[:,perm],b.families),atol=2e-6,rtol=2e-5)
    shifted=xy+torch.tensor(c.domain['lengths']);torch.testing.assert_close(a,m.decoder(state,shifted,b.families),atol=4e-6,rtol=2e-5)
    grad=torch.autograd.grad(a.sum(),xy,create_graph=True)[0];second=torch.autograd.grad(grad.sum(),xy)[0];assert torch.isfinite(second).all() and grad.abs().sum()>0


def test_fgn_fair_scores_and_coherent_fixed_noise(tmp_path):
    assert fair_crps(torch.tensor(0.),torch.tensor(2.),torch.tensor(1.))==0
    assert fair_crps(torch.tensor(0.),torch.tensor(0.),torch.tensor(1.))==1
    a=torch.tensor([[0.,0.]]);b=torch.tensor([[2.,2.]]);truth=torch.ones_like(a);torch.testing.assert_close(fair_energy(a,b,truth),torch.zeros(1),atol=1e-6,rtol=0)
    c=config('mesh_m3_fgn_v1');m=MeshModel(c).eval();data=Data(c,tmp_path);batch=data.prepare(0);context=m.condition(batch.observations,batch.parameter,batch.known,batch.families);g=torch.Generator().manual_seed(3);noise=fgn_noise(c,2,g,'cpu');other=fgn_noise(c,2,g,'cpu');assert not torch.equal(noise[0],other[0]);one,_=generate(m,context,batch.parameter,batch.known,batch.families,noise=noise);two,_=generate(m,context,batch.parameter,batch.known,batch.families,noise=noise);torch.testing.assert_close(one,two,atol=0,rtol=0);assert (one-generate(m,context,batch.parameter,batch.known,batch.families,noise=other)[0]).abs().max()>1e-5


def test_fm_endpoint_and_distinct_diffusion_rotation_replay(tmp_path):
    c=config('mesh_m4_flow_v1');data=Data(c,tmp_path);b=data.prepare(0);target=b.target_state;noise=torch.randn_like(target);s=torch.tensor([.2,.8]);current,v=flow_path(c,target,noise,s,b.parameter,b.known,b.families);torch.testing.assert_close(enforce(c,flow_endpoint(current,v,s),b.parameter,b.known,b.families),target,atol=4e-7,rtol=2e-6)
    dc=replace(c,architecture='mesh_m5_diffusion_v1').validate();k=torch.tensor([1,128]);x,v=diffusion_path(dc,target,noise,k,b.parameter,b.known,b.families);alpha,sigma=diffusion_schedule(128);clean,eps=diffusion_endpoint(x,v,alpha[k,None,None],sigma[k,None,None]);mask=active_mask(c,b.known,b.families);torch.testing.assert_close(clean[mask],target[mask],atol=5e-7,rtol=2e-6);torch.testing.assert_close(eps[mask],noise[mask],atol=5e-7,rtol=2e-6);assert alpha[0]==1 and sigma[0]==0 and alpha[-1]>0 and torch.all(alpha[1:]<alpha[:-1])
    m=MeshModel(dc).eval();context=m.condition(b.observations,b.parameter,b.known,b.families);a,meta=generate(m,context,b.parameter,b.known,b.families,noise=noise,steps=4);z,_=generate(m,context,b.parameter,b.known,b.families,noise=noise,steps=4);torch.testing.assert_close(a,z,atol=0,rtol=0);assert meta['method'].startswith('VP-v') and meta['indices']==[128,96,64,32,0]
