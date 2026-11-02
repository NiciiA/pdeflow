"""Numerical invariants of versioned sensor conditioning and independent A2 term."""
from dataclasses import replace
import math
import pytest
import torch
from pdeflow.colab import training_preset
from pdeflow.coordinate.experiments import conditional_experiment
from pdeflow.coordinate.model import Autoencoder,Velocity
from pdeflow.coordinate.spatial_conditioning import relative_features
from pdeflow.coordinate.evolution_objective import decoded_kinematic,kinematic_loss
from pdeflow.coordinate.quantity_weights import combine


def tiny(name='B3'):
    torch.set_num_threads(2)
    return conditional_experiment(training_preset('smoke','cpu','fp32',{'encoder_points':16,'query_points':16}),name,{'warmup_updates':0,'sensor_budget':4})


def inputs(c):
    torch.manual_seed(123);ae=Autoencoder(c).eval().requires_grad_(False);m=Velocity(c).eval()
    f=torch.tensor([0,1]);p=torch.tensor([.4,.6]);known=torch.tensor([True,False]);z=torch.randn(2,c.tokens,c.width);s=torch.tensor([.7,.8]);q=torch.rand(2,5,3);feat=torch.randn(2,5,3);feat[...,1]=.1;feat[...,2]=1;valid=torch.ones(2,5,dtype=torch.bool);qty=torch.tensor([[0]*5,[0,1,0,1,0]])
    return ae,m,z,s,p,(q,feat,valid,qty),known,f


def forward(a,obs=None):
    ae,m,z,s,p,o,k,f=a
    return m(z,s,p,o if obs is None else obs,k,f,ae.graph_encoder(f))


def test_sensor_order_padding_empty_time_and_pairing():
    c=tiny();a=inputs(c);y=forward(a);o=a[5];perm=torch.tensor([4,1,3,0,2]);yp=forward(a,tuple(x[:,perm] for x in o));torch.testing.assert_close(y,yp,atol=2e-6,rtol=2e-5)
    pad=[]
    for x in o:
        extra=torch.full_like(x[:,:2],False if x.dtype==torch.bool else 99 if x.dtype==torch.long else float('nan'));pad.append(torch.cat([x,extra],1))
    torch.testing.assert_close(y,forward(a,tuple(pad)),atol=2e-6,rtol=2e-5)
    empty=[x.clone() for x in o];empty[2][:]=False;empty[0][:]=float('nan');empty[1][:]=float('nan');empty[3][:]=99
    assert torch.isfinite(forward(a,tuple(empty))).all()
    swapped=[x.clone() for x in o];swapped[1][...,0]=o[1][:,perm,0]
    assert (y-forward(a,tuple(swapped))).abs().max()>1e-5
    times=[x.clone() for x in o];times[0][...,-1]=1-times[0][...,-1]
    assert (y-forward(a,tuple(times))).abs().max()>1e-5
    y.square().mean().backward()
    assert a[1].sensor_attention.out.weight.grad.abs().sum()>0
    assert a[1].observation_encoder.input[0].weight.grad.abs().sum()>0
    assert all(p.grad is None for p in a[0].parameters())


def test_streamed_attention_value_and_gradient_matches_dense():
    c=tiny();a=inputs(c);reader=a[1].sensor_attention
    reader.relative[-1].weight.data.normal_(0,.1)
    conditioning=a[1].condition(a[4],a[5],a[6],a[7],a[0].graph_encoder(a[7]))
    state=torch.randn(2,c.tokens,c.width,requires_grad=True);sensor=conditioning[2]
    reader.cfg.update(query_chunk=2,sensor_chunk=2);small=reader(state,sensor);g=torch.autograd.grad(small.square().sum(),state,retain_graph=True)[0]
    reader.cfg.update(query_chunk=1024,sensor_chunk=1024);dense=reader(state,sensor);dg=torch.autograd.grad(dense.square().sum(),state)[0]
    torch.testing.assert_close(small,dense,atol=2e-6,rtol=2e-5);torch.testing.assert_close(g,dg,atol=2e-6,rtol=2e-5)


def test_periodic_bias_global_roles_and_no_fictitious_anchors():
    c=tiny();a=inputs(c);r=a[1].sensor_attention;q=a[5][0];qty=a[5][3]
    feat=relative_features(c,r.anchors,r.local,r.roles,q,qty);shift=q.clone();shift[...,:2]+=torch.tensor(c.lengths)
    torch.testing.assert_close(feat,relative_features(c,r.anchors,r.local,r.roles,shift,qty),atol=2e-6,rtol=2e-5)
    assert torch.equal(feat[:,~r.local,:,:4],torch.zeros_like(feat[:,~r.local,:,:4]))
    assert set(r.roles.tolist())=={0,1,2}


def test_wave_kinematic_independent_analytic_and_real_frozen_gradient(monkeypatch):
    import pdeflow.coordinate.evolution_objective as objective
    c=tiny();ae=Autoencoder(c).eval().requires_grad_(False);f=torch.ones(1,dtype=torch.long);p=torch.ones(1)*.5;k=torch.ones(1,dtype=torch.bool);z=torch.randn(1,c.tokens,c.width,requires_grad=True);q=torch.rand(1,4,3)
    norms={'heat':[{'mean':0.,'scale':1.}], 'wave':[{'mean':0.,'scale':2.},{'mean':0.,'scale':3.}]}
    original=objective.decode_points
    def analytic(ae,z,coords,ids,p,k,f):
        t=coords[...,-1];u=t.square()+coords[...,0];v=2*t
        return torch.where(ids.eq(0),u/2,v/3)
    monkeypatch.setattr(objective,'decode_points',analytic)
    torch.testing.assert_close(decoded_kinematic(ae,z,q,p,k,f,norms),torch.zeros(1,4),atol=1e-7,rtol=0)
    monkeypatch.setattr(objective,'decode_points',original)
    ae.evolution.requires_grad_(True);cfg=replace(c,evolution={**c.evolution,'kinematic':{'weight':.02,'query_points':2}}).validate();loss,info=kinematic_loss(cfg,ae,z,p,k,f,norms,7);loss.mean().backward()
    assert torch.isfinite(loss).all() and info['kinematic_examples']==1
    assert any(x.grad is not None and x.grad.abs().sum()>0 for x in ae.evolution.parameters())
    assert all(x.grad is None for x in ae.decoder.parameters())
    assert z.grad.abs().sum()>0
    with pytest.raises(ValueError,match='counted twice'):replace(cfg,evolution={**cfg.evolution,'physics':{'weight':.1}}).validate()


def test_quantity_weight_normalized_active_reduction_and_unchanged_default():
    c=tiny();u=torch.tensor(2.,requires_grad=True);v=torch.tensor(6.,requires_grad=True)
    assert combine(c,[u,v])==4 and combine(c,[u])==2
    weighted=replace(c,representation_weights={'u':1,'v':3}).validate();value=combine(weighted,[u,v]);value.backward()
    assert value==5 and u.grad==.25 and v.grad==.75


def test_new_route_microbatch_fm_endpoint_gradients_and_cpu_amp():
    from pdeflow.coordinate.joint import flow_loss
    from pdeflow.coordinate.observation_objective import endpoint_loss
    c=tiny();a=inputs(c);ae,m,z,s,p,obs,known,f=a
    norms={'heat':[{'mean':0.,'scale':1.}], 'wave':[{'mean':0.,'scale':2.},{'mean':0.,'scale':3.}]};target=torch.randn_like(z);mean=torch.zeros_like(z[:1]);scale=torch.ones_like(mean)
    def loss(start,end):
        o=tuple(x[start:end] for x in obs);fam=f[start:end];k=known[start:end];state=z[start:end];pred=m(state,s[start:end],p[start:end],o,k,fam,ae.graph_encoder(fam))
        extra,_=endpoint_loss(ae,c,state,pred,s[start:end],mean,scale,p[start:end],k,fam,o,norms,0,44+start)
        return flow_loss(c,pred,target[start:end],k,fam)+extra.mean()
    parameters=tuple(m.parameters());full=loss(0,2);gradient=torch.autograd.grad(full,parameters,allow_unused=True);parts=(loss(0,1)+loss(1,2))/2;other=torch.autograd.grad(parts,parameters,allow_unused=True)
    torch.testing.assert_close(full,parts,atol=3e-6,rtol=3e-5)
    for x,y in zip(gradient,other):
        if x is None or y is None:assert x is None and y is None
        else:torch.testing.assert_close(x,y,atol=8e-6,rtol=5e-5)
    with torch.autocast('cpu',dtype=torch.bfloat16):value=loss(0,2)
    value.backward();assert torch.isfinite(value) and all(torch.isfinite(p.grad).all() for p in parameters if p.grad is not None)
