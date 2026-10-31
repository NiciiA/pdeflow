"""Independent spectral checks and shared physical-state architecture regressions."""
from dataclasses import replace
import numpy as np
import pytest
import torch
from pdeflow.numerics.dynamic import SpectralState
from pdeflow.coordinate.graph import canonical,parse_equation,GraphEncoder,PDEGraph,Node
from pdeflow.coordinate.model import Autoencoder,Velocity,decode_chunks
from pdeflow.coordinate.physics import problem,strong_residual,invariant_drift
from pdeflow.coordinate.joint import layout,active_mask,flow_loss
from pdeflow.colab import training_preset
from pdeflow.core import GridSpec,DomainSpec

@pytest.fixture
def config():
    torch.set_num_threads(2)
    return training_preset('smoke','cpu','fp32',{'field_loss':{'analysis_resolution':8,'region_queries':16,'time_groups':1},'encoder_points':16,'query_points':16})

def test_wave_zero_mode_and_heat_constant():
    q=np.array([[.2,.3,.237],[.7,.4,1.]])
    w=SpectralState(np.full((8,10),2.),np.full((8,10),3.),family='wave')
    np.testing.assert_allclose(w.evaluate(q,.4,(0,1)),2+3*q[:,-1],atol=1e-12)
    np.testing.assert_allclose(w.evaluate(q,.4,(0,1),'v'),3,atol=1e-12)
    h=SpectralState(np.full((8,10),2.));np.testing.assert_allclose(h.evaluate(q,.07,(0,1)),2,atol=1e-12)

@pytest.mark.parametrize('shape',[(9,11),(8,10)])
def test_standing_wave_arbitrary_physical_coordinates(shape):
    L=(2.,3.);o=(-.4,.1);x,y=np.meshgrid(np.arange(shape[0])/shape[0],np.arange(shape[1])/shape[1],indexing='ij');u=.2+np.cos(2*np.pi*x)*np.cos(4*np.pi*y);s=SpectralState(u,np.zeros_like(u),L,o,'wave');rng=np.random.default_rng(31);q=rng.random((30,3));q[:,:2]=q[:,:2]*L+o;c=.37;omega=c*np.hypot(2*np.pi/L[0],4*np.pi/L[1]);a=np.cos(2*np.pi*(q[:,0]-o[0])/L[0])*np.cos(4*np.pi*(q[:,1]-o[1])/L[1])
    np.testing.assert_allclose(s.evaluate(q,c,(0,1)),.2+a*np.cos(omega*q[:,-1]),atol=1e-12)
    np.testing.assert_allclose(s.evaluate(q,c,(0,1),'v'),-a*omega*np.sin(omega*q[:,-1]),atol=1e-12)
    assert s.invariants(.13,c)['energy']==pytest.approx(s.invariants(.83,c)['energy'],rel=1e-12)


def test_travelling_velocity_only_nyquist_and_chunks():
    n=8;x,y=np.meshgrid(np.arange(n)/n,np.arange(n)/n,indexing='ij');c=.4;omega=2*np.pi*c;u=np.cos(2*np.pi*x);v=omega*np.sin(2*np.pi*x);s=SpectralState(u,v,family='wave');q=np.array([[.13,.2,.237],[.83,.4,.91]])
    np.testing.assert_allclose(s.evaluate(q,c,(0,1)),np.cos(2*np.pi*q[:,0]-omega*q[:,2]),atol=1e-12)
    v0=SpectralState(np.zeros_like(u),u,family='wave');np.testing.assert_allclose(v0.evaluate(q,c,(0,1)),np.cos(2*np.pi*q[:,0])*np.sin(omega*q[:,2])/omega,atol=1e-12)
    ny=SpectralState(np.cos(n*np.pi*x)*np.cos(n*np.pi*y),np.zeros_like(u),family='wave');np.testing.assert_allclose(ny.frame(0,c,(0,1)),ny.initial,atol=1e-12)
    expected=np.cos(n*np.pi*q[:,0])*np.cos(n*np.pi*q[:,1])*np.cos(c*n*np.pi*np.sqrt(2)*q[:,2]);np.testing.assert_allclose(ny.evaluate(q,c,(0,1),chunk=1),expected,atol=1e-12)
    np.testing.assert_allclose(s.evaluate(q,c,(0,1),chunk=1),s.evaluate(q,c,(0,1),chunk=10),atol=1e-12)


def test_reference_first_order_derivatives():
    n=12;x,y=np.meshgrid(np.arange(n)/n,np.arange(n)/n,indexing='ij');s=SpectralState(np.cos(2*np.pi*x),np.sin(2*np.pi*y),family='wave');q=np.array([[.13,.27,.237],[.33,.41,.71]]);h=1e-5;c=.4
    def at(axis,sign,name):a=q.copy();a[:,axis]+=sign*h;return s.evaluate(a,c,(0,1),name)
    u=s.evaluate(q,c,(0,1));v=s.evaluate(q,c,(0,1),'v');lap=sum((at(i,1,'u')-2*u+at(i,-1,'u'))/h**2 for i in (0,1))
    np.testing.assert_allclose((at(2,1,'u')-at(2,-1,'u'))/(2*h),v,atol=1e-8)
    np.testing.assert_allclose((at(2,1,'v')-at(2,-1,'v'))/(2*h),c*c*lap,atol=2e-6)

@pytest.mark.parametrize('text,family',[('D*lap(u)-u_t=0','heat'),('0=c*c*lap(u)-u_tt','wave'),('2*u_tt=2*c**2*lap(u)','wave')])
def test_safe_equation_rearrangement(text,family):assert parse_equation(text)==canonical(family)

@pytest.mark.parametrize('text',['u_t=c*lap(u)','u_tt=c*lap(u)','u_t=D*lap(u)+u','u_t=__import__("os")','u_tt=D*lap(u)'])
def test_unsupported_equations(text):
    with pytest.raises(ValueError):parse_equation(text)


def test_graph_relabel_edges_and_real_conditioning(config):
    torch.manual_seed(19);enc=GraphEncoder(config.width);g=canonical('wave');order=list(reversed(range(len(g.nodes))));mapping={old:new for new,old in enumerate(order)};permuted=PDEGraph(tuple(g.nodes[i] for i in order),tuple((mapping[a],mapping[b],p) for a,b,p in g.edges),'wave');torch.testing.assert_close(enc.one(g),enc.one(permuted),atol=2e-6,rtol=2e-6)
    assert not torch.allclose(enc.one(g),enc.one(canonical('heat')))
    ae=Autoencoder(config).eval();z=torch.randn(1,config.tokens,config.width);f=torch.ones(1,dtype=torch.long);p=torch.tensor([.5]);k=torch.ones(1,dtype=torch.bool);graph=ae.graph_encoder(f).detach().requires_grad_();e=ae.evolve(z,0.,.237,p,k,f,graph);assert torch.autograd.grad(e.square().sum(),graph)[0].abs().sum()>0
    vel=Velocity(config).eval();obs=(torch.zeros(1,1,3),torch.zeros(1,1,3),torch.zeros(1,1,dtype=torch.bool),torch.zeros(1,1,dtype=torch.long));out=vel(z,torch.tensor([.4]),p,obs,k,f,graph);assert torch.autograd.grad(out.square().sum(),graph)[0].abs().sum()>0


def test_complete_state_identity_coefficient_authority_and_derivatives(config):
    torch.manual_seed(4);ae=Autoencoder(config).eval();f=torch.tensor([1]);p=torch.tensor([.4]);k=torch.tensor([True]);z=torch.randn(1,config.tokens,config.width);g=layout(config)[-1];torch.testing.assert_close(ae.evolve(z,.237,.237,p,k,f),z,atol=0,rtol=0);e=ae.evolve(z,0.,.237,p,k,f);torch.testing.assert_close(e[:,g.start:],z[:,g.start:],atol=0,rtol=0)
    q=torch.tensor([[.13,.27,.237],[.64,.43,.7]],requires_grad=True);u=ae.decode(z,q,p,k,'u',f);dt=torch.autograd.grad(u.sum(),q,create_graph=True)[0][:,-1];h=1e-3;plus=q.detach().clone();minus=plus.clone();plus[:,-1]+=h;minus[:,-1]-=h;fd=(ae.decode(z,plus,p,k,'u',f)-ae.decode(z,minus,p,k,'u',f))/(2*h);torch.testing.assert_close(dt,fd[0],atol=2e-4,rtol=.01)
    residual=strong_residual(ae,z.requires_grad_(),q.detach(),p,k,f,[{'mean':.2,'scale':.3},{'mean':-.1,'scale':2.}]);sum(v.square().mean() for v in residual.values()).backward();assert torch.isfinite(z.grad).all() and z.grad.abs().sum()>0
    torch.testing.assert_close(ae.decode_parameter(z,p,k,False,f),p,atol=0,rtol=0)
    torch.testing.assert_close(decode_chunks(ae,z,q.detach(),p,1,k,'v',f),decode_chunks(ae,z,q.detach(),p,100,k,'v',f),atol=1e-6,rtol=1e-5)


def test_factorized_encoder_and_observation_quantities(config):
    torch.manual_seed(7);ae=Autoencoder(config).eval();q=torch.rand(1,16,2);state=torch.randn(1,16,2);f=torch.tensor([1]);a=ae.encode(q,state,torch.tensor([.2]),f);b=ae.encode(q,state,torch.tensor([.8]),f);g=layout(config)[-1];torch.testing.assert_close(a[:,:g.start],b[:,:g.start]);c=ae.encode(q,state+10,torch.tensor([.2]),f);torch.testing.assert_close(a[:,g.start:],c[:,g.start:]);assert not torch.allclose(a[:,g.start:],b[:,g.start:])
    obs=Velocity(config).observation_encoder.eval();oq=torch.rand(1,5,3);features=torch.rand(1,5,3);valid=torch.tensor([[True,True,True,False,False]]);ids=torch.tensor([[0,1,0,0,0]]);order=torch.tensor([2,0,4,1,3]);v=obs((oq,features,valid,ids));torch.testing.assert_close(v,obs((oq[:,order],features[:,order],valid[:,order],ids[:,order])),atol=1e-6,rtol=1e-5);assert not torch.allclose(v,obs((oq,features,valid,torch.zeros_like(ids))))
    oq[:,3:]=float('nan');features[:,3:]=float('nan');assert torch.isfinite(obs((oq,features,valid,ids))).all();assert torch.isfinite(obs((oq,features,torch.zeros_like(valid),ids))).all()


def test_family_masks_reductions_and_compatibility(config):
    f=torch.tensor([0,1]);known=torch.tensor([True,False]);mask=active_mask(config,known,f);u,v,c=layout(config);assert not mask[0,v.start:v.stop].any() and not mask[0,c.start:].any() and mask[1].all()
    x=torch.ones(2,config.tokens,config.width)*mask[...,None];assert flow_loss(config,x,torch.zeros_like(x),known,f)==pytest.approx(1.)
    grid=GridSpec((9,13),DomainSpec((1.,1.)),(0.,.237,1.));config.compatible(problem(grid,'wave',None));config.compatible(problem(grid,'heat',.07))
    with pytest.raises(ValueError):config.compatible(replace(problem(grid,'wave',None),unknown_parameters=('diffusivity',)))
    from pdeflow.coordinate.contracts import validate_rows
    with pytest.raises(ValueError):validate_rows(config,[dict(field='v',x=.1,y=.2,time=.237,value=1.)],problem(grid,'heat',.07))


def test_wave_invariant_mean_is_linear():
    t=np.array([0.,.237,1.]);u=np.broadcast_to((2+3*t)[:,None,None],(3,8,8));v=np.full_like(u,3);r=invariant_drift(u,v,t,(1,1),.4);np.testing.assert_allclose(r['linear_mean_u_error'],0,atol=1e-12);np.testing.assert_allclose(r['energy_relative_drift'],0,atol=1e-12)


def test_model_nonunit_periodicity_and_full_wave_composition(config):
    c=replace(config,lengths=(2.,3.),origins=(-.4,.1));torch.manual_seed(11);ae=Autoencoder(c).eval();f=torch.tensor([1]);k=torch.tensor([True]);p=torch.tensor([.4]);z=torch.randn(1,c.tokens,c.width)
    q=torch.tensor([[-.4,.7,.237],[1.6,.7,.237],[.2,.1,.237],[.2,3.1,.237]])
    for name in ('u','v'):
        values=ae.decode(z,q,p,k,name,f);torch.testing.assert_close(values[:,0],values[:,1],atol=2e-6,rtol=2e-5);torch.testing.assert_close(values[:,2],values[:,3],atol=2e-6,rtol=2e-5)
    via=ae.evolve(ae.evolve(z,0.,.2,p,k,f),.2,.7,p,k,f);direct=ae.evolve(z,0.,.7,p,k,f);g=layout(c)[-1];torch.testing.assert_close(via[:,g.start:],z[:,g.start:]);assert torch.isfinite((via-direct).square().mean())
    # Complete-state composition is measured for BOTH physical quantities, not assumed exact.
    errors=[float((ae.decode_state(via,q[:,:2],j,f)-ae.decode_state(direct,q[:,:2],j,f)).square().mean().detach()) for j in (0,1)];assert all(np.isfinite(errors))


def test_invalid_derivative_graph_and_zero_displacement_velocity_gradients(config):
    g=canonical('heat');nodes=list(g.nodes);nodes[6]=Node('dt1','field',(1,0,-2))
    with pytest.raises(ValueError,match='units'):PDEGraph(tuple(nodes),g.edges,g.family).validate()
    ae=Autoencoder(config);q=torch.rand(1,16,2);state=torch.zeros(1,16,2);state[...,1]=torch.sin(2*torch.pi*q[...,0]);f=torch.tensor([1]);z=ae.encode(q,state,torch.tensor([.5]),f);prediction=ae.decode_state(z,q[0],1,f);loss=(prediction-state[...,1]).square().mean();loss.backward()
    assert ae.encoder.fields[1][0].weight.grad.abs().sum()>0 and ae.decoder.output[1].weight.grad.abs().sum()>0
