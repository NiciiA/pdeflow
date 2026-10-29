from dataclasses import replace
import numpy as np
import pytest
import torch
from pdeflow.coordinate.contracts import CoordinateConfig,grid_queries,validate_queries,validate_rows,padded_observations
from pdeflow.coordinate.model import Autoencoder,Velocity,decode_chunks
from pdeflow.core import GridSpec,DomainSpec
from pdeflow.equations import heat_problem
from pdeflow.coordinate.evaluation import error_summary,interpolate_grid

def config(d=2):return CoordinateConfig(dimension=d,reference_shape=(4,)*d,lengths=(1.,)*d,origins=(0.,)*d,width=8,heads=2,tokens=3,layers=1,ae_steps=2,flow_steps=2,train_count=3,validation_count=1,test_count=1,encoder_points=24,query_points=24,checkpoint_every=1,batch_size=1,observation_counts=(0,8)).validate()

@pytest.mark.parametrize('d',[1,2,3])
def test_permutation_padding_query_order_chunk_and_physical_periodicity(d):
    torch.manual_seed(13);c=config(d);g=GridSpec(c.reference_shape,DomainSpec(c.lengths),c.times);q=grid_queries(g);ae=Autoencoder(c).eval();v=Velocity(c).eval();p=torch.tensor([.4]);y=torch.randn(1,len(q));latent=ae.encode(q[None],y,p)
    permutation=torch.randperm(len(q));torch.testing.assert_close(latent,ae.encode(q[permutation][None],y[:,permutation],p),rtol=1e-5,atol=1e-6)
    whole=decode_chunks(ae,latent,q,p,999);chunked=decode_chunks(ae,latent,q,p,7);torch.testing.assert_close(whole,chunked,rtol=1e-5,atol=1e-6)
    torch.testing.assert_close(ae.decode(latent,q[permutation],p),whole[:,permutation],rtol=1e-5,atol=1e-6)
    norm={'mean':2.,'scale':3.};item=(q[:5],torch.arange(5.),torch.ones(5)*.1);obs=padded_observations([item],norm,'cpu');before=v(latent,torch.tensor([.25]),p,obs)
    reorder=torch.tensor([4,1,2,0,3]);after=v(latent,torch.tensor([.25]),p,tuple(t[:,reorder] for t in obs));torch.testing.assert_close(before,after,rtol=1e-5,atol=1e-6)
    extended=(torch.cat([obs[0],torch.full((1,3,d+1),float('nan'))],1),torch.cat([obs[1],torch.full((1,3,3),float('nan'))],1),torch.cat([obs[2],torch.zeros(1,3,dtype=torch.bool)],1));torch.testing.assert_close(before,v(latent,torch.tensor([.25]),p,extended),rtol=1e-5,atol=1e-6)
    no=padded_observations([(q[:0],torch.zeros(0),torch.zeros(0))],norm,'cpu');assert torch.isfinite(v(latent,torch.tensor([.25]),p,no)).all()
    assert not torch.allclose(before,v(latent,torch.tensor([.75]),p,obs)) # flow time is a real independent condition
    qperiod=q[:2].clone();qperiod[:,0]+=1.;torch.testing.assert_close(ae.decode(latent,q[:2],p),ae.decode(latent,qperiod,p),rtol=1e-5,atol=1e-6)
    # Physical derivatives must survive second differentiation for diagnostics/guidance.
    physical=q[:3].clone().requires_grad_(True);u=ae.decode(latent.detach(),physical,p);du=torch.autograd.grad(u.sum(),physical,create_graph=True)[0];assert torch.isfinite(torch.autograd.grad(du[:,0].sum(),physical)[0]).all()


def test_contract_changes_queries_not_physics_and_noise_normalization():
    c=config();g=GridSpec((6,6),DomainSpec((1.,1.)),(0.,.025,.1));p=heat_problem(g,.07);c.compatible(p)
    c.compatible(replace(p,grid=replace(g,times=(.02,.08))))
    for bad in [replace(p,known_parameters={'diffusivity':.8}),replace(p,grid=replace(g,domain=DomainSpec((2.,1.)))),replace(p,grid=replace(g,times=(0.,.2))),replace(p,known_fields={'f':0})]:
        with pytest.raises(ValueError):c.compatible(bad)
    with pytest.raises(ValueError):validate_queries(c,[[1.1,0,.05]])
    with pytest.raises(ValueError):replace(c,observation_counts=(0,1000)).validate()
    rows=[{'field':'u','x':.123,'y':.234,'time':.033,'value':0.,'sigma':.2}];item=validate_rows(c,rows,p);obs=padded_observations([item],{'mean':2.,'scale':4.},'cpu');assert obs[1][0,0,0]==-.5;assert obs[1][0,0,1]==.05
    assert obs[1][0,0,0]*4+2==0 # measured zero is retained
    with pytest.raises(ValueError):validate_rows(c,rows+rows)
    with pytest.raises(ValueError):validate_rows(c,[{**rows[0],'value':None}])
    with pytest.raises(ValueError):validate_rows(c,[{**rows[0],'value':1e300}])
    with pytest.raises(ValueError):validate_rows(c,[{**rows[0],'x':1.}])


def test_interpolation_order_no_extrapolation_and_near_zero_errors():
    g=GridSpec((4,),DomainSpec((1.,)),(0.,.05,.1));values=np.array([np.arange(4)+t for t in g.times]);q=grid_queries(g).numpy();result=interpolate_grid(values,g,q)
    np.testing.assert_allclose(result,values.reshape(-1),atol=1e-7)
    q=q[::-1].copy();np.testing.assert_allclose(interpolate_grid(values,g,q),result[::-1],atol=1e-7)
    with pytest.raises(ValueError):interpolate_grid(values,g,np.array([[.1,.2]]))
    assert error_summary(np.ones((2,5)),np.zeros(5))['relative_l2_per_sample'] is None

def test_heteroscedastic_noise_weights_baseline():
    from pdeflow.observations import ObservationSet,PointOperator
    from pdeflow.evaluation.baseline import inverse_baseline
    g=GridSpec((8,),DomainSpec((1.,)),(0.,.05,.1));indices=torch.tensor([[0,0,0],[0,0,4]],dtype=torch.long)
    op=PointOperator(g,indices);values=torch.tensor([[1.,-10.]],dtype=torch.float64);mask=torch.ones_like(values,dtype=torch.bool)
    hetero=ObservationSet(op,values,mask,10.,.01,measurement_noise_std=(.01,10.))
    uniform=ObservationSet(op,values,mask,10.,.01)
    weighted=inverse_baseline(g,.07,hetero,ridge=.1).samples[0]
    unweighted=inverse_baseline(g,.07,uniform,ridge=.1).samples[0]
    assert abs(float(op.apply(weighted)[0,0])-1)<abs(float(op.apply(unweighted)[0,0])-1)
