import numpy as np
import pytest
import torch
from pdeflow.core import DomainSpec,GridSpec
from pdeflow.darcy import problem,default_forcing,solve,numpy_operator
from pdeflow.darcy.inverse import DarcyInverseObjective,inverse_baseline
from pdeflow.observations import sensor_observations,ObservationSet,PointOperator

@pytest.mark.parametrize('dimension',[2,3])
def test_implicit_adjoint_gradient_and_physical_forward(dimension):
 g=GridSpec((8,)*dimension,DomainSpec((1.,)*dimension),(0.,),'periodic_cell_centers','static');f=default_forcing(g);p=problem(g,f)
 coordinates=np.meshgrid(*[x.numpy() for x in g.coordinates()],indexing='ij');a=np.exp(.3*np.tanh(sum(np.cos(2*np.pi*x) for x in coordinates)));u,_=solve(a,f,g)
 reference=torch.from_numpy(np.stack((a,u)))[None,:,None];obs=sensor_observations(reference,g,8,17,.005,.02);objective=DarcyInverseObjective(p,obs,.01)
 theta=np.random.default_rng(5).normal(0,.1,objective.size);value,grad,fields,diag=objective.evaluate(theta);direction=np.random.default_rng(6).normal(size=objective.size);direction/=np.linalg.norm(direction);eps=1e-5
 fd=(objective.evaluate(theta+eps*direction)[0]-objective.evaluate(theta-eps*direction)[0])/(2*eps)
 assert grad@direction==pytest.approx(fd,rel=2e-5,abs=1e-6)
 assert np.sqrt(np.mean((numpy_operator(fields[0,0],fields[1,0],g)-f)**2))<1e-7 and abs(fields[1].mean())<1e-14
 result=inverse_baseline(p,obs,max_iterations=4)
 assert result.diagnostics['objective']<=result.diagnostics['objective_trace'][0]['objective'] and result.samples[:,:,0].min()>0
 assert result.diagnostics['forward_solves']>0 and result.diagnostics['adjoint_solves']>0


def test_zero_forcing_is_labelled_uninformative():
 g=GridSpec((8,8),DomainSpec((1.,1.)),(0.,),'periodic_cell_centers','static');p=problem(g,np.zeros(g.shape))
 op=PointOperator(g,torch.tensor([[1,0,1,2]],dtype=torch.long),2);obs=ObservationSet(op,torch.zeros(1,1),torch.ones(1,1,dtype=torch.bool),.01)
 r=inverse_baseline(p,obs);assert r.diagnostics['zero_forcing_degenerate'] and r.diagnostics['pressure_only']
 assert torch.count_nonzero(r.samples[:,:,1])==0
