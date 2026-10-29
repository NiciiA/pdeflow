import numpy as np
import pytest
import torch
from pdeflow.core import DomainSpec,GridSpec
from pdeflow.darcy import solve,numpy_operator,DarcyPhysics,DarcyTransform,tensor_fluxes


def manufactured(n,dimension,variable=True):
    grid=GridSpec((n,)*dimension,DomainSpec((1.,)*dimension),(0.,),'periodic_cell_centers','static')
    x=np.meshgrid(*[c.numpy() for c in grid.coordinates()],indexing='ij');w=2*np.pi
    u=sum(np.sin(w*z) for z in x);a=1+.1*sum(np.cos(w*z) for z in x) if variable else np.ones_like(u)
    f=a*w*w*u+(.1*w*w*sum(np.sin(w*z)*np.cos(w*z) for z in x) if variable else 0)
    return grid,a,u,f


@pytest.mark.parametrize('dimension',[2,3])
@pytest.mark.parametrize('variable',[False,True])
def test_independent_manufactured_convergence_flux_and_gauge(dimension,variable):
    errors=[]
    for n in (8,16,32):
        g,a,exact,f=manufactured(n,dimension,variable);u,diag=solve(a,f,g,rtol=1e-11)
        errors.append(np.linalg.norm(u-exact)/np.linalg.norm(exact))
        assert diag['converged'] and diag['relative_residual']<2e-11
        assert abs(u.mean())<1e-14
        physical=torch.from_numpy(np.stack((a,u)))[None,:,None]
        residual=DarcyPhysics(g,torch.from_numpy(f)[None]).evaluate(physical).pointwise['pde']
        assert residual.abs().max()<1e-7
        assert abs(numpy_operator(a,u,g).sum())<1e-8
        for flux in tensor_fluxes(physical,g):assert abs((flux-torch.roll(flux,1,1)).sum())<1e-9
    assert errors[0]/errors[1]>3.7 and errors[1]/errors[2]>3.9


def test_solver_compatibility_failure_and_positivity():
    g,a,u,f=manufactured(8,3)
    with pytest.raises(ValueError,match='mean-zero'):solve(a,f+1,g)
    with pytest.raises(ValueError,match='positive'):solve(-a,f,g)
    with pytest.raises(RuntimeError,match='failed'):solve(a,f,g,max_iterations=1)


def test_two_channel_transform_and_physical_residual_gradient():
    g,a,u,f=manufactured(8,2)
    physical=torch.from_numpy(np.stack((a,u)))[None,:,None];transform=DarcyTransform.fit(physical,['train'])
    encoded=transform.encode(physical);decoded=transform.decode(encoded)
    assert torch.allclose(decoded,physical,atol=1e-14) and (decoded[:,0]>0).all()
    state=(encoded+.03*torch.randn_like(encoded)).requires_grad_(True)
    def energy(x):return DarcyPhysics(g,torch.from_numpy(f)[None]).evaluate(transform.decode(x)).mean_square()
    gradient,=torch.autograd.grad(energy(state),state);direction=torch.randn_like(state);direction/=direction.norm();eps=1e-6
    fd=(energy(state+eps*direction)-energy(state-eps*direction))/(2*eps)
    assert (gradient*direction).sum().item()==pytest.approx(fd.item(),rel=1e-6,abs=1e-6)
    assert gradient[:,0].abs().sum()>0 and gradient[:,1].abs().sum()>0


def test_static_observation_metadata_legacy_index_is_not_physical_time():
    from pdeflow.observations import sensor_observations,ObservationSet
    from copy import deepcopy
    g,a,u,f=manufactured(8,2)
    physical=torch.from_numpy(np.stack((a,u)))[None,:,None]
    obs=sensor_observations(physical,g,3,7,.005)
    saved=obs.to_dict();assert saved['coordinates']['physical_times'] is None
    legacy=deepcopy(saved);legacy['coordinates']['physical_times']=[0.]*len(obs.operator.indices)
    restored=ObservationSet.from_dict(legacy,g)
    assert torch.equal(restored.values,obs.values) and restored.operator.coordinates()['physical_times'] is None
    legacy['coordinates']['physical_times'][0]=1.
    with pytest.raises(ValueError,match='coordinate metadata'):ObservationSet.from_dict(legacy,g)


def test_darcy_does_not_relabel_fv_residual_as_spectral():
    from pdeflow.darcy import problem
    from pdeflow.core import Conditioning
    from pdeflow.specialists import get_specialist
    g,a,u,f=manufactured(8,2);p=problem(g,f)
    with pytest.raises(ValueError,match='finite-volume'):
        get_specialist('darcy').physics(p,Conditioning.from_problem(p,1),'spectral')
