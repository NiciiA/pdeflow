"""Periodic scalar-permeability Darcy: independent NumPy solve and tensor fluxes."""
from dataclasses import dataclass
import numpy as np
import torch
from pdeflow.core import FieldSpec, GridSpec, ProblemSpec, TransformBundle


def operator_spec():
    return {'op':'negative_divergence_coefficient_gradient','coefficient':'a','solution':'u','forcing':'f','face_mean':'harmonic','gauge':'zero_mean'}


def problem(grid, forcing):
    return ProblemSpec('darcy',grid,(FieldSpec('a'),FieldSpec('u')),{},operator_spec(),initial_condition='not_applicable',known_fields={'f':np.asarray(forcing).tolist()})


def default_forcing(grid):
    coordinates=np.meshgrid(*[x.numpy() for x in grid.coordinates()],indexing='ij')
    return sum(np.sin(2*np.pi*(x-o)/l) for x,o,l in zip(coordinates,grid.domain.origins,grid.domain.lengths))


def numpy_operator(a, u, grid):
    result=np.zeros_like(u,dtype=np.float64)
    for axis,h in enumerate(grid.spacing):
        right=np.roll(a,-1,axis);face=2*a*right/(a+right)
        flux=-face*(np.roll(u,-1,axis)-u)/h
        result+=(flux-np.roll(flux,1,axis))/h
    return result


def solve(a, forcing, grid, rtol=1e-10, max_iterations=4000):
    a=np.asarray(a,dtype=np.float64);f=np.asarray(forcing,dtype=np.float64)
    if a.shape!=grid.shape or f.shape!=grid.shape or not np.isfinite(a).all() or not np.isfinite(f).all() or np.any(a<=0):raise ValueError('Darcy requires finite positive cell permeability and finite matching forcing')
    if abs(f.mean())>1e-12*max(1,float(np.abs(f).max())):raise ValueError('Periodic Darcy requires compatible mean-zero forcing; no implicit source projection')
    if not 0<rtol<1 or max_iterations<1:raise ValueError('Invalid CG tolerance/iteration limit')
    u=np.zeros_like(f);r=f.copy();r-=r.mean();p=r.copy();rr=float(np.sum(r*r));initial=rr**.5
    if initial==0:return u,{'iterations':0,'relative_residual':0.,'converged':True}
    for iteration in range(1,max_iterations+1):
        ap=numpy_operator(a,p,grid);den=float(np.sum(p*ap))
        if den<=0 or not np.isfinite(den):raise RuntimeError('Darcy projected CG lost positive definiteness')
        alpha=rr/den;u+=alpha*p;u-=u.mean();r-=alpha*ap;r-=r.mean();new=float(np.sum(r*r))
        # Recheck the true linear residual before claiming solver convergence.
        if new**.5<=rtol*initial:
            actual=float(np.sqrt(np.sum((numpy_operator(a,u,grid)-f)**2))/initial)
            if actual<=max(2*rtol,1e-13):return u,{'iterations':iteration,'relative_residual':actual,'converged':True,'solver':'matrix_free_projected_CG','gauge_mean':float(u.mean())}
        p=r+(new/rr)*p;p-=p.mean();rr=new
    raise RuntimeError(f'Darcy CG failed to converge in {max_iterations} iterations (relative residual {rr**.5/initial:.3g})')


def tensor_fluxes(physical,grid):
    a,u=physical[:,0,0],physical[:,1,0];fluxes=[]
    for j,h in enumerate(grid.spacing):
        axis=j+1;right=torch.roll(a,-1,axis);face=2*a*right/(a+right)
        fluxes.append(-face*(torch.roll(u,-1,axis)-u)/h)
    return fluxes


@dataclass
class DarcyPhysics:
    grid: GridSpec
    forcing: torch.Tensor
    def evaluate(self,physical):
        from pdeflow.equations import ResidualBundle
        divergence=torch.zeros_like(physical[:,1,0])
        for j,(flux,h) in enumerate(zip(tensor_fluxes(physical,self.grid),self.grid.spacing)):
            divergence=divergence+(flux-torch.roll(flux,1,j+1))/h
        residual=(divergence-self.forcing.to(physical))[:,None,None]
        return ResidualBundle({'pde':residual},{'pde':'forcing units (nondimensional)'},physical.new_ones(1))


@dataclass(frozen=True)
class DarcyTransform(TransformBundle):
    def encode(self,physical):
        if torch.any(physical[:,0]<=0):raise ValueError('Permeability transform requires a>0')
        transformed=torch.stack((physical[:,0].log(),physical[:,1]),dim=1)
        return super().encode(transformed)
    def decode(self,model):
        transformed=super().decode(model)
        pressure=transformed[:,1];axes=tuple(range(1,pressure.ndim))
        pressure=pressure-pressure.mean(axes,keepdim=True)
        return torch.stack((transformed[:,0].exp(),pressure),dim=1)
    @classmethod
    def fit(cls,physical,ids):
        transformed=torch.stack((physical[:,0].log(),physical[:,1]),dim=1)
        base=TransformBundle.fit(transformed,ids)
        return cls(base.mean,base.scale,base.fitted_ids)
