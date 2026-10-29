"""Periodic A+B reversible C: positive Strang reference and physical residual."""
from dataclasses import dataclass
import math
import numpy as np
import torch
from pdeflow.core import FieldSpec,ProblemSpec,TransformBundle

PARAMETERS=('D_A','D_B','D_C','k_f','k_r')

def operator_spec():return {'op':'reversible_mass_action_diffusion','species':['a','b','c'],'stoichiometry':[-1,-1,1],'rate':'k_f*a*b-k_r*c','boundary':'periodic'}

def problem(grid,parameters):return ProblemSpec('chemistry',grid,tuple(FieldSpec(n) for n in ('a','b','c')),dict(parameters),operator_spec())


def reaction(state,kf,kr,dt):
    """Exact scalar Riccati update in the local invariant subspace; no clipping."""
    if min(kf,kr,dt)<0 or not np.isfinite(state).all() or np.any(state<0):raise ValueError('Reaction needs nonnegative concentrations/rates/time')
    a,b,c=state;A=a+c;B=b+c
    if kf==0:new_c=c*np.exp(-kr*dt)
    else:
        s=kf*(A+B)+kr;discriminant=np.sqrt((kf*(A-B))**2+2*kf*kr*(A+B)+kr*kr)
        root=np.divide(2*kf*A*B,s+discriminant,out=np.zeros_like(c),where=(s+discriminant)>0)
        delta=c-root;decay=np.exp(-discriminant*dt)
        ratio=np.divide(kf*delta,discriminant,out=np.zeros_like(delta),where=discriminant>0)
        denominator=1-ratio*(-np.expm1(-discriminant*dt))
        regular=root+delta*decay/denominator
        repeated=A+(c-A)/(1-kf*(c-A)*dt)
        new_c=np.where(discriminant>0,regular,repeated)
    result=np.stack((A-new_c,B-new_c,new_c))
    if np.any(result<0) or not np.isfinite(result).all():raise FloatingPointError('Exact reaction update lost positivity through arithmetic; no hidden clipping')
    return result


def diffusion_kernel(n,lam,tolerance=1e-14):
    """Positive random-walk uniformization of a periodic 1D FD semigroup."""
    if not 0<=lam<=32:raise ValueError('Diffusion substep rate must lie in [0,32]')
    walk=np.zeros(n);walk[0]=1.;weight=math.exp(-lam);kernel=weight*walk;mass=weight
    for order in range(1,257):
        if 1-mass<=tolerance:break
        walk=.5*(np.roll(walk,1)+np.roll(walk,-1));weight*=lam/order;kernel+=weight*walk;mass+=weight
    else:raise RuntimeError('Poisson diffusion series did not converge')
    # Tail renormalization <= tolerance preserves mass; record its approximation.
    return kernel/kernel.sum(),max(0.,1-mass)


def solve(initial,parameters,grid,max_step=.0025,max_steps=4000):
    if initial.shape!=(3,*grid.shape) or np.any(initial<0) or not np.isfinite(initial).all():raise ValueError('Chemistry requires finite nonnegative [3,*space] initial concentrations')
    if set(parameters)!=set(PARAMETERS) or min(parameters[n] for n in PARAMETERS[:3])<=0 or min(parameters[n] for n in PARAMETERS[3:])<0 or max_step<=0:raise ValueError('Positive diffusivities, nonnegative kinetics and positive reference step required')
    if grid.temporal_kind!='transient' or grid.domain.boundary!='periodic' or grid.times[0]!=0:raise ValueError('Chemistry reference requires periodic physical times starting at zero')
    u=np.array(initial,dtype=np.float64,copy=True);frames=[u.copy()];cache={};max_tail=0.;steps=0
    invariant0=np.array([(u[0]+u[2]).sum(),(u[1]+u[2]).sum()])*grid.cell_volume
    def diffuse(x,dt):
        nonlocal max_tail
        output=[]
        for species,D in enumerate(parameters[n] for n in PARAMETERS[:3]):
            field=x[species].copy()
            for axis,(n,h) in enumerate(zip(grid.shape,grid.spacing)):
                lam=2*D*dt/(h*h);sub=max(1,math.ceil(lam/16));key=(n,lam/sub)
                if key not in cache:cache[key]=diffusion_kernel(*key)
                kernel,tail=cache[key];max_tail=max(max_tail,tail)
                for _ in range(sub):
                    value=np.zeros_like(field)
                    for shift,weight in enumerate(kernel):
                        if weight:value+=weight*np.roll(field,shift,axis)
                    field=value
            output.append(field)
        return np.stack(output)
    for first,last in zip(grid.times,grid.times[1:]):
        count=max(1,math.ceil((last-first)/max_step));dt=(last-first)/count
        if steps+count>max_steps:raise RuntimeError('Chemistry reference exceeded bounded physical step budget')
        for _ in range(count):
            u=diffuse(u,dt/2);u=reaction(u,parameters['k_f'],parameters['k_r'],dt);u=diffuse(u,dt/2);steps+=1
        frames.append(u.copy())
    values=np.stack(frames,axis=1)
    axes=tuple(range(1,grid.dimension+1))
    invariants=np.stack([(values[0]+values[2]).sum(axes),(values[1]+values[2]).sum(axes)])*grid.cell_volume
    diagnostics={'solver':'positive_uniformization_strang_exact_reaction_v1','converged':True,'physical_steps':steps,'max_physical_step':max_step,'max_diffusion_poisson_tail':max_tail,'diffusion_tail_renormalized':True,'clipped_values':0,'minimum_concentration':float(values.min()),'max_stoichiometric_integral_drift':float(np.max(np.abs(invariants-invariant0[:,None]))),'total_species_sum_change':float((values[:,-1].sum()-values[:,0].sum())*grid.cell_volume)}
    return values,diagnostics


@dataclass(frozen=True)
class ConcentrationTransform(TransformBundle):
    @classmethod
    def fit(cls,physical,ids):
        if torch.any(physical<0):raise ValueError('Concentrations must be nonnegative')
        base=TransformBundle.fit(physical.sqrt(),ids);return cls(base.mean,base.scale,base.fitted_ids)
    def encode(self,physical):
        if torch.any(physical<0):raise ValueError('Concentrations must be nonnegative')
        return super().encode(physical.sqrt())
    def decode(self,model):return super().decode(model).square()


@dataclass
class ChemistryPhysics:
    grid: object
    condition: object
    spatial: str='fd'
    def evaluate(self,physical):
        from pdeflow.numerics import time_derivative,fd_laplacian,spectral_laplacian
        from pdeflow.equations import ResidualBundle
        if tuple(physical.shape[1:])!=(3,len(self.grid.times),*self.grid.shape):raise ValueError('Chemistry physical channel/time layout mismatch')
        params=self.condition.known_parameters;shape=(-1,1,*([1]*self.grid.dimension));lap={'fd':fd_laplacian,'spectral':spectral_laplacian}.get(self.spatial)
        if lap is None:raise ValueError('Unsupported chemistry residual discretization')
        dt=time_derivative(physical,self.grid);diffusion=lap(physical,self.grid)
        rate=params['k_f'].reshape(shape)*physical[:,0]*physical[:,1]-params['k_r'].reshape(shape)*physical[:,2]
        signs=(1,1,-1)
        residual=torch.stack([dt[:,i]-params[name].reshape(shape)*diffusion[:,i]+signs[i]*rate for i,name in enumerate(PARAMETERS[:3])],dim=1)[:,:,1:-1]
        return ResidualBundle({'pde':residual},{'pde':'nondimensional concentration per physical time'},self.grid.time_weights(device=physical.device,dtype=physical.dtype)[1:-1])
