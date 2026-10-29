"""Conventional restricted-coefficient inverse with verified implicit adjoint."""
from dataclasses import dataclass
import time
import numpy as np
import torch
from pdeflow.core import ReconstructionResult
from pdeflow.darcy import solve,numpy_operator


def basis(grid,degree=1):
    fields=[np.ones(grid.shape)];penalty=[1.]
    for j,(x,o,length) in enumerate(zip(grid.coordinates(),grid.domain.origins,grid.domain.lengths)):
        shape=[1]*grid.dimension;shape[j]=len(x)
        for k in range(1,degree+1):
            phase=2*np.pi*k*(x.numpy()-o)/length
            for v in (np.cos(phase),np.sin(phase)):
                fields.append(np.broadcast_to(v.reshape(shape),grid.shape));penalty.append((1+k*k)**2)
    return np.stack(fields),np.array(penalty)


@dataclass
class DarcyInverseObjective:
    problem: object
    observations: object
    ridge: float=.01
    degree: int=1
    rtol: float=1e-10
    def __post_init__(self):
        from pdeflow.specialists import get_specialist
        get_specialist('darcy').validate(self.problem,'independent_gaussian_linear_cfm','heun','uniform')
        get_specialist('darcy').validate_observations(self.problem,self.observations)
        if self.ridge<=0 or self.observations.values.shape[0]!=1 or not 1<=self.degree<min(self.problem.grid.shape)/2:raise ValueError('Positive regularization, one problem and valid Fourier degree required')
        self.features,self.penalty=basis(self.problem.grid,self.degree);self.forward_calls=0;self.adjoint_calls=0;self.forward_iterations=0;self.adjoint_iterations=0;self.failure_count=0
    @property
    def size(self):return len(self.features)+2
    def coefficient(self,theta):
        if len(theta)!=self.size:raise ValueError('Darcy inverse coefficient vector mismatch')
        offset=.2*np.tanh(theta[-2]);sigmoid=1/(1+np.exp(-np.clip(theta[-1],-500,500)));amplitude=.2+.4*sigmoid
        g=np.einsum('k,k...->...',theta[:-2],self.features);t=np.tanh(g)
        return np.exp(offset+amplitude*t),t,amplitude,sigmoid
    def evaluate(self,theta):
        grid=self.problem.grid;a,t,amp,sigmoid=self.coefficient(theta);f=np.asarray(self.problem.known_fields['f'])
        self.forward_calls+=1
        try:u,diagnostic=solve(a,f,grid,self.rtol)
        except RuntimeError:
            self.failure_count+=1;raise
        self.forward_iterations+=diagnostic['iterations']
        values=np.stack((a,u))[:,None];obs=self.observations;ix=obs.operator.indices.numpy();mask=obs.mask[0].numpy();indices=tuple(ix[mask].T)
        scales=obs.scales(obs.values).numpy()[mask];errors=values[indices]-obs.values[0].numpy()[mask]
        loss=.5*np.mean((errors/scales)**2)+.5*self.ridge*np.sum(self.penalty*theta[:-2]**2)+.005*self.ridge*np.sum(theta[-2:]**2)
        derivative=np.zeros_like(values);np.add.at(derivative,indices,errors/(len(errors)*scales**2))
        rhs=derivative[1,0];rhs=rhs-rhs.mean() # derivative restricted to the pressure gauge subspace
        self.adjoint_calls+=1
        try:adjoint,diag=solve(a,rhs,grid,self.rtol)
        except RuntimeError:
            self.failure_count+=1;raise
        self.adjoint_iterations+=diag['iterations']
        da=derivative[0,0].copy()
        for j,h in enumerate(grid.spacing):
            neighbor=np.roll(a,-1,j);face_sensitivity=-(np.roll(adjoint,-1,j)-adjoint)*(np.roll(u,-1,j)-u)/(h*h)
            da+=face_sensitivity*2*neighbor**2/(a+neighbor)**2+np.roll(face_sensitivity*2*a**2/(a+neighbor)**2,1,j)
        log_gradient=da*a
        gradient=np.empty_like(theta)
        gradient[:-2]=np.sum(self.features*(log_gradient*amp*(1-t*t))[None],axis=tuple(range(1,self.features.ndim)))+self.ridge*self.penalty*theta[:-2]
        gradient[-2]=np.sum(log_gradient)*.2*(1-np.tanh(theta[-2])**2)+.01*self.ridge*theta[-2]
        gradient[-1]=np.sum(log_gradient*t)*.4*sigmoid*(1-sigmoid)+.01*self.ridge*theta[-1]
        return float(loss),gradient,values,diagnostic


def inverse_baseline(problem,observations,degree=1,ridge=.01,max_iterations=24):
    start=time.perf_counter();objective=DarcyInverseObjective(problem,observations,ridge,degree);theta=np.zeros(objective.size)
    loss,gradient,values,diagnostic=objective.evaluate(theta);H=np.eye(len(theta));trace=[];reason='max_iterations';steps=0
    for step in range(max_iterations):
        norm=float(np.linalg.norm(gradient));trace.append({'iteration':step,'objective':loss,'gradient_norm':norm})
        if norm<=1e-5:reason='gradient_tolerance';break
        direction=-H@gradient
        if direction@gradient>=0:H=np.eye(len(theta));direction=-gradient
        alpha=1.;accepted=None
        for _ in range(16):
            trial=theta+alpha*direction
            trial_loss,trial_gradient,trial_values,trial_diag=objective.evaluate(trial)
            if trial_loss<=loss+1e-4*alpha*(gradient@direction):accepted=(trial,trial_loss,trial_gradient,trial_values,trial_diag);break
            alpha*=.5
        if accepted is None:reason='line_search_failed';break
        new,new_loss,new_grad,new_values,new_diag=accepted;s=new-theta;y=new_grad-gradient;ys=float(y@s)
        if ys>1e-12:
            rho=1/ys;V=np.eye(len(theta))-rho*np.outer(s,y);H=V@H@V.T+rho*np.outer(s,s)
        else:H=np.eye(len(theta))
        change=loss-new_loss;theta,loss,gradient,values,diagnostic=new,new_loss,new_grad,new_values,new_diag;steps=step+1
        if abs(change)<=1e-9*(1+abs(loss)):reason='objective_tolerance';break
    f=np.asarray(problem.known_fields['f']);degenerate=bool(np.sqrt(np.sum(f*f))<=1e-12)
    data=torch.from_numpy(values)[None,None]
    diagnostics={'seconds':time.perf_counter()-start,'model_evaluations':0,'gradient_evaluations':objective.adjoint_calls,'forward_solves':objective.forward_calls,'adjoint_solves':objective.adjoint_calls,'forward_cg_iterations':objective.forward_iterations,'adjoint_cg_iterations':objective.adjoint_iterations,'solver_failures':objective.failure_count,'solver_rtol':objective.rtol,'physical_relative_residual':diagnostic['relative_residual'],'optimizer_steps':steps,'objective':loss,'objective_trace':trace,'ridge':ridge,'degree':degree,'parameter_count':len(theta),'zero_forcing_degenerate':degenerate,'pressure_only':bool(torch.all(observations.operator.indices[:,0]==1))}
    provenance={'method':'darcy_restricted_fourier_implicit_adjoint_BFGS','parameterization':'a=exp(.2*tanh(offset_parameter)+(.2+.4*sigmoid(amplitude_parameter))*tanh(additive_real_Fourier_field))','initialization':'all unconstrained parameters zero; a=1; no truth inputs','likelihood':'mean Gaussian squared observation residual tempered by declared per-field tolerance floors','prior':'ridge weighted Fourier smoothness plus weak offset/amplitude parameter ridge','prior_mismatch':'additive truncated Fourier log-field versus full smooth random generator; permeability support matches configured benchmark prior','selection':'single deterministic initialization; Armijo objective descent only; no hidden truth','identifiability':'pressure-only, single forcing does not identify arbitrary permeability','posterior_claim':False}
    return ReconstructionResult(data,diagnostics,provenance,stop_reason=reason,corrected=False)
