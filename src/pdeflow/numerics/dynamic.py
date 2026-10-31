"""Finite, real periodic Fourier heat/wave references. Never used by neural inference."""
from dataclasses import dataclass
from collections import OrderedDict
import numpy as np
from pdeflow.heat_continuous import bounded_time

@dataclass
class SpectralState:
    initial:np.ndarray
    velocity:np.ndarray|None=None
    lengths:tuple=(1.,1.)
    origins:tuple=(0.,0.)
    family:str='heat'
    mode_limit:int|None=None
    backend:str='optimized'
    def __post_init__(self):
        self.initial=np.array(self.initial,dtype=np.float64,copy=True)
        if self.initial.ndim!=2 or not np.isfinite(self.initial).all():raise ValueError('Reference needs a finite scalar2D initial field')
        if self.family not in ('heat','wave'):raise ValueError('Only registered smooth heat/wave reference physics')
        if self.family=='wave' and self.velocity is None:raise ValueError('Wave requires an explicit initial velocity (zero must be explicit)')
        self.velocity=np.zeros_like(self.initial) if self.velocity is None else np.array(self.velocity,dtype=np.float64,copy=True)
        if self.velocity.shape!=self.initial.shape or not np.isfinite(self.velocity).all():raise ValueError('Initial u/v shape or values invalid')
        if len(self.lengths)!=2 or min(self.lengths)<=0:raise ValueError('Positive rectangular physical domain lengths required')
        # Split every Nyquist axis into symmetric half coefficients. This defines
        # the real tensor-product trigonometric interpolant even away from nodes.
        u=np.fft.fftn(self.initial)/self.initial.size;v=np.fft.fftn(self.velocity)/self.initial.size
        axes=[np.arange(-(n//2),(n//2)+1) if n%2==0 else np.arange(-(n//2),n//2+1) for n in self.initial.shape]
        self.modes=np.stack(np.meshgrid(*axes,indexing='ij'),-1).reshape(-1,2)
        indices=tuple(self.modes[:,j]%n for j,n in enumerate(self.initial.shape))
        weights=np.ones(len(self.modes))
        for j,n in enumerate(self.initial.shape):
            if n%2==0:weights[np.abs(self.modes[:,j])==n//2]*=.5
        self.u=u[indices]*weights;self.v=v[indices]*weights
        # Generated examples already have an EXPLICIT spectral bandlimit. Remove
        # only out-of-band roundoff, checking its absolute reconstruction bound.
        self.discarded_bound=0.
        if self.mode_limit is not None and self.backend!='legacy':
            keep=(np.abs(self.modes)<=self.mode_limit).all(1)
            self.discarded_bound=float(np.abs(self.u[~keep]).sum()+np.abs(self.v[~keep]).sum())
            if self.discarded_bound>1e-12*max(1.,float(np.abs(self.initial).max()),float(np.abs(self.velocity).max())):raise ValueError('Declared reference bandlimit would discard physical signal')
            self.modes=self.modes[keep];self.u=self.u[keep];self.v=self.v[keep]
        self.wave=self.modes*2*np.pi/np.asarray(self.lengths);self.wave2=(self.wave**2).sum(1)
        if self.backend not in ('optimized','legacy'):raise ValueError('Reference backend: optimized or legacy')
        for value in (self.initial,self.velocity,self.modes,self.wave,self.wave2,self.u,self.v):value.setflags(write=False)
        self._coefficients=OrderedDict();self._frames=OrderedDict();self.profiler=None
    def coefficients(self,t,parameter):
        if not np.isfinite(parameter) or parameter<=0:raise ValueError('D/c must be finite positive')
        t=np.asarray(t)[...,None]
        if self.family=='heat':return self.u*np.exp(-parameter*t*self.wave2),None
        omega=parameter*np.sqrt(self.wave2);co=np.cos(omega*t);si_over=t*np.sinc(omega*t/np.pi)
        return self.u*co+self.v*si_over,-self.u*omega*np.sin(omega*t)+self.v*co
    def evaluate(self,queries,parameter,interval=(0.,1.),quantity='u',chunk=256):
        return self.evaluate_state(queries,parameter,interval,(quantity,),chunk)[:,0]
    def evaluate_state(self,queries,parameter,interval=(0.,1.),quantities=('u','v'),chunk=256):
        q=np.asarray(queries,dtype=np.float64)
        if q.ndim!=2 or q.shape[1]!=3 or not np.isfinite(q).all():raise ValueError('Reference queries require finite[x,y,t]')
        if any(quantity not in ('u','v') or quantity=='v' and self.family!='wave' for quantity in quantities):raise ValueError('Velocity is an active state only for wave')
        bounded_time(q[:,-1],interval)
        if np.any(q[:,:2]<self.origins) or np.any(q[:,:2]>np.asarray(self.origins)+self.lengths):raise ValueError('Reference query outside domain')
        if self.profiler is not None:
            with self.profiler.measure('reference_evaluation',host=True):return self._evaluate(q,parameter,quantities,chunk)
        return self._evaluate(q,parameter,quantities,chunk)
    def _evaluate(self,q,parameter,quantities,chunk):
        out=[]
        for start in range(0,len(q),chunk):
            a=q[start:start+chunk];x=a[:,:2]-self.origins
            phase=x[:,0,None]*self.wave[:,0]+x[:,1,None]*self.wave[:,1]
            phase=np.exp(1j*phase)
            times=a[:,-1]
            if self.backend!='legacy' and np.all(times==times[0]):
                key=(float(times[0]),float(parameter))
                if key not in self._coefficients:
                    self._coefficients[key]=self.coefficients(times[0],parameter)
                    if len(self._coefficients)>8:self._coefficients.popitem(last=False)
                u,v=self._coefficients[key]
            else:u,v=self.coefficients(times,parameter)
            out.append(np.stack([np.real((phase*(u if quantity=='u' else v)).sum(1)) for quantity in quantities],-1))
        return np.concatenate(out) if out else np.empty((0,len(quantities)))
    def frame(self,time,parameter,interval=(0.,1.),shape=None,quantity='u'):
        if self.profiler is not None:
            with self.profiler.measure('reference_evaluation',host=True):return self._frame(time,parameter,interval,shape,quantity)
        return self._frame(time,parameter,interval,shape,quantity)
    def _frame(self,time,parameter,interval,shape,quantity):
        t=float(bounded_time(time,interval));shape=tuple(shape or self.initial.shape)
        key=(t,float(parameter),shape,quantity)
        if self.backend!='legacy' and key in self._frames:return self._frames[key]
        u,v=self.coefficients(t,parameter)
        if quantity not in ('u','v') or quantity=='v' and v is None:raise ValueError('Unsupported reference state quantity')
        spectrum=np.zeros(shape,dtype=np.complex128);indices=tuple((self.modes[:,j]%shape[j]) for j in range(2))
        np.add.at(spectrum,indices,u if quantity=='u' else v)
        result=np.fft.ifftn(spectrum*np.prod(shape)).real
        if self.backend!='legacy':
            result.setflags(write=False);self._frames[key]=result
            if len(self._frames)>8:self._frames.popitem(last=False)
        return result
    def invariants(self,time,parameter):
        u,v=self.coefficients(float(time),parameter);zero=np.all(self.modes==0,axis=1)
        if self.family=='heat':return {'mean_u':float(u[zero].real.sum()),'l2_energy':float(.5*np.prod(self.lengths)*np.abs(u).dot(np.abs(u)))}
        return {'mean_u':float(u[zero].real.sum()),'mean_v':float(v[zero].real.sum()),'energy':float(.5*np.prod(self.lengths)*(np.abs(v)**2+parameter**2*self.wave2*np.abs(u)**2).sum())}


def evaluate_batch(solvers,queries,parameters,interval=(0.,1.),quantities=('u','v'),chunk=256):
    """Bounded batched Fourier evaluation across compatible independent fields.

    No full spatial grid or space-time trajectory. All modes retained by each
    solver participate. Wave zero frequency uses t*sinc(omega*t/pi), as above.
    """
    q=np.asarray(queries,dtype=np.float64);parameters=np.asarray(parameters)
    if q.ndim!=3 or q.shape[0]!=len(solvers) or q.shape[2]!=3 or len(parameters)!=len(solvers):raise ValueError('Batched reference expects [batch,points,3] and one parameter per field')
    if not np.isfinite(q).all() or not np.isfinite(parameters).all() or (parameters<=0).any():raise ValueError('Invalid batched reference coordinates/parameter')
    if not solvers:return np.empty((*q.shape[:2],len(quantities)))
    first=solvers[0]
    compatible=all(s.backend!='legacy' and np.array_equal(s.wave,first.wave) and s.origins==first.origins and s.lengths==first.lengths for s in solvers)
    if any(v not in ('u','v') for v in quantities):raise ValueError('Unknown batched reference quantity')
    if not compatible:
        return np.stack([np.stack([np.zeros(len(x)) if name=='v' and s.family=='heat' else s.evaluate(x,p,interval,name) for name in quantities],-1) for s,x,p in zip(solvers,q,parameters)])
    bounded_time(q[...,-1],interval)
    if np.any(q[...,:2]<first.origins) or np.any(q[...,:2]>np.asarray(first.origins)+first.lengths):raise ValueError('Batched reference outside physical domain')
    if any(v not in ('u','v') for v in quantities):raise ValueError('Unknown batched reference quantity')
    out=np.empty((*q.shape[:2],len(quantities)));u0=np.stack([s.u for s in solvers])[:,None];v0=np.stack([s.v for s in solvers])[:,None]
    wave=first.wave;wave2=first.wave2;heat=np.array([s.family=='heat' for s in solvers])[:,None,None]
    # Bound TOTAL query-mode temporary memory, not chunk per example.
    block=max(1,chunk//len(solvers));p=parameters[:,None,None]
    for start in range(0,q.shape[1],block):
        a=q[:,start:start+block];t=a[...,-1,None];x=a[...,:2]-first.origins
        phase=np.exp(1j*(x[...,0,None]*wave[:,0]+x[...,1,None]*wave[:,1]))
        omega=p*np.sqrt(wave2);co=np.cos(omega*t);si=t*np.sinc(omega*t/np.pi)
        u=np.where(heat,u0*np.exp(-p*t*wave2),u0*co+v0*si)
        v=np.where(heat,0.,-u0*omega*np.sin(omega*t)+v0*co)
        for j,name in enumerate(quantities):out[:,start:start+block,j]=np.real((phase*(u if name=='u' else v)).sum(-1))
    return out
