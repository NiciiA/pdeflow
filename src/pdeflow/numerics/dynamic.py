"""Finite, real periodic Fourier heat/wave references. Never used by neural inference."""
from dataclasses import dataclass
import numpy as np
from pdeflow.heat_continuous import bounded_time

@dataclass
class SpectralState:
    initial:np.ndarray
    velocity:np.ndarray|None=None
    lengths:tuple=(1.,1.)
    origins:tuple=(0.,0.)
    family:str='heat'
    def __post_init__(self):
        self.initial=np.asarray(self.initial,dtype=np.float64)
        if self.initial.ndim!=2 or not np.isfinite(self.initial).all():raise ValueError('Reference needs a finite scalar2D initial field')
        if self.family not in ('heat','wave'):raise ValueError('Only registered smooth heat/wave reference physics')
        if self.family=='wave' and self.velocity is None:raise ValueError('Wave requires an explicit initial velocity (zero must be explicit)')
        self.velocity=np.zeros_like(self.initial) if self.velocity is None else np.asarray(self.velocity,dtype=np.float64)
        if self.velocity.shape!=self.initial.shape or not np.isfinite(self.velocity).all():raise ValueError('Initial u/v shape or values invalid')
        if len(self.lengths)!=2 or min(self.lengths)<=0:raise ValueError('Positive rectangular physical domain lengths required')
        # Split every Nyquist axis into symmetric half coefficients. This defines
        # the real tensor-product trigonometric interpolant even away from nodes.
        u=np.fft.fftn(self.initial)/self.initial.size;v=np.fft.fftn(self.velocity)/self.initial.size
        modes=[];a=[];b=[]
        for i in np.ndindex(self.initial.shape):
            choices=[]
            for j,n in zip(i,self.initial.shape):
                k=j if j<=(n-1)//2 else j-n
                choices.append([(-n//2,.5),(n//2,.5)] if n%2==0 and j==n//2 else [(k,1.)])
            for kx,wx in choices[0]:
                for ky,wy in choices[1]:modes.append((kx,ky));a.append(u[i]*wx*wy);b.append(v[i]*wx*wy)
        self.modes=np.asarray(modes);self.wave=self.modes*2*np.pi/np.asarray(self.lengths);self.wave2=(self.wave**2).sum(1);self.u=np.asarray(a);self.v=np.asarray(b)
    def coefficients(self,t,parameter):
        if not np.isfinite(parameter) or parameter<=0:raise ValueError('D/c must be finite positive')
        t=np.asarray(t)[...,None]
        if self.family=='heat':return self.u*np.exp(-parameter*t*self.wave2),None
        omega=parameter*np.sqrt(self.wave2);co=np.cos(omega*t);si_over=t*np.sinc(omega*t/np.pi)
        return self.u*co+self.v*si_over,-self.u*omega*np.sin(omega*t)+self.v*co
    def evaluate(self,queries,parameter,interval=(0.,1.),quantity='u',chunk=256):
        q=np.asarray(queries,dtype=np.float64)
        if q.ndim!=2 or q.shape[1]!=3 or not np.isfinite(q).all():raise ValueError('Reference queries require finite[x,y,t]')
        if quantity not in ('u','v') or quantity=='v' and self.family!='wave':raise ValueError('Velocity is an active state only for wave')
        bounded_time(q[:,-1],interval)
        if np.any(q[:,:2]<self.origins) or np.any(q[:,:2]>np.asarray(self.origins)+self.lengths):raise ValueError('Reference query outside domain')
        out=[]
        for start in range(0,len(q),chunk):
            a=q[start:start+chunk];x=a[:,:2]-self.origins
            phase=x[:,0,None]*self.wave[:,0]+x[:,1,None]*self.wave[:,1]
            u,v=self.coefficients(a[:,-1],parameter);out.append(np.real((np.exp(1j*phase)*(u if quantity=='u' else v)).sum(1)))
        return np.concatenate(out) if out else np.empty(0)
    def frame(self,time,parameter,interval=(0.,1.),shape=None,quantity='u'):
        t=float(bounded_time(time,interval));shape=tuple(shape or self.initial.shape)
        u,v=self.coefficients(t,parameter)
        if quantity not in ('u','v') or quantity=='v' and v is None:raise ValueError('Unsupported reference state quantity')
        spectrum=np.zeros(shape,dtype=np.complex128);indices=tuple((self.modes[:,j]%shape[j]) for j in range(2))
        np.add.at(spectrum,indices,u if quantity=='u' else v)
        return np.fft.ifftn(spectrum*np.prod(shape)).real
    def invariants(self,time,parameter):
        u,v=self.coefficients(float(time),parameter);zero=np.all(self.modes==0,axis=1)
        if self.family=='heat':return {'mean_u':float(u[zero].real.sum()),'l2_energy':float(.5*np.prod(self.lengths)*np.abs(u).dot(np.abs(u)))}
        return {'mean_u':float(u[zero].real.sum()),'mean_v':float(v[zero].real.sum()),'energy':float(.5*np.prod(self.lengths)*(np.abs(v)**2+parameter**2*self.wave2*np.abs(u)**2).sum())}
