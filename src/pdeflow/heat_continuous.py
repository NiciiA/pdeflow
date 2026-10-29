"""Finite periodic Fourier heat reference at ordered physical space-time points.

Even-grid Nyquist ambiguity is resolved by taking the real Fourier sum. This
recovers nodal real data; it is a declared finite representation of a drawing.
No animation-frame interpolation is used. Tiny coefficients are discarded with
an explicit uniform absolute bound, not a change of temporal discretization.
"""
from dataclasses import dataclass
import numpy as np

TIME_TOLERANCE=1e-7


def bounded_time(times,interval):
    t=np.asarray(times,dtype=np.float64)
    if not np.isfinite(t).all() or np.any(t<interval[0]-TIME_TOLERANCE) or np.any(t>interval[1]+TIME_TOLERANCE):
        raise ValueError(f'Physical time must be finite and inside [{interval[0]}, {interval[1]}]; no extrapolation (endpoint tolerance 1e-7)')
    # Canonicalize numerical endpoint roundoff only, never material invalid data.
    return np.clip(t,*interval)


@dataclass
class FourierHeat:
    initial: np.ndarray
    lengths: tuple=(1.,1.)
    origins: tuple=(0.,0.)
    def __post_init__(self):
        self.initial=np.asarray(self.initial,dtype=np.float64)
        if self.initial.ndim!=2 or not np.isfinite(self.initial).all() or min(self.initial.shape)<2:raise ValueError('Finite scalar 2D initial grid required')
        if len(self.lengths)!=2 or len(self.origins)!=2 or not np.isfinite([*self.lengths,*self.origins]).all() or min(self.lengths)<=0:raise ValueError('Finite domain with positive lengths required')
        coeff=np.fft.fftn(self.initial)/self.initial.size
        k=np.meshgrid(*[2*np.pi*np.fft.fftfreq(n,d=L/n) for n,L in zip(self.initial.shape,self.lengths)],indexing='ij')
        active=np.abs(coeff)>1e-13
        active[0,0]=True # preserve physical mean even for very small constant fields
        self.omitted_absolute_bound=float(np.abs(coeff[~active]).sum())
        self.coeff=coeff[active];self.wave=np.stack([a[active] for a in k],-1);self.wave2=np.square(self.wave).sum(1)
    def evaluate(self,points,D,interval=(0.,1.),chunk=128):
        q=np.asarray(points,dtype=np.float64)
        if q.ndim!=2 or q.shape[1]!=3 or not np.isfinite(q[:,:2]).all() or not np.isfinite(D) or D<=0:raise ValueError('Finite [x,y,physical_time] points and positive diffusivity required')
        t=bounded_time(q[:,2],interval)
        if np.any(q[:,:2]<self.origins) or np.any(q[:,:2]>np.asarray(self.origins)+self.lengths):raise ValueError('Reference query outside physical domain')
        result=[]
        for start in range(0,len(q),chunk):
            a=q[start:start+chunk];phase=(a[:,:2]-self.origins)@self.wave.T
            result.append(np.real((np.exp(1j*phase-D*t[start:start+chunk,None]*self.wave2)*self.coeff).sum(1)))
        return np.concatenate(result) if result else np.empty(0)
    def frame(self,time,D,interval=(0.,1.),shape=None):
        # FFT on original nodes avoids a dense point-by-mode matrix for animation.
        t=float(bounded_time(time,interval));shape=shape or self.initial.shape
        if tuple(shape)==self.initial.shape:
            k=np.meshgrid(*[2*np.pi*np.fft.fftfreq(n,d=L/n) for n,L in zip(shape,self.lengths)],indexing='ij')
            return np.fft.ifftn(np.fft.fftn(self.initial)*np.exp(-D*t*sum(a*a for a in k))).real
        x,y=np.meshgrid(*[o+np.arange(n)*L/n for o,L,n in zip(self.origins,self.lengths,shape)],indexing='ij')
        return self.evaluate(np.c_[x.ravel(),y.ravel(),np.full(x.size,t)],D,interval).reshape(shape)
