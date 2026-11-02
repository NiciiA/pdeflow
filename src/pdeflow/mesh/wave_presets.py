"""Complete periodic wave initial conditions; requested catalogue, not PDF formulas."""
import math
import numpy as np

DESCRIPTIONS={
 'standing':'A periodic standing mode oscillates in place. Temporal phase fixes both initial displacement and velocity.',
 'travelling':'A periodic plane wave with matched velocity travels in the selected direction at phase speed c.',
 'pulse':'A smooth periodic displacement pulse with zero initial velocity expands/splits; it is not a one-way travelling pulse.',
 'interference':'Two smooth periodic pulses superpose. Signs, centers and widths control their interference; initial velocity is zero by default.',
 'velocity_pulse':'Initial displacement is zero; a localized velocity kick creates the subsequent wave.',
 'multimode':'A seeded mixture of periodic travelling modes. Each mode has its matched initial velocity.',
 'mean_velocity':'Advanced zero mode: constant velocity gives linearly changing mean displacement.',
}


def initial(spec,shape=(64,64)):
    x,y=np.meshgrid(np.arange(shape[0])/shape[0],np.arange(shape[1])/shape[1],indexing='ij');L=spec.lengths;kx=2*np.pi*spec.kx/L[0];ky=2*np.pi*spec.ky/L[1];omega=spec.c*np.hypot(kx,ky);phase=2*np.pi*(spec.kx*x+spec.ky*y)+spec.phix;zero=np.zeros(shape)
    def pulse(center,width):
        # von Mises periodic analogue of a Gaussian; width is domain fraction.
        return np.exp(-2*(np.sin(np.pi*(x-center[0]))**2+np.sin(np.pi*(y-center[1]))**2)/(2*np.pi*width)**2)
    name=spec.preset
    if name in ('cosine','standing'):
        spatial=spec.amplitude*np.cos(2*np.pi*spec.kx*x+spec.phix)*np.cos(2*np.pi*spec.ky*y+spec.phiy);u=spec.background+spatial*np.cos(spec.wave_phase);v=-omega*spatial*np.sin(spec.wave_phase)
    elif name=='travelling':u=spec.background+spec.amplitude*np.cos(phase);v=spec.direction*omega*spec.amplitude*np.sin(phase)
    elif name in ('pulse','velocity_pulse','interference','signed_pulses'):
        field=spec.amplitude*pulse(spec.center1,spec.width1)
        if name in ('interference','signed_pulses'):field=field+spec.amplitude2*pulse(spec.center2,spec.width2)
        u=spec.background+field;v=zero.copy()
        if name=='velocity_pulse':u=zero.copy();v=field
    elif name=='multimode':
        rng=np.random.default_rng(spec.seed);u=np.full(shape,spec.background);v=zero.copy()
        for i in range(spec.complexity):
            a=spec.amplitude/math.sqrt(spec.complexity);m=int(rng.integers(1,max(2,abs(spec.kx)+1)));n=int(rng.integers(-max(1,abs(spec.ky)),max(1,abs(spec.ky))+1));theta=2*np.pi*(m*x+n*y)+rng.uniform(-np.pi,np.pi);w=spec.c*np.hypot(2*np.pi*m/L[0],2*np.pi*n/L[1]);direction=rng.choice([-1.,1.]);u+=a*np.cos(theta);v+=direction*a*w*np.sin(theta)
    elif name=='mean_velocity':u=np.full(shape,spec.background);v=np.full(shape,spec.velocity_background)
    else:return None
    return u,v,{'preset':name,'description':DESCRIPTIONS.get(name,DESCRIPTIONS['interference']),'omega':float(omega),'wave_numbers':[float(kx),float(ky)],'direction':spec.direction,'width_units':'Fraction of each periodic domain extent','complete_initial_state':True,'velocity_override':False}


def exact_mode(spec,x,y,time,quantity='u'):
    X=(x-spec.origins[0])/spec.lengths[0];Y=(y-spec.origins[1])/spec.lengths[1];omega=spec.c*np.hypot(2*np.pi*spec.kx/spec.lengths[0],2*np.pi*spec.ky/spec.lengths[1])
    if spec.preset=='travelling':
        phase=2*np.pi*(spec.kx*X+spec.ky*Y)+spec.phix-spec.direction*omega*time
        return spec.background+spec.amplitude*np.cos(phase) if quantity=='u' else spec.direction*omega*spec.amplitude*np.sin(phase)
    if spec.preset in ('standing','cosine'):
        spatial=spec.amplitude*np.cos(2*np.pi*spec.kx*X+spec.phix)*np.cos(2*np.pi*spec.ky*Y+spec.phiy);phase=omega*time+spec.wave_phase
        return spec.background+spatial*np.cos(phase) if quantity=='u' else -omega*spatial*np.sin(phase)
    if spec.preset=='mean_velocity':return np.full_like(x,spec.background+spec.velocity_background*time if quantity=='u' else spec.velocity_background)
    return None
