import numpy as np
import pytest
from pdeflow.workbench.playground import Heat,reference
from pdeflow.mesh.wave_presets import initial,exact_mode
from pdeflow.numerics.dynamic import SpectralState

@pytest.mark.parametrize('name',['standing','travelling','pulse','interference','velocity_pulse','multimode','mean_velocity'])
def test_complete_periodic_presets_derivative_and_reference(name):
    spec=Heat(family='wave',preset=name,lengths=[2.,.7],origins=[-.4,.2],wave_phase=.31,direction=-1,complexity=3);u,v,meta=initial(spec);assert np.isfinite(u).all() and np.isfinite(v).all();solver=SpectralState(u,v,tuple(spec.lengths),tuple(spec.origins),'wave');q=np.array([[.31,.43,.237],[.8,.6,.61],[1.2,.21,.83]]);h=1e-5;plus=q.copy();minus=q.copy();plus[:,-1]+=h;minus[:,-1]-=h;derivative=(solver.evaluate(plus,spec.c,(0.,1.),'u')-solver.evaluate(minus,spec.c,(0.,1.),'u'))/(2*h);np.testing.assert_allclose(derivative,solver.evaluate(q,spec.c,(0.,1.),'v'),atol=1e-6,rtol=1e-5)
    edge=q.copy();edge[:,0]=spec.origins[0];shift=edge.copy();shift[:,0]=spec.origins[0]+spec.lengths[0];np.testing.assert_allclose(solver.evaluate(edge,spec.c,(0.,1.),'u'),solver.evaluate(shift,spec.c,(0.,1.),'u'),atol=1e-8)
    if name=='velocity_pulse':assert np.max(abs(u))==0 and np.max(abs(v))>0
    if name=='mean_velocity':np.testing.assert_allclose(solver.frame(.7,spec.c,(0.,1.)),spec.background+.7*spec.velocity_background,atol=1e-10)
    if name in ('standing','travelling'):
        expected=exact_mode(spec,q[:,0],q[:,1],q[:,-1]);np.testing.assert_allclose(solver.evaluate(q,spec.c,(0.,1.)),expected,atol=1e-7)


def test_two_pulse_superposition_and_multimode_replay():
    from pdeflow.mesh.wave_presets import initial
    combined=Heat(family='wave',preset='interference',background=0.,amplitude=.6,amplitude2=-.3);a=initial(combined)[0];one=initial(combined.model_copy(update={'amplitude2':0.}))[0];two=initial(combined.model_copy(update={'amplitude':0.}))[0];np.testing.assert_allclose(a,one+two,atol=1e-12)
    cfg=Heat(family='wave',preset='multimode',seed=13);first=initial(cfg);second=initial(cfg);np.testing.assert_array_equal(first[0],second[0]);assert not np.array_equal(first[0],initial(cfg.model_copy(update={'seed':14}))[0])
