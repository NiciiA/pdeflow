"""Shared physical-coordinate evaluation; never selects samples with reference truth."""
from pathlib import Path
import numpy as np
from pdeflow.runtime import read_json,write_json
from pdeflow.core import GridSpec

def interpolate_grid(values,grid,queries):
    """Periodic multilinear space / linear physical-time interpolation, no temporal extrapolation."""
    q=np.asarray(queries);axes=[np.asarray(grid.times),*[c.numpy() for c in grid.coordinates()]]
    if q.shape[1]!=grid.dimension+1 or not np.isfinite(q).all():raise ValueError('Invalid evaluation queries')
    if np.any(q[:,-1]<grid.times[0]-1e-8) or np.any(q[:,-1]>grid.times[-1]+1e-8):raise ValueError('Evaluation cannot extrapolate physical time')
    spatial=[]
    for j,(o,L) in enumerate(zip(grid.domain.origins,grid.domain.lengths)):
        if np.any(q[:,j]<o-1e-8) or np.any(q[:,j]>o+L+1e-8):raise ValueError('Evaluation cannot extrapolate spatial domain')
        spatial.append((q[:,j]-o)%L+o)
    padded=np.asarray(values)
    for j in range(grid.dimension):
        padded=np.concatenate([padded,np.take(padded,[0],axis=j+1)],axis=j+1);axes[j+1]=np.r_[axes[j+1],grid.domain.origins[j]+grid.domain.lengths[j]]
    points=np.stack([np.clip(q[:,-1],grid.times[0],grid.times[-1]),*spatial],-1)
    # Vectorized multilinear interpolation in physical coordinates; at most16 corners in3D+time.
    import itertools
    lows=[];weights=[]
    for j,a in enumerate(axes):
        lo=np.clip(np.searchsorted(a,points[:,j],side='right')-1,0,len(a)-2)
        lows.append(lo);weights.append((points[:,j]-a[lo])/(a[lo+1]-a[lo]))
    result=np.zeros(len(q))
    for corner in itertools.product((0,1),repeat=len(axes)):
        w=np.ones(len(q));ix=[]
        for j,offset in enumerate(corner):w*=weights[j] if offset else 1-weights[j];ix.append(lows[j]+offset)
        result+=w*padded[tuple(ix)]
    return result

def error_summary(samples,truth):
    a=np.asarray(samples);truth=np.asarray(truth);den=float(np.linalg.norm(truth));err=a-truth
    return {'mse_per_sample':np.mean(err**2,axis=1).tolist(),'rmse_per_sample':np.sqrt(np.mean(err**2,axis=1)).tolist(),'mae_per_sample':np.mean(np.abs(err),axis=1).tolist(),'relative_l2_per_sample':(np.linalg.norm(err,axis=1)/den).tolist() if den>1e-10*np.sqrt(truth.size) else None,'reference_l2':den,'reference_rms':float(np.sqrt(np.mean(truth**2))),'near_zero_convention':'Relative L2 unavailable when reference RMS<=1e-10; absolute errors always retained','ensemble_mean_rmse':float(np.sqrt(np.mean((a.mean(0)-truth)**2))),'spread_rms':float(np.sqrt(np.mean(a.var(0)))),'spread_meaning':'Empirical ensemble spread, not calibrated uncertainty'}

def common_grid_diagnostics(samples,grid,D):
    if len(grid.times)<3:return {'physics_rmse_per_sample':None,'physics_method':'Unavailable: common grid FD comparison needs at least3 physical times; coordinate model separately reports autodiff residuals'}
    u=np.asarray(samples).reshape(-1,len(grid.times),*grid.shape);lap=np.zeros_like(u)
    for j,h in enumerate(grid.spacing):lap+=(np.roll(u,1,j+2)-2*u+np.roll(u,-1,j+2))/h**2
    dt=np.gradient(u,np.asarray(grid.times),axis=1,edge_order=2)
    return {'physics_rmse_per_sample':np.sqrt(np.mean((dt-D*lap)**2,axis=tuple(range(1,u.ndim)))).tolist(),'physics_method':'Common central periodic FD spatial Laplacian and second-order physical-time gradient on the same query grid; discretization error included'}

def attach_reference(directory,reference):
    from pdeflow.reference import load
    from pdeflow.core import ProblemSpec
    directory=Path(directory);p=ProblemSpec.from_dict(read_json(directory/'problem.json'));rp,truth,manifest=load(reference)
    if p.family!=rp.family or p.fields!=rp.fields or p.grid.domain!=rp.grid.domain or p.known_parameters!=rp.known_parameters:raise ValueError('Reference physical contract differs; no comparison attached')
    with np.load(directory/'result.npz',allow_pickle=False) as f:q=f['queries'];samples=f['samples']
    spectral=rp.grid.dimension==2 and rp.grid.times[0]==0 and manifest['diagnostics'].get('solver')=='periodic_heat_v1'
    if spectral:
        from pdeflow.heat_continuous import FourierHeat
        target=FourierHeat(truth[0,0,0],rp.grid.domain.lengths,rp.grid.domain.origins).evaluate(q,rp.known_parameters['diffusivity'],(0.,rp.grid.times[-1]))
    else:target=interpolate_grid(truth[0,0],rp.grid,q)
    report={**error_summary(samples,target),'reference_id':manifest['id'],'reference_type':manifest['diagnostics']['classification'],'transfer':'Direct finite Fourier evolution at requested physical coordinates/times, no frame interpolation' if spectral else 'periodic multilinear spatial and linear time interpolation of reference; exact at saved nodes. No extrapolation. This is not fine-resolution truth validation.'}
    if read_json(directory/'metrics.json')['regular_grid']:
        report.update(common_grid_diagnostics(samples,p.grid,p.known_parameters['diffusivity']))
        if p.grid.dimension==2:
            from pdeflow.field_losses import trajectory_metrics
            provenance=read_json(directory/'metrics.json')
            report['field_metrics']=trajectory_metrics(samples.mean(0).reshape(len(p.grid.times),*p.grid.shape),target.reshape(len(p.grid.times),*p.grid.shape),p.grid.times,p.grid.domain.lengths,provenance.get('field_loss_provenance',{}),provenance.get('field_normalization_scale',1.))
        errors=(samples-target).reshape(len(samples),len(p.grid.times),-1)
        report['rmse_per_physical_time_per_sample']=np.sqrt(np.mean(errors**2,axis=2)).tolist();report['initial_rmse_per_sample']=np.sqrt(np.mean(errors[:,0]**2,axis=1)).tolist() if p.grid.times[0]==0 else None
    from pdeflow.runtime import atomic_write
    atomic_write(directory/'evaluation-reference.npz',lambda f:np.savez_compressed(f,reference=target))
    write_json(directory/'comparison.json',report);return report
