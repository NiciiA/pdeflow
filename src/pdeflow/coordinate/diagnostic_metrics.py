"""Evaluation-only physical field/sensor statistics, never training conditioning."""
import numpy as np
from pdeflow.field_losses import metric_record,trajectory_metrics


def moments(value):
    a=np.asarray(value,dtype=np.float64)
    return {'mean':float(a.mean()),'variance':float(a.var()),'minimum':float(a.min()),'maximum':float(a.max())}


def spectral_error(prediction,reference,cutoff=2.):
    """Parseval MSE by radial integer domain-mode index, including phase error."""
    a=np.asarray(prediction,dtype=np.float64);b=np.asarray(reference,dtype=np.float64)
    modes=np.meshgrid(*[np.fft.fftfreq(n)*n for n in b.shape],indexing='ij');low=sum(k*k for k in modes)<=cutoff**2
    delta=np.fft.fftn(a-b)/b.size;ref=np.fft.fftn(b)/b.size
    return {'low_mode_mse':float(np.abs(delta[low]).dot(np.abs(delta[low]))),
            'high_mode_mse':float(np.abs(delta[~low]).dot(np.abs(delta[~low]))),
            'reference_high_mode_power_fraction':float((abs(ref[~low])**2).sum()/max((abs(ref)**2).sum(),1e-30)),
            'radial_integer_mode_cutoff':cutoff,'definition':'Complex Fourier error (phase retained); low+high = spatial MSE by Parseval; normalized domain modes, not physical frequency units'}


def frame_metrics(prediction,reference,lengths,origins,loss,normalization):
    a=np.asarray(prediction,dtype=np.float64);b=np.asarray(reference,dtype=np.float64)
    r=metric_record(a,b,lengths,loss,normalization['scale'])
    def location(x,maximum):
        index=np.unravel_index(np.argmax(x) if maximum else np.argmin(x),x.shape)
        return [float(o+l*i/n) for o,l,i,n in zip(origins,lengths,index,x.shape)]
    r.update(reference=moments(b),prediction=moments(a),signed_mean_drift=float((a-b).mean()),
             constant_training_mean_mse=float(((normalization['mean']-b)**2).mean()),
             peak={'reference':float(b.max()),'prediction':float(a.max()),'reference_location':location(b,True),'prediction_location':location(a,True)},
             trough={'reference':float(b.min()),'prediction':float(a.min()),'reference_location':location(b,False),'prediction_location':location(a,False)},
             spectral=spectral_error(a,b))
    return r


def field_metrics(samples,reference,times,c,quantity,normalization):
    a=np.asarray(samples,dtype=np.float64);b=np.asarray(reference,dtype=np.float64)
    if a.ndim!=4 or a.shape[1:]!=b.shape or b.shape[0]!=len(times) or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Diagnostic fields must share [sample,time,x,y] coordinates and finite physical values')
    def one(x):
        base=np.full_like(b,normalization['mean']);summary=trajectory_metrics(x,b,times,c.lengths,c.field_loss,normalization['scale'])
        summary['constant_training_mean']=trajectory_metrics(base,b,times,c.lengths,c.field_loss,normalization['scale'])['uniform_physical']
        summary['frames']=[{'time':float(t),**frame_metrics(p,r,c.lengths,c.origins,c.field_loss,normalization)} for t,p,r in zip(times,x,b)]
        return summary
    return {'quantity':quantity,'units':'physical','per_sample':[one(x) for x in a],'ensemble_mean':one(a.mean(0)),
            'sample_count':len(a),'spread_rms':float(np.sqrt(a.var(0).mean())),
            'selection':'All samples and ensemble mean; no hidden-reference best-of-N selection',
            'relative_l2_floor':'Unavailable for reference RMS<=1e-10 physical units; see metric definitions',
            'aggregation':'Per sample spatial quadrature then normalized trapezoid in physical time; RMSE=sqrt(aggregate MSE). Initial/early errors remain in frames.'}


def sensor_metrics(predictions,rows,clean=None,atol=1e-5,rtol=1e-4):
    pred=np.asarray(predictions,dtype=np.float64);y=np.asarray([r['value'] for r in rows]);sigma=np.asarray([r.get('sigma',0.) for r in rows])
    if pred.ndim!=2 or pred.shape[1]!=len(rows) or not np.isfinite(pred).all():raise ValueError('Exact sensor predictions must be [sample,sensor] and finite')
    truth=None if clean is None else np.asarray(clean,dtype=np.float64)
    if truth is not None and truth.shape!=y.shape:raise ValueError('Clean sensor truth shape mismatch')
    if not rows:return {'count':0,'per_sample':[],'ensemble_mean':None}
    def one(x,ix):
        e=x[ix]-y[ix];noise=sigma[ix];positive=noise>0;zero=~positive
        out={'mse':float(np.mean(e*e)),'rmse':float(np.sqrt(np.mean(e*e))),'mae':float(np.mean(abs(e))),
             'noisy_count':int(positive.sum()),'noiseless_count':int(zero.sum()),'standardized_noisy':None,'noiseless':None}
        if truth is not None:
            d=x[ix]-truth[ix];out['against_clean_truth']={'mse':float(np.mean(d*d)),'mae':float(np.mean(abs(d)))}
        if positive.any():
            standardized=e[positive]/noise[positive]
            out['standardized_noisy']={'mean':float(standardized.mean()),'rms':float(np.sqrt((standardized**2).mean())),'maximum_absolute':float(abs(standardized).max())}
        if zero.any():
            tol=atol+rtol*abs(y[ix][zero]);out['noiseless']={'mse':float(np.mean(e[zero]**2)),'mae':float(np.mean(abs(e[zero]))),'within_tolerance_fraction':float(np.mean(abs(e[zero])<=tol)),'atol':atol,'rtol':rtol}
        return out
    def group(ix):
        q=np.quantile(pred[:,ix],[.05,.95],axis=0)
        return {'count':int(ix.sum()),'per_sample':[one(x,ix) for x in pred],'ensemble_mean':one(pred.mean(0),ix),
                'spread_rms':float(np.sqrt(pred[:,ix].var(0).mean())),
                'empirical_90_interval':{'measured_value_coverage':float(np.mean((y[ix]>=q[0])&(y[ix]<=q[1]))),'clean_value_coverage':None if truth is None else float(np.mean((truth[ix]>=q[0])&(truth[ix]<=q[1]))),'interpretation':'Tiny empirical ensemble, not calibrated uncertainty; conditioning-data fit is not held-out coverage'}}
    quantities={name:group(np.array([r['field']==name for r in rows])) for name in sorted({r['field'] for r in rows})}
    return {**group(np.ones(len(rows),dtype=bool)),'by_quantity':quantities,'noise_model':'Independent absolute sigma per scalar observation; zero sigma uses explicit physical tolerance, never division by zero',
            'combined_score_limit':'Mixed u/v physical units are not a normalized wave score; use by_quantity',
            'physical_times':sorted(set(float(r['time']) for r in rows))}
