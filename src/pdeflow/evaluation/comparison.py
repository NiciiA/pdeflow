"""Explicit physical quadrature, coordinate transfer and reference error reports."""
from dataclasses import dataclass
import numpy as np
from pdeflow.core import ProblemSpec

@dataclass(frozen=True)
class EvaluationSpec:
    policy: str='native'
    shared_times_only: bool=False
    relative_norm_tolerance: float=1e-12
    metrics: tuple[str,...]=('mse','rmse','mae','relative_l2')

def align(reference: np.ndarray,reference_problem: ProblemSpec,problem: ProblemSpec,spec: EvaluationSpec):
    g,r=problem.grid,reference_problem.grid
    if problem.family!=reference_problem.family or [(f.name,f.units) for f in problem.fields]!=[(f.name,f.units) for f in reference_problem.fields]:raise ValueError('Reference family/field names/units mismatch')
    if problem.operator_spec!=reference_problem.operator_spec:raise ValueError('Reference equation operator differs')
    if g.shape==r.shape and problem.known_fields!=reference_problem.known_fields:raise ValueError('Reference known fields differ')
    if problem.known_parameters!=reference_problem.known_parameters:raise ValueError('Reference physical parameters differ')
    if g.domain!=r.domain or g.convention!=r.convention or g.temporal_kind!=r.temporal_kind:raise ValueError('Reference domain/origins/boundary/coordinate convention mismatch')
    if spec.policy not in ('native','restrict_reference'):raise ValueError('Comparison policy: native or restrict_reference; no implicit interpolation')
    if tuple(reference.shape)!=(1,len(problem.fields),len(r.times),*r.shape):raise ValueError('Reference metadata/array layout mismatch')
    out=np.asarray(reference,dtype=np.float64)[0]
    transfer='identity'
    if spec.policy=='native' and g.shape!=r.shape:raise ValueError('Native comparison requires matching coordinates/grid; choose explicit restrict_reference')
    if g.shape!=r.shape:
        if any(a<b or a%b for a,b in zip(r.shape,g.shape)):raise ValueError('Restriction requires nested integer refinement of every spatial axis')
        ratios=[a//b for a,b in zip(r.shape,g.shape)]
        for axis,(coarse,ratio) in enumerate(zip(g.shape,ratios),start=2):
            if g.convention=='periodic_nodes_endpoint_excluded':out=np.take(out,np.arange(coarse)*ratio,axis=axis)
            else:
                shape=list(out.shape);shape[axis:axis+1]=[coarse,ratio]
                out=out.reshape(shape).mean(axis=axis+1)
        transfer='coincident periodic nodes' if g.convention=='periodic_nodes_endpoint_excluded' else 'conservative cell-volume block averages'
    if set(problem.known_fields)!=set(reference_problem.known_fields):raise ValueError('Reference known field names differ')
    known_transfers={}
    if g.shape!=r.shape:
        for name,value in reference_problem.known_fields.items():
            transferred=np.asarray(value,dtype=float)
            for axis,(coarse,ratio) in enumerate(zip(g.shape,ratios)):
                if g.convention=='periodic_nodes_endpoint_excluded':transferred=np.take(transferred,np.arange(coarse)*ratio,axis=axis)
                else:
                    shape=list(transferred.shape);shape[axis:axis+1]=[coarse,ratio]
                    transferred=transferred.reshape(shape).mean(axis=axis+1)
            if not np.allclose(transferred,np.asarray(problem.known_fields[name]),rtol=1e-9,atol=1e-12):raise ValueError(f'Restricted reference known field {name} differs from reconstruction input; use the matching restricted forcing/coefficient')
            known_transfers[name]=transfer
    times=[t for t in g.times if t in r.times]
    if not times or (not spec.shared_times_only and (g.times!=r.times)):raise ValueError('Physical times differ; choose shared_times_only explicitly, or supply matching saved times')
    reconstruction_indices=[g.times.index(t) for t in times]
    reference_indices=[r.times.index(t) for t in times]
    out=out[:,reference_indices]
    return out,reconstruction_indices,{'reference_grid':r.to_dict() if hasattr(r,'to_dict') else __import__('dataclasses').asdict(r),'reconstruction_grid':__import__('dataclasses').asdict(g),
        'evaluation_grid':{**__import__('dataclasses').asdict(g),'times':times},'spatial_transfer':transfer,'known_field_transfers':known_transfers,'time_transfer':'exact shared saved times; no interpolation','shared_times_only':spec.shared_times_only,
        'warning':'Fine/coarse discretizations can differ. Error is not solely attributable to the learned model.' if transfer!='identity' else None}

def _errors(prediction,truth,weights,mask,tolerance):
    w=weights*mask;total=float(w.sum())
    if total==0:return {'mse':None,'rmse':None,'mae':None,'relative_l2':None,'status':'no evaluation points','quadrature_volume':0.,'point_count':0}
    difference=prediction-truth;mse=float((w*difference**2).sum()/total);norm=float(np.sqrt((w*truth**2).sum()/total))
    return {'mse':mse,'rmse':float(np.sqrt(mse)),'mae':float((w*np.abs(difference)).sum()/total),'relative_l2':float(np.sqrt(mse)/norm) if norm>tolerance else None,
        'reference_rms_norm':norm,'status':'ok' if norm>tolerance else 'relative L2 undefined: near-zero reference RMS norm','quadrature_volume':total,'point_count':int(mask.sum())}

def compare(samples,problem: ProblemSpec,observations,reference,reference_problem=None,weights=None,spec: EvaluationSpec | None=None):
    spec=spec or EvaluationSpec()
    if not np.isfinite(spec.relative_norm_tolerance) or spec.relative_norm_tolerance<=0:raise ValueError('Positive finite relative norm tolerance required')
    if set(spec.metrics)-{'mse','rmse','mae','relative_l2'} or not spec.metrics:raise ValueError('Select supported error metrics')
    reference_problem=reference_problem or problem
    truth,time_indices,transfer=align(np.asarray(reference),reference_problem,problem,spec)
    samples=np.asarray(samples,dtype=np.float64)
    if samples.shape[0]!=1 or samples.shape[2:]!=(len(problem.fields),len(problem.grid.times),*problem.grid.shape):raise ValueError('Expected samples[1,particle,field,time,*space]')
    candidates=samples[0][:,:,time_indices]
    if not np.isfinite(candidates).all() or not np.isfinite(truth).all():raise ValueError('Comparison fields must be finite')
    w=np.asarray(weights,dtype=float).reshape(-1) if weights is not None else np.ones(len(candidates))
    if w.shape!=(len(candidates),) or not np.isfinite(w).all() or (w<0).any() or w.sum()<=0:raise ValueError('Invalid population weights')
    w=w/w.sum();mean=np.einsum('p,pc...->c...',w,candidates)
    g=problem.grid;times=[g.times[i] for i in time_indices]
    dt=np.ones(len(times)) if len(times)==1 else np.r_[ (times[1]-times[0])/2,[(b-a)/2 for a,b in zip(times[:-2],times[2:])],(times[-1]-times[-2])/2 ]
    quadrature=np.broadcast_to(dt.reshape(-1,*([1]*g.dimension))*g.cell_volume,(len(times),*g.shape))
    observed=np.zeros(truth.shape,dtype=bool)
    for index,valid in zip(observations.operator.indices.tolist(),observations.mask[0].tolist()):
        c,t,*space=index
        if valid and t in time_indices:observed[(c,time_indices.index(t),*space)]=True
    fields={}
    for c,field in enumerate(problem.fields):
        fields[field.name]=(mean[c],truth[c],candidates[:,c],observed[c])
    if problem.family=='darcy':
        for value in [mean[1],truth[1],*candidates[:,1]]:
            if np.max(np.abs(value.mean(tuple(range(1,value.ndim)))))>1e-8:raise ValueError('Darcy comparison requires the declared zero-mean pressure gauge; no truth-based alignment is applied')
        if np.any(candidates[:,0]<=0) or np.any(truth[0]<=0):raise ValueError('Log-permeability comparison requires positive physical permeability')
        fields['log_a']=(np.einsum('p,p...->...',w,np.log(candidates[:,0])),np.log(truth[0]),np.log(candidates[:,0]),observed[0])
    report={}
    def selected(error):
        return {k:v for k,v in error.items() if k not in {'mse','rmse','mae','relative_l2'} or k in spec.metrics}
    for name,(estimate,exact,population,mask) in fields.items():
        regions={'full':np.ones_like(mask),'unobserved':~mask,'observed':mask}
        errors={region:selected(_errors(estimate,exact,quadrature,selection,spec.relative_norm_tolerance)) for region,selection in regions.items()}
        per_sample=[{region:selected(_errors(p,exact,quadrature,selection,spec.relative_norm_tolerance)) for region,selection in regions.items()} for p in population]
        average={region:{key:float(sum(weight*error[region][key] for weight,error in zip(w,per_sample))) if all(error[region][key] is not None for error in per_sample) else None for key in spec.metrics} for region in regions}
        per_time=[{'physical_time':t if g.temporal_kind=='transient' else None,**{region:selected(_errors(estimate[i],exact[i],np.full(g.shape,g.cell_volume),selection[i],spec.relative_norm_tolerance)) for region,selection in regions.items()}} for i,t in enumerate(times)]
        variance=np.einsum('p,pt...->t...',w,(population-estimate)**2)
        report[name]={'units':'log(nondimensional permeability)' if name=='log_a' else next(f.units for f in problem.fields if f.name==name),'ensemble_mean':errors,'per_sample':per_sample,'weighted_mean_of_sample_errors':average,'per_time':per_time,'empirical_rms_spread':float(np.sqrt((quadrature*variance).sum()/quadrature.sum()))}
    return {'schema_version':1,'fields':report,'population_weights':w.tolist(),'comparison':transfer,'relative_norm_rule':f'Relative L2 is null if reference RMS <= {spec.relative_norm_tolerance}; absolute metrics remain defined','metric_selection':list(spec.metrics),
        'observation_exclusion':'Exact on-grid point indices excluded, independently per field and saved time; no interpolation','pressure_gauge':'declared zero spatial mean; no truth alignment' if problem.family=='darcy' else None,
        'summary_rule':'Error of weighted ensemble mean is separate from weighted mean of individual errors. log_a uses mean(log a); no cross-field score. Spread is not calibrated uncertainty.'}
