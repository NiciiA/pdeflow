"""Reference-free component reports on physical fields; no policy gradients here."""
from dataclasses import asdict, dataclass
import math
import torch
from pdeflow.core import Conditioning
from pdeflow.specialists import get_specialist


@dataclass(frozen=True)
class Diagnostic:
    value: float | None
    units: str
    reduction: str
    component: str
    count: int
    threshold: float | None = None
    applicable: bool = True
    note: str = ''

    def to_dict(self):
        d=asdict(self)
        d['finite']=self.value is not None and math.isfinite(self.value)
        if not d['finite']:d['value']=None
        d['status']='not_applicable' if not self.applicable else 'nonfinite' if not d['finite'] else 'unconfigured' if self.threshold is None else 'passed' if self.value<=self.threshold else 'violated'
        return d


def report(physical, problem, observations, label='final_field', thresholds=None):
    """Worst candidate RMSE per component, physical quadrature; never truth."""
    thresholds=thresholds or {};grid=problem.grid
    expected=(len(problem.fields),len(grid.times),*grid.shape)
    if tuple(physical.shape[1:])!=expected:raise ValueError('Diagnostic physical field/grid mismatch')
    entries={};mandatory=[]
    def add(name,value,units,reduction,component,count,required=True,applicable=True,note='',exact_limit=None):
        limit=thresholds.get(name,exact_limit)
        entries[name]=Diagnostic(value,units,reduction,component,count,limit,applicable,note).to_dict()
        if required and applicable:mandatory.append(name)
    finite=bool(torch.isfinite(physical).all())
    add('field.max_abs',float(physical.abs().max()),'physical field units','maximum over candidates/components/grid','all',physical.numel(),False)
    predictions=observations.operator.apply(physical)
    y=observations.values.to(predictions);mask=observations.mask.to(predictions.device)
    if len(y)!=1:raise ValueError('Process diagnostics reconstruct one physical problem')
    residual=predictions-y
    scales=observations.scales(predictions)
    fields=observations.operator.indices[:,0].to(predictions.device)
    for c,f in enumerate(problem.fields):
        selected=mask[0] & (fields==c);count=int(selected.sum())
        for kind,scaled in [('rmse',residual),('standardized_rmse',residual/scales)]:
            value=float(scaled[:,selected].square().mean(1).sqrt().max()) if count else None
            add(f'observation.{f.name}.{kind}',value,'dimensionless tempered' if kind.startswith('standard') else f.units,'worst candidate; equal weight measured rows',f.name,count,kind.startswith('standard') and count>0,count>0)
        add(f'observation.{f.name}.max_abs',float(residual[:,selected].abs().max()) if count else None,f.units,'maximum measured residual',f.name,count,False,count>0)
    if finite:
        specialist=get_specialist(problem.family)
        physics=specialist.physics(problem,Conditioning.from_problem(problem,len(physical),device=physical.device,dtype=physical.dtype)).evaluate(physical)
        r=physics.pointwise['pde'];weights=physics.time_weights.reshape(1,-1,*([1]*grid.dimension))
        components=['u'] if problem.family=='darcy' else [f.name for f in problem.fields]
        for c,name in enumerate(components):
            w=weights.expand_as(r[:,c]);axes=tuple(range(1,w.ndim))
            value=float(((r[:,c].square()*w).sum(axes)/w.sum(axes)).sqrt().max())
            add(f'pde.{name}.rmse',value,physics.units['pde'],'worst candidate; cell volume and trapezoidal time normalized; interior times only for transient',name,r[:,c].numel())
        extras=specialist.extra_metrics(physical,problem)
        if problem.family=='heat':
            integrals=physical.sum(tuple(range(3,physical.ndim)))*grid.cell_volume
            add('conservation.u.max_drift',float((integrals-integrals[:,:,:1]).abs().max()),'field × spatial volume','maximum change from first saved time','u',integrals.numel())
        for key,name in [('invariant_a_plus_c_max_drift','a_plus_c'),('invariant_b_plus_c_max_drift','b_plus_c')]:
            if key in extras:add(f'conservation.{name}.max_drift',extras[key],'concentration × spatial volume','maximum change from first saved time',name,len(physical)*len(grid.times))
        if problem.family=='darcy':add('gauge.u.abs_mean',extras['pressure_gauge_max_abs_mean'],'pressure','worst candidate spatial mean','u',len(physical))
        violations=extras.get('positivity_violations',0)
        if problem.family!='heat':add('admissibility.violations',float(violations),'count','all candidates; a>0 for Darcy, species>=0 for chemistry','physical',physical.numel(),exact_limit=0.)
    add('boundary.periodic_residual',None,'not independently evaluated','periodic topology imposed by representation','boundary',0,False,False,'No independent wall/seam error; seam variation is not a boundary violation')
    add('initial.known_residual',None,'not applicable','initial field unknown; explicit information is observations','initial',0,False,False)
    missing=set(thresholds)-set(entries)
    if missing and finite:raise ValueError(f'Unknown diagnostic thresholds: {sorted(missing)}')
    failed=[k for k in mandatory if entries[k]['status']!='passed']
    # Optional configured guards are also mandatory; a low average cannot hide them.
    failed+= [k for k in thresholds if k not in mandatory and entries[k]['status']!='passed']
    count=int(mask.sum());safe=lambda x: [[float(v) if math.isfinite(float(v)) else None for v in row] for row in x.detach().cpu()]
    noise=observations.field_noise_std or (observations.noise_std,)*len(problem.fields)
    standardized=residual.clone()
    for c,std in enumerate(noise):standardized[:,fields==c]=residual[:,fields==c]/std if std>0 else float('nan')
    if observations.measurement_noise_std is not None:
        stds=residual.new_tensor(observations.measurement_noise_std)
        standardized=torch.where(stds[None]>0,residual/stds.clamp_min(torch.finfo(residual.dtype).tiny),float('nan'))
    raw=residual.masked_fill(~mask,float('nan'));standardized=standardized.masked_fill(~mask,float('nan'))
    return {'schema_version':1,'label':label,'grid':asdict(grid),'fields':[f.name for f in problem.fields],'entries':entries,'mandatory':mandatory,'mandatory_failures':failed,'criteria_met':finite and bool(mandatory) and not failed,'finite':finite and all(e['finite'] for e in entries.values() if e['applicable']), 'observation':{'operator':'on_grid_point','raw_residuals':safe(raw),'noise_standardized_residuals':safe(standardized),'noise_note':'Independent Gaussian; null for missing/noiseless. Policy uses max(noise,tolerance) tempered residual, not zero variance.','count':count,'missing_count':int(mask.numel()-count),'energy_reduction':'half mean squared tempered residual per candidate','spatial_interpolation':'unsupported; exact nodes required','physical_time_interpolation':'unsupported; exact saved times required'},'provenance':{'reference_used':False,'physics':'specialist physical decode, finite difference/flux; same steering discretization','independent_continuum_certificate':False}}
