"""Reference-only periodic heat regions, differentiable reconstruction losses and metrics.

Detection is CPU/NumPy, detached from predictions. Training reductions stay on the
model device in FP32. Regions are per physical time, not persistent source IDs.
"""
from dataclasses import dataclass, asdict
from collections import OrderedDict
import math
import numpy as np
import torch


@dataclass(frozen=True)
class FieldLoss:
    type: str = 'mse'
    alpha: float = .5
    contrast_strength: float = 2.
    relative_threshold: float = .25
    absolute_threshold: float = .01  # units of GLOBAL training normalization scale
    numerical_relative_floor: float = 1e-6
    minimum_area_fraction: float = .003
    smoothing_fraction: float = 0.  # sigma / shortest domain length, detection only
    analysis_resolution: int = 32
    time_groups: int = 4
    region_queries: int = 256  # additional TOTAL per solution, NOT per region/time
    estimator: str = 'grouped_stratified_v1'

    def validate(self):
        if self.type not in ('mse','weighted_mse','region_balanced_mse','mae'):raise ValueError('Unknown field reconstruction loss')
        if self.estimator not in ('legacy_uniform_points','grouped_stratified_v1'):raise ValueError('Unknown field-query estimator')
        if self.estimator=='legacy_uniform_points' and self.type!='mse':raise ValueError('Non-MSE losses require grouped_stratified_v1')
        for key in ('alpha','contrast_strength','relative_threshold','absolute_threshold','numerical_relative_floor','minimum_area_fraction','smoothing_fraction'):
            if type(getattr(self,key)) not in (int,float) or not math.isfinite(getattr(self,key)):raise ValueError('Nonfinite field loss setting: '+key)
        if not 0<=self.alpha<=1 or not 0<=self.contrast_strength<=100 or not 0<self.relative_threshold<=1 or self.absolute_threshold<0 or not 0<self.numerical_relative_floor<=.01:raise ValueError('Invalid field loss alpha/contrast/threshold')
        if not 0<self.minimum_area_fraction<=.5 or not 0<=self.smoothing_fraction<=.25:raise ValueError('Region minimum area/smoothing must be physical domain fractions')
        if type(self.analysis_resolution)!=int or not 8<=self.analysis_resolution<=128:raise ValueError('Region analysis resolution must be8..128')
        if type(self.time_groups)!=int or not 1<=self.time_groups<=16 or type(self.region_queries)!=int or not 1<=self.region_queries<=8192:raise ValueError('Invalid bounded time-group/region-query budget')
        return self


def settings(raw=None):
    if raw is not None and not isinstance(raw,dict):raise ValueError('Field reconstruction loss settings must be an object')
    if raw and set(raw)-FieldLoss.__dataclass_fields__.keys():raise ValueError('Unknown field reconstruction loss settings: '+', '.join(sorted(set(raw)-FieldLoss.__dataclass_fields__.keys())))
    # ABSENT settings mean the exact historical MSE/query path, including RNG order.
    return FieldLoss(**({'estimator':'legacy_uniform_points'} if not raw else raw)).validate()


def recommended():return asdict(FieldLoss(type='region_balanced_mse'))


def periodic_components(mask):
    """Four-neighbour torus components, including transitive corner wrapping."""
    mask=np.asarray(mask,dtype=bool);labels=np.zeros(mask.shape,dtype=np.int32);identifier=0
    nx,ny=mask.shape
    for x,y in np.argwhere(mask):
        if labels[x,y]:continue
        identifier+=1;labels[x,y]=identifier;pending=[(x,y)]
        while pending:
            i,j=pending.pop()
            for a,b in (((i-1)%nx,j),((i+1)%nx,j),(i,(j-1)%ny),(i,(j+1)%ny)):
                if mask[a,b] and not labels[a,b]:labels[a,b]=identifier;pending.append((a,b))
    return labels


def analyze(reference,lengths,raw=None,normalization_scale=1.):
    """Labels/weights on uniform periodic endpoint-excluded analysis nodes."""
    cfg=settings(raw);u=np.asarray(reference,dtype=np.float64)
    if u.ndim!=2 or min(u.shape)<2 or not np.isfinite(u).all():raise ValueError('Field loss analysis requires a finite scalar2D periodic field')
    if len(lengths)!=2 or not np.isfinite(lengths).all() or min(lengths)<=0 or not math.isfinite(normalization_scale) or normalization_scale<=0:raise ValueError('Invalid physical domain/normalization scale')
    n=cfg.analysis_resolution
    if u.shape!=(n,n):
        from pdeflow.heat_continuous import FourierHeat
        u=FourierHeat(u,tuple(lengths),(0.,0.)).frame(0.,1.,(0.,1.),shape=(n,n))
    detection=u
    if cfg.smoothing_fraction:
        sigma=cfg.smoothing_fraction*min(lengths)
        k=[2*np.pi*np.fft.fftfreq(n,d=L/n) for L in lengths]
        detection=np.fft.ifftn(np.fft.fftn(u)*np.exp(-.5*sigma*sigma*(k[0][:,None]**2+k[1][None,:]**2))).real
    contrast=detection-detection.mean();peak=float(np.abs(contrast).max())
    floor=max(cfg.absolute_threshold*normalization_scale,cfg.numerical_relative_floor*max(float(np.abs(detection).max()),normalization_scale))
    threshold=max(floor,cfg.relative_threshold*peak)
    labels=np.zeros(u.shape,dtype=np.int32);regions=[]
    if peak>floor:
        for sign in (1,-1):
            components=periodic_components(sign*contrast>=threshold)
            for k in np.unique(components):
                if not k:continue
                mask=components==k;fraction=float(mask.mean())
                if fraction<cfg.minimum_area_fraction:continue
                identifier=len(regions)+1;labels[mask]=identifier
                regions.append({'label':identifier,'sign':'hot' if sign==1 else 'cold','area_fraction':fraction,'physical_area':fraction*float(np.prod(lengths))})
    # Contrast weighting uses UNSMOOTHED reference; smoothing affects region detection only.
    a=np.abs(u-u.mean());maximum=float(a.max())
    w=1+cfg.contrast_strength*a/(maximum+max(floor,np.finfo(float).tiny)) if maximum>floor else np.ones_like(u)
    k=len(regions);contribution=np.ones_like(u)
    if k:
        contribution.fill(1-cfg.alpha)
        for r in regions:contribution[labels==r['label']]+=cfg.alpha/(k*r['area_fraction'])
    return {'labels':labels,'reference':u,'contrast_weights':w,'contribution_weights':contribution,'regions':regions,'region_count':k,
            'threshold_physical':threshold,'contrast_floor_physical':floor,'analysis_shape':list(u.shape),'connectivity':'four-neighbour periodic; hot/cold separate','cap_activations':0}


def project(array,shape):
    """Nearest periodic analysis-cell value at evaluation nodes i/N (Voronoi cells)."""
    ix=[np.floor(np.arange(m)*n/m+.5).astype(int)%n for m,n in zip(shape,array.shape)]
    return array[np.ix_(*ix)]


def field_metrics(prediction,reference,lengths,raw=None,normalization_scale=1.):
    cfg=settings(raw);a=np.asarray(prediction,dtype=np.float64);b=np.asarray(reference,dtype=np.float64)
    if a.shape!=b.shape or b.ndim!=2 or not np.isfinite(a).all():raise ValueError('Field metrics require identical finite physical coordinate grids')
    analysis=analyze(b,lengths,raw,normalization_scale);labels=project(analysis['labels'],b.shape);w=project(analysis['contrast_weights'],b.shape)
    e=a-b;sq=e**2;mse=float(sq.mean());mae=float(np.abs(e).mean());rms=float(np.sqrt(np.mean(b*b)))
    regions=[]
    # A coarse evaluation grid must not silently drop analysis regions.
    missing=[r for r in analysis['regions'] if not np.any(labels==r['label'])]
    for r in analysis['regions']:
        mask=labels==r['label']
        regions.append({**r,'mse':float(sq[mask].mean()) if mask.any() else None,'evaluation_nodes':int(mask.sum())})
    region_mean=float(np.mean([r['mse'] for r in regions])) if regions and not missing else None
    rb=(1-cfg.alpha)*mse+cfg.alpha*region_mean if region_mean is not None else mse if not regions else None
    bg=labels==0
    return {'mse':mse,'rmse':math.sqrt(mse),'mae':mae,'relative_l2':math.sqrt(mse)/rms if rms>1e-10 else None,
            'weighted_mse':float((sq*w).mean()/w.mean()),'region_balanced_mse':rb,'region_count':len(regions),'regions':regions,
            'mean_region_mse':region_mean,'worst_region_mse':max((r['mse'] for r in regions),default=None) if not missing else None,
            'background_mse':float(sq[bg].mean()) if bg.any() else None,'unrepresented_region_count':len(missing),
            'region_metric_unavailable_reason':'Evaluation grid misses analysis regions; increase evaluation resolution' if missing else None,
            'relative_l2_floor':'Unavailable for reference RMS<=1e-10 in physical units','units':'physical fields','settings':asdict(cfg),'normalization_scale_for_thresholds':float(normalization_scale),'threshold_physical':analysis['threshold_physical'],'contrast_floor_physical':analysis['contrast_floor_physical'],
            'labels':labels,'analysis':analysis}


def metric_record(*args,**kwargs):
    return {k:v for k,v in field_metrics(*args,**kwargs).items() if k not in ('labels','analysis')}


def differentiable_loss(prediction,target,weights,kind='mse'):
    # Cast BEFORE subtraction/reduction: safe under mixed precision, gradients retained.
    error=prediction.float()-target.detach().float();w=weights.detach().to(error.device,dtype=torch.float32)
    if error.shape!=w.shape or error.shape!=target.shape or not torch.isfinite(w).all() or torch.any(w<0):raise ValueError('Invalid field loss query weights/layout')
    value=error.abs() if kind=='mae' else error.square()
    result=(value*w).sum(1).mean()
    if not torch.isfinite(result):raise FloatingPointError('Nonfinite field reconstruction loss')
    return result


class AnalysisCache:
    def __init__(self,limit=32):self.values=OrderedDict();self.limit=limit
    def get(self,key,make):
        if key not in self.values:
            self.values[key]=make()
            if len(self.values)>self.limit:self.values.popitem(last=False)
        self.values.move_to_end(key);return self.values[key]


def grouped_queries(c,items,g,normalization,cache):
    """Uniform-domain + region-stratified queries; each example/time reduced equally.

    Region cells are uniform-area periodic Voronoi cells around analysis nodes.
    Jitter is uniform inside cells. Every retained region has >=1 query or we fail.
    Query budget is domain query_points + fixed region_queries (RB only).
    """
    from pdeflow.heat_continuous import FourierHeat
    from pdeflow.coordinate.temporal import sample_times
    cfg=settings(c.field_loss);groups=cfg.time_groups if c.temporal_mode=='continuous' else len(c.times)
    if c.query_points<groups:raise ValueError('Domain queries must cover every supervised physical time group')
    extra=cfg.region_queries if cfg.type=='region_balanced_mse' else 0
    queries=[];targets=[];all_weights=[];stats=[]
    seed=int(torch.randint(2**31,(),generator=g))
    for b,item in enumerate(items):
        g=torch.Generator().manual_seed(seed+b*100003)
        region_rng=torch.Generator().manual_seed(seed+b*100003+731)
        solver=item.get('solver') or FourierHeat(np.asarray(item['values']).reshape(len(c.times),*c.reference_shape)[0],c.lengths,c.origins)
        times=sample_times(c,groups,g,'query').numpy() if c.temporal_mode=='continuous' else np.asarray(c.times)
        analyses=[cache.get((item['id'],float(t),str(asdict(cfg)),normalization['scale']),lambda t=t:analyze(solver.frame(float(t),item['D'],c.interval,shape=(cfg.analysis_resolution,)*2),c.lengths,c.field_loss,normalization['scale'])) for t in times]
        counts=[len(a['regions']) for a in analyses];total=sum(counts)
        if extra and total>extra:raise ValueError(f'{total} retained time-regions exceed region_queries={extra}; increase the explicit fixed region query budget or adjust documented detection settings. No region was dropped.')
        region_allocation=[]
        if extra and total:region_allocation=[extra//total+(i<extra%total) for i in range(total)]
        coords=[];weights=[];cursor=0
        for j,(t,a) in enumerate(zip(times,analyses)):
            count=c.query_points//groups+(j<c.query_points%groups)
            xy=torch.rand(count,2,generator=g,dtype=torch.float64).numpy()*np.asarray(c.lengths)+np.asarray(c.origins)
            coords.extend(np.c_[xy,np.full(count,t)])
            domain_weight=np.full(count,1/(groups*count))
            if cfg.type=='weighted_mse':
                cell=np.floor((xy-c.origins)/c.lengths*cfg.analysis_resolution+.5).astype(int)%cfg.analysis_resolution
                domain_weight*=a['contrast_weights'][cell[:,0],cell[:,1]]/a['contrast_weights'].mean()
            elif cfg.type=='region_balanced_mse' and counts[j]:domain_weight*=1-cfg.alpha
            weights.extend(domain_weight)
            if extra and counts[j]:
                for r in a['regions']:
                    n=region_allocation[cursor];cursor+=1
                    cells=np.argwhere(a['labels']==r['label'])
                    ids=torch.randint(len(cells),(n,),generator=region_rng).numpy()
                    jitter=torch.rand(n,2,generator=region_rng,dtype=torch.float64).numpy()-.5
                    xy=((cells[ids]+jitter)/cfg.analysis_resolution)%1*np.asarray(c.lengths)+np.asarray(c.origins)
                    coords.extend(np.c_[xy,np.full(n,t)]);weights.extend([cfg.alpha/(groups*counts[j]*n)]*n)
        used=len(coords)
        # No meaningful regions: ordinary MSE, remaining fixed slots masked out.
        if used<c.query_points+extra:
            coords.extend([coords[0]]*(c.query_points+extra-used));weights.extend([0.]*(c.query_points+extra-used))
        q=np.asarray(coords);queries.append(q);targets.append(solver.evaluate(q,item['D'],c.interval));all_weights.append(weights)
        stats.append({'regions_per_time':counts,'physical_times':times.tolist(),'domain_queries':c.query_points,'additional_region_queries':extra,'active_queries':used,'decoded_query_budget':len(q),'cap_activations':0})
    return torch.tensor(np.asarray(queries),dtype=torch.float32),torch.tensor(np.asarray(targets),dtype=torch.float32),torch.tensor(np.asarray(all_weights),dtype=torch.float32),stats


def trajectory_metrics(prediction,reference,times,lengths,raw=None,normalization_scale=1.):
    """Uniform physical-time trapezoidal quadrature on recorded (possibly uneven) times."""
    a=np.asarray(prediction);b=np.asarray(reference)
    if a.shape!=b.shape or a.ndim!=3 or len(times)!=len(a):raise ValueError('Trajectory metric coordinates/times differ')
    rows=[{'time':float(t),**metric_record(p,r,lengths,raw,normalization_scale)} for t,p,r in zip(times,a,b)]
    times=np.asarray(times);w=np.ones(1) if len(times)==1 else np.r_[np.diff(times)[0]/2,(times[2:]-times[:-2])/2,np.diff(times)[-1]/2]/(times[-1]-times[0])
    result={}
    for key in ('mse','mae','weighted_mse','region_balanced_mse','mean_region_mse','background_mse'):
        values=[r[key] for r in rows]
        result[key]=float(np.dot(w,values)) if all(v is not None for v in values) else None
    result['rmse']=math.sqrt(result['mse']);den=float(np.dot(w,np.mean(b*b,axis=(1,2))))
    result['reference_mse']=den;result['relative_l2']=math.sqrt(result['mse']/den) if den>1e-20 else None
    return {'uniform_physical':result,'per_time':rows,'aggregation':'Per solution, periodic spatial node quadrature; normalized trapezoidal physical-time weights on listed times. RMSE=sqrt(aggregated MSE), relative L2 uses aggregated reference squared norm. Region mean is equal-region within each time. No pooling of regions across times/solutions.'}


def aggregate_metrics(reports):
    rows=[r['uniform_physical'] for r in reports];out={}
    for key in ('mse','mae','weighted_mse','region_balanced_mse','mean_region_mse','background_mse'):
        out[key]=float(np.mean([r[key] for r in rows])) if all(r[key] is not None for r in rows) else None
    out['rmse']=math.sqrt(out['mse']);den=float(np.mean([r['reference_mse'] for r in rows]));out['relative_l2']=math.sqrt(out['mse']/den) if den>1e-20 else None
    return {**out,'aggregation':'Equal solutions after physical-time quadrature; RMSE=sqrt(mean solution MSE), NOT mean solution RMSE. Relative L2 uses pooled equal-solution squared reference norm.'}
