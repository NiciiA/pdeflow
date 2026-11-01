"""Known measurement times, independent of evolution targets and flow time."""
import numpy as np


def settings(value=None):
    p={'policy':'continuous_uniform','stratified':True,**(value or {})}
    if p['policy']=='legacy_mixture_v1':
        if set(p)-{'policy','stratified'}:raise ValueError('Legacy observation mixture uses recorded time_sampling.observation')
        return {'policy':'legacy_mixture_v1'}
    if p['policy'] not in ('continuous_uniform','continuous_mixture'):raise ValueError('Unknown observation-time policy')
    allowed={'policy','stratified','whole','early','initial','early_min','early_max','placement_groups'}
    if set(p)-allowed:raise ValueError('Unknown observation-time settings')
    if type(p['stratified'])!=bool:raise ValueError('stratified must be boolean')
    p={'whole':1.,'early':0.,'initial':0.,'early_min':1e-4,'early_max':.2,'placement_groups':8,**p}
    if p['policy']=='continuous_uniform' and (p['whole'],p['early'],p['initial'])!=(1.,0.,0.):raise ValueError('Whole-interval policy has no early/initial weighting; select continuous_mixture')
    w=np.array([p[k] for k in ('whole','early','initial')])
    if not np.isfinite(w).all() or np.any(w<0) or p['whole']<=0 or not np.isclose(w.sum(),1,rtol=0,atol=1e-10):raise ValueError('Time mixture must sum to one with a positive whole-interval component')
    if not np.isfinite([p['early_min'],p['early_max']]).all() or not 0<p['early_min']<p['early_max']<=1:raise ValueError('Early fractions require 0 < min < max <= 1')
    if type(p['placement_groups'])!=int or not 1<=p['placement_groups']<=16:raise ValueError('Placement analysis groups must be 1..16')
    return p


def continuous(n,interval,rng,value=None):
    p=settings(value);lo,hi=map(float,interval)
    if p['policy']=='legacy_mixture_v1':raise ValueError('Legacy sampling requires the versioned training sampler')
    if n<0 or not np.isfinite([lo,hi]).all() or not lo<hi:raise ValueError('Invalid observation count or physical interval')
    if p['initial']>0 and lo!=0:raise ValueError('Exact t=0 mixture component requires a measurement interval starting at zero')
    if p['policy']=='continuous_uniform':counts=[n,0,0]
    else:
        # Multinomial allocation preserves mixture probabilities even for N=1.
        counts=rng.multinomial(n,[p[k] for k in ('whole','early','initial')])
    chunks=[]
    for kind,count in zip(('whole','early','initial'),counts):
        r=(np.arange(count)+rng.random(count))/max(1,count) if p['stratified'] else rng.random(count)
        if kind=='early':r=np.exp(np.log(p['early_min'])+r*np.log(p['early_max']/p['early_min']))
        elif kind=='initial':r=np.zeros(count)
        chunks.append(lo+(hi-lo)*r)
    return rng.permutation(np.concatenate(chunks))


def explicit(times,total,interval,counts=None):
    t=np.asarray(times,dtype=float);lo,hi=interval
    if t.ndim!=1 or not len(t) or not np.isfinite(t).all() or np.any(t<lo) or np.any(t>hi) or len(set(t))!=len(t):raise ValueError('Provide distinct finite measurement times inside reference support')
    if counts:
        if len(counts)!=len(t) or any(type(v)!=int or v<0 for v in counts):raise ValueError('Provide one nonnegative integer count per time')
        if sum(counts)!=total:raise ValueError('Total measurement count must equal the sum of per-time counts')
        return np.repeat(t,counts)
    return np.resize(t,total)  # TOTAL, with at most one extra per entered time.


def focused_positions(times,interval,groups,place_at):
    """Bounded reference-only placement approximation; target times stay exact.

    Equal-width temporal bins, one map at the mean actual measurement time in
    each occupied bin. No sensor timestamp is rounded. Each group retains the
    spatial sampler's background coverage. Groups are not inference features.
    """
    times=np.asarray(times);lo,hi=interval;bins=np.minimum(groups-1,((times-lo)/(hi-lo)*groups).astype(int))
    xy=np.empty((len(times),2));reports=[]
    for k in np.unique(bins):
        ix=np.flatnonzero(bins==k);at=float(times[ix].mean());positions,report=place_at(len(ix),at,int(k))
        xy[ix]=positions;reports.append({'group':int(k),'count':len(ix),'analysis_time':at,'sensor_time_min':float(times[ix].min()),'sensor_time_max':float(times[ix].max()),'placement':report})
    return xy,{'reference_assisted':True,'temporal_grouping':'equal-width bins; map at mean actual sensor time; values evaluated at exact sensor times','max_groups':groups,'time_designs':reports,'message':'Feature placement uses bounded time-group maps; measurement values use exact timestamps.'}


def summary(times,interval):
    times=np.asarray(times);counts,edges=np.histogram(times,bins=8,range=interval)
    return {'count':len(times),'minimum':float(times.min()) if len(times) else None,'maximum':float(times.max()) if len(times) else None,'histogram':{'counts':counts.tolist(),'edges':edges.tolist()}}
