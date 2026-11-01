"""Physical-time distributions, separate from flow time and display frames."""
import numpy as np
import torch

DEFAULTS={
 'encoder':{'initial':.35,'early':.35,'uniform':.25,'final':.05},
 'query':{'initial':.25,'early':.35,'uniform':.35,'final':.05},
 'observation':{'initial':.15,'early':.35,'uniform':.45,'final':.05},
}


def policies(c):
    value={'early_min':1e-4,'early_max':.2,**c.time_sampling}
    if set(value)-{'early_min','early_max',*DEFAULTS}:raise ValueError('Unknown continuous time-sampling setting')
    if not np.isfinite([value['early_min'],value['early_max']]).all() or not 0<value['early_min']<value['early_max']<=1:raise ValueError('Early time fractions require 0 < early_min < early_max <= 1')
    for role,default in DEFAULTS.items():
        p={**default,**value.get(role,{})}
        if set(p)!=set(default) or not all(np.isfinite(v) and v>=0 for v in p.values()) or not np.isclose(sum(p.values()),1,rtol=0,atol=1e-10):raise ValueError('Time-mixture probabilities must be finite, nonnegative and sum to one: '+role)
        if p['uniform']<=0 or p['initial']<=0:raise ValueError('Continuous training must cover the full interval and explicit initial time')
        value[role]=p
    return value


def sample_times(c,n,g,role='query'):
    p=policies(c);weights=np.asarray(list(p[role].values()));counts=np.floor(n*weights).astype(int)
    if n<8:counts=torch.bincount(torch.multinomial(torch.tensor(weights),n,replacement=True,generator=g),minlength=4).numpy() if n else np.zeros(4,dtype=int)
    else:
        for i in np.argsort(-(n*weights-counts),kind='stable')[:n-int(counts.sum())]:counts[i]+=1
    chunks=[];lo,hi=c.time_interval;span=hi-lo
    for key,count in zip(p[role],counts):
        r=torch.rand(int(count),generator=g,dtype=torch.float64)
        if key=='initial':r=torch.zeros_like(r)
        elif key=='final':r=torch.ones_like(r)
        elif key=='early':r=torch.exp(np.log(p['early_min'])+r*np.log(p['early_max']/p['early_min']))
        chunks.append(lo+span*r)
    t=torch.cat(chunks);return t[torch.randperm(n,generator=g)]


def points(c,n,g,role='query'):
    xy=torch.rand(n,c.dimension,generator=g,dtype=torch.float64)*torch.tensor(c.lengths)+torch.tensor(c.origins)
    return torch.cat([xy,sample_times(c,n,g,role)[:,None]],1).float()


def capability(c):
    return {'mode':c.temporal_mode,'interval':list(c.interval),'trained_continuous':c.temporal_mode=='continuous',
            'snapshot_times':list(c.times) if c.temporal_mode!='continuous' else None,
            'validated_temporal_accuracy':False,'meaning':'Training-support declaration, not demonstrated temporal interpolation accuracy'}


def state_time(c,iteration,g,role='query',condition_nonzero=False):
    """Recorded initial/early/uniform/final mixture; uniform part uses32 jittered strata."""
    policy=policies(c);weights=policy[role]
    if condition_nonzero:
        total=sum(v for k,v in weights.items() if k!='initial');weights={k:(v/total if k!='initial' else 0.) for k,v in weights.items()}
    kind=list(weights)[int(torch.multinomial(torch.tensor(list(weights.values())),1,generator=g))];r=float(torch.rand((),generator=g))
    fraction=0. if kind=='initial' else 1. if kind=='final' else float(np.exp(np.log(policy['early_min'])+r*np.log(policy['early_max']/policy['early_min']))) if kind=='early' else (iteration%32+r)/32
    if condition_nonzero:fraction=max(fraction,torch.finfo(torch.float32).eps)
    return c.interval[0]+fraction*(c.interval[1]-c.interval[0])


def evolution_time(c,iteration,g):
    """Opt-in NEW A2 experiment; default preserves the historical RNG path."""
    return state_time(c,iteration,g,'query',c.evolution.get('time_policy','legacy_query_v1')=='nonzero_conditioned_v1')
