"""Deterministic shared heat/wave realization splits and bounded references."""
from pathlib import Path
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import time
import numpy as np
import torch
from pdeflow.runtime import digest,read_json,write_json
from pdeflow.numerics.dynamic import SpectralState

FAMILIES=('cosine','fourier_sum','smooth_random','hot_spot','multiple_spots','brush')
WAVE_CASES=('standing','travelling','pulse','velocity_pulse','signed_pulses','random')

def recipe(c):
    d={'families':{f:1. for f in FAMILIES},'max_wave':max(1,min(c.reference_shape)//8),'amplitude':[.4,1.2],'background':[-.5,.5],'width':[max(.08,3/min(c.reference_shape)),max(.18,4/min(c.reference_shape))]}
    if set(c.distribution)-d.keys():raise ValueError('Unknown distribution setting')
    d.update(c.distribution)
    if not d['families'] or set(d['families'])-set(FAMILIES) or any(not np.isfinite(v) or v<=0 for v in d['families'].values()):raise ValueError('Positive supported heat feature-family weights required')
    if min(c.train_count,c.validation_count,c.test_count)<6*len(c.family_weights):raise ValueError('Every split needs six independent realizations per enabled PDE family')
    if not 1<=d['max_wave']<min(c.reference_shape)/2 or d['width'][0]<3/min(c.reference_shape):raise ValueError('Generated spatial frequencies/features must be resolved by the reference grid')
    return {**d,'family_weights':c.family_weights,'wave_cases':WAVE_CASES,'wave_speed_range':c.wave_speed_range,'diffusivity_range':c.diffusivity_range,
            'counts':[c.train_count,c.validation_count,c.test_count],'seed':c.seed,'reference_shape':c.reference_shape,'domain':[c.lengths,c.origins],'time':c.interval,
            'encoder_points':c.encoder_points,'wave_phase_strata':32,'near_constant_policy':'Every seventeenth realization after the twelve-case coverage prefix; independent means plus1e-6 smooth contrast','version':'physical_state_mixture_v2'}

def identity(c):return digest(recipe(c))

def initial_pair(c,case,seed,shape=None):
    rng=np.random.default_rng(seed);shape=tuple(shape or c.reference_shape);x,y=np.meshgrid(np.arange(shape[0])/shape[0],np.arange(shape[1])/shape[1],indexing='ij');r=recipe(c);k=r['max_wave']
    def random_field():
        a=np.full(shape,rng.uniform(*r['background']))
        for j in range(1,k+1):
            for l in range(0,k+1):a+=rng.normal()*rng.uniform(*r['amplitude'])/(k+1)*np.cos(2*np.pi*(j*x+l*y)+rng.uniform(-np.pi,np.pi))
        return a
    def pulse(cx,cy,width):return np.exp(-(np.sin(np.pi*(x-cx))**2+np.sin(np.pi*(y-cy))**2)/(np.pi*width)**2)
    amplitude=rng.uniform(*r['amplitude']);phase=rng.uniform(-np.pi,np.pi);kx=int(rng.integers(1,k+1));ky=int(rng.integers(0,k+1));theta=2*np.pi*(kx*x+ky*y)+phase
    u=random_field();v=random_field()
    if case in ('cosine','standing','travelling'):u=rng.uniform(*r['background'])+amplitude*np.cos(theta);v=np.zeros(shape)
    elif case in ('hot_spot','pulse','velocity_pulse'):u=amplitude*pulse(*rng.random(2),rng.uniform(*r['width']));v=np.zeros(shape)
    elif case in ('multiple_spots','signed_pulses','brush'):
        u=sum(rng.choice([-1.,1.])*amplitude*pulse(*rng.random(2),rng.uniform(*r['width'])) for _ in range(3));v=random_field()
    # Explicit bandlimit after drawing the smooth generated fields: avoids
    # unresolved tails and gives a known highest temporal wave frequency.
    def bandlimit(a):
        modes=np.meshgrid(*[np.fft.fftfreq(n)*n for n in shape],indexing='ij');keep=(np.abs(modes[0])<=k)&(np.abs(modes[1])<=k)
        return np.fft.ifftn(np.fft.fftn(a)*keep).real
    u=bandlimit(u);v=bandlimit(v)
    if case=='velocity_pulse':v=u.copy();u=np.zeros_like(u)
    return u,v,{'kx':kx,'ky':ky,'phase':phase,'amplitude':amplitude,'theta':theta}

def family_for(c,split,index):
    names=sorted(c.family_weights);rng=np.random.default_rng(np.random.SeedSequence([c.seed,('train','validation','test').index(split),index]));weights=np.array([c.family_weights[n] for n in names]);weights/=weights.sum()
    return names[index%len(names)] if index<6*len(names) else str(rng.choice(names,p=weights))

def example(c,split,index,shape=None,dtype=np.float32):
    r=recipe(c);s=('train','validation','test').index(split);names=sorted(c.family_weights);rng=np.random.default_rng(np.random.SeedSequence([c.seed,s,index]));weights=np.array([c.family_weights[n] for n in names]);weights=weights/weights.sum()
    # First six realizations of EACH family guarantee fixed validation coverage.
    family=names[index%len(names)] if index<6*len(names) else str(rng.choice(names,p=weights));case=(FAMILIES if family=='heat' else WAVE_CASES)[(index//len(names))%6]
    if family=='heat' and index>=6*len(names):
        cases=list(r['families']);cw=np.array(list(r['families'].values()));case=str(rng.choice(cases,p=cw/cw.sum()))
    seed=int(np.random.SeedSequence([c.seed,s,index,271]).generate_state(1)[0]);u,v,meta=initial_pair(c,case,seed,shape)
    bounds=c.diffusivity_range if family=='heat' else c.wave_speed_range;parameter=float(rng.uniform(*bounds))
    if case=='travelling':
        omega=parameter*np.sqrt((2*np.pi*meta['kx']/c.lengths[0])**2+(2*np.pi*meta['ky']/c.lengths[1])**2);v=meta['amplitude']*omega*np.sin(meta['theta'])
    if index>=12 and index%17==0:
        u=float(rng.uniform(-.5,.5))+1e-6*(u-u.mean());v=float(rng.uniform(-.2,.2))+1e-6*(v-v.mean());case='approximately_constant'
    solver=SpectralState(u,v if family=='wave' else None,c.lengths,c.origins,family,mode_limit=r['max_wave'],backend=c.training_options.get('performance_mode','optimized'))
    if family=='heat':v=np.zeros_like(u)
    generator=torch.Generator().manual_seed(seed);xy=torch.rand(c.encoder_points,2,generator=generator)*torch.tensor(c.lengths)+torch.tensor(c.origins)
    return {'values':u.astype(dtype).ravel(),'initial':np.stack([u,v],-1).astype(np.float32),'D':parameter,'parameter':parameter,'family':family,'case':case,'id':f'{split}:{index}:{identity(c)[:16]}','seed':[c.seed,s,index],
            'context':xy.numpy(),'solver':solver,'metadata':meta}

class QuantityReference:
    def __init__(self,solver,quantity):self.solver=solver;self.quantity=quantity
    def frame(self,t,p,interval,shape=None):return self.solver.frame(t,p,interval,shape,self.quantity)
    def evaluate(self,q,p,interval):return self.solver.evaluate(q,p,interval,self.quantity)

def item_quantity(item,quantity):return {**item,'id':item['id']+':'+quantity,'solver':QuantityReference(item['solver'],quantity)}

def state_samples(item,xy,t,c,norm=None):
    q=np.c_[np.asarray(xy),np.full(len(xy),float(t))]
    if item['solver'].backend=='legacy':
        u=item['solver'].evaluate(q,item['parameter'],c.interval,'u');v=item['solver'].evaluate(q,item['parameter'],c.interval,'v') if item['family']=='wave' else np.zeros(len(q));values=np.stack([u,v],-1)
    else:
        values=item['solver'].evaluate_state(q,item['parameter'],c.interval,('u','v') if item['family']=='wave' else ('u',))
        if item['family']=='heat':values=np.c_[values[:,0],np.zeros(len(q))]
    if norm is not None:
        for j,n in enumerate(norm):values[:,j]=(values[:,j]-n['mean'])/n['scale']
        if item['family']=='heat':values[:,1]=0
    return torch.tensor(values,dtype=torch.float32)

class Dataset:
    def __init__(self,c,root,create=True):
        self.c=c;self._items=OrderedDict();self.cache_limit=c.training_options.get('reference_cache_items',32);self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);path=self.root/'dataset.json';prefix=identity(c)[:16];ids=[f'{s}:{i}:{prefix}' for s,n in zip(('train','validation','test'),(c.train_count,c.validation_count,c.test_count)) for i in range(n)]
        # IDs do not require solving every realization.
        if path.exists():
            self.metadata=read_json(path)
            if self.metadata['sha256']!=identity(c):raise ValueError('Dataset mixture/recipe identity changed')
            return
        if not create:raise ValueError('Missing representation dataset recipe; restore the run')
        norms={};fit={f:[] for f in c.family_weights};samples={f:[[],[]] for f in c.family_weights};started=time.perf_counter()
        for i in range(min(c.train_count,max(12,c.training_options.get('calibration_count',12)))):
            item=self.get('train',i);family=item['family'];fit[family].append(item['id'])
            states=state_samples_batch([item]*4,(0.,.137,.4,.8),c).numpy()
            for state in states:
                for j in range(2 if family=='wave' else 1):samples[family][j].append(state[:,j])
        for family,parts in samples.items():
            norms[family]=[{'mean':float(np.concatenate(a).mean()),'scale':max(float(np.concatenate(a).std()),1e-3)} if a else {'mean':0.,'scale':1.} for a in parts]
        self.metadata={'schema_version':3,'kind':'coordinate_training_data','sha256':identity(c),'recipe':recipe(c),'ids':ids,'normalizations':norms,'fitted_ids':fit,'normalization':{**norms.get('heat',norms.get('wave'))[0],'fitted_ids':sum(fit.values(),[])},'preparation_seconds':time.perf_counter()-started}
        write_json(path,self.metadata)
    def get(self,split,index):
        key=(split,int(index))
        if key not in self._items:
            self._items[key]=example(self.c,split,int(index))
            if len(self._items)>self.cache_limit:self._items.popitem(last=False)
        self._items.move_to_end(key);return self._items[key]
    def items(self,indices,split='train',workers=0):
        if workers:
            with ThreadPoolExecutor(max_workers=workers) as pool:items=list(pool.map(lambda i:example(self.c,split,int(i)),indices))
        else:items=[self.get(split,int(i)) for i in indices]
        return items
    def batch(self,indices,split='train',workers=0):
        items=self.items(indices,split,workers)
        return torch.tensor(np.stack([x['values'] for x in items])),torch.tensor([x['parameter'] for x in items]),items


def state_samples_batch(items,times,c,norms=None):
    """Shared heat/wave reference evaluation; fixed context count, arbitrary times."""
    from pdeflow.numerics.dynamic import evaluate_batch
    q=np.stack([np.c_[item['context'],np.full(len(item['context']),float(t))] for item,t in zip(items,times)])
    values=evaluate_batch([x['solver'] for x in items],q,[x['parameter'] for x in items],c.interval)
    if norms is not None:
        for b,item in enumerate(items):
            for j,n in enumerate(norms[item['family']]):values[b,:,j]=(values[b,:,j]-n['mean'])/n['scale']
            if item['family']=='heat':values[b,:,1]=0
    return torch.tensor(values,dtype=torch.float32)
