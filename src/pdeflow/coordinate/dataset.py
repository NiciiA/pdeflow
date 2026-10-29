"""Bounded deterministic complete heat solutions. Splits precede observations."""
from functools import lru_cache
from pathlib import Path
from dataclasses import replace
import time
import numpy as np
import torch
from pdeflow.core import GridSpec,DomainSpec
from pdeflow.numerics import heat_reference
from pdeflow.runtime import digest,write_json,read_json,file_digest

FAMILIES=('cosine','fourier_sum','smooth_random','hot_spot','multiple_spots','brush')


def recipe(c):
    from .optimization import options
    d={'families':{f:1. for f in FAMILIES},'max_wave':max(1,min(c.reference_shape)//8),'amplitude':[.4,1.2],
       'background':[-.5,.5],'width':[max(.08,3/min(c.reference_shape)),max(.18,4/min(c.reference_shape))]}
    if set(c.distribution)-d.keys():raise ValueError('Unknown distribution setting')
    d.update(c.distribution)
    d['families']=dict(sorted(d['families'].items())) # JSON/key order must never change a realization
    if c.dimension!=2:raise ValueError('Procedural mixture currently supports heat2D; legacy materialized training retains 1D–3D')
    if not d['families'] or set(d['families'])-set(FAMILIES) or any(not np.isfinite(w) or w<=0 for w in d['families'].values()):raise ValueError('Choose positive weights for supported initial-field families')
    if min(c.train_count,c.validation_count,c.test_count)<len(d['families']):raise ValueError('Each fixed split must contain at least one solution per enabled initial-field family')
    if type(d['max_wave'])!=int or not 1<=d['max_wave']<=min(c.reference_shape)//8:raise ValueError('Max wave needs at least eight reference nodes per wavelength')
    for key in ('amplitude','background','width'):
        if len(d[key])!=2 or not np.isfinite(d[key]).all() or d[key][1]<d[key][0]:raise ValueError('Invalid distribution range: '+key)
    if d['amplitude'][0]<=0 or d['width'][0]<3/min(c.reference_shape) or d['width'][1]>1:raise ValueError('Spot/brush widths must span at least three grid cells and be <= domain length')
    temporal=({'temporal_mode':'continuous','time_interval':list(c.interval),'time_sampling':__import__('pdeflow.coordinate.temporal',fromlist=['policies']).policies(c),'observation_placement':c.observation_placement or {'random':.5,'even':.25,'heat':.25}} if c.temporal_mode=='continuous' else {'times':list(c.times)})
    return {**temporal,'version':2 if c.temporal_mode=='continuous' else 1,'seed':c.seed,'shape':list(c.reference_shape),'lengths':list(c.lengths),'origins':list(c.origins),
            'diffusivity_range':list(c.diffusivity_range),'counts':[c.train_count,c.validation_count,c.test_count],
            'normalization_calibration_count':min(c.train_count,options(c)['calibration_count']),**d}


def identity(c):return digest(recipe(c))


def example(c,split,index,shape=None,dtype=np.float32):
    r=recipe(c);split_index={'train':0,'validation':1,'test':2}[split]
    if not 0<=index<r['counts'][split_index]:raise ValueError('Example index outside fixed split')
    rng=np.random.default_rng(np.random.SeedSequence([c.seed,split_index,index]))
    families=list(r['families']);weights=np.asarray(list(r['families'].values()),float);weights/=weights.sum()
    family=families[index] if index<len(families) else str(rng.choice(families,p=weights))
    grid=GridSpec(tuple(shape or c.reference_shape),DomainSpec(c.lengths,c.origins),c.times)
    x,y=np.meshgrid(np.arange(grid.shape[0])/grid.shape[0],np.arange(grid.shape[1])/grid.shape[1],indexing='ij')
    amplitude=rng.uniform(*r['amplitude']);background=rng.uniform(*r['background']);D=rng.uniform(*c.diffusivity_range)
    k=r['max_wave'];initial=np.full(grid.shape,background)
    if family=='cosine':
        a,b=rng.integers(0,k+1,2)
        if a==b==0:a=1
        initial+=amplitude*np.cos(2*np.pi*a*x+rng.uniform(-np.pi,np.pi))*np.cos(2*np.pi*b*y+rng.uniform(-np.pi,np.pi))
    elif family in ('fourier_sum','smooth_random'):
        modes=[(a,b) for a in range(-k,k+1) for b in range(-k,k+1) if (a>0 or a==0 and b>0)]
        if family=='fourier_sum':modes=[modes[j] for j in rng.choice(len(modes),min(6,len(modes)),replace=False)]
        weights=[np.exp(-(a*a+b*b)/(2*max(1,k/2)**2)) if family=='smooth_random' else 1. for a,b in modes]
        scale=np.sqrt(sum(w*w for w in weights))
        for (a,b),w in zip(modes,weights):initial+=amplitude*w/scale*rng.normal()*np.cos(2*np.pi*(a*x+b*y)+rng.uniform(-np.pi,np.pi))
    else:
        count=1 if family=='hot_spot' else int(rng.integers(2,5)) if family=='multiple_spots' else int(rng.integers(4,9))
        centers=rng.random((count,2))
        if family=='brush':centers=(centers[0]+np.arange(count)[:,None]*rng.uniform(-.08,.08,(1,2)))%1
        for center in centers:
            width=rng.uniform(*r['width']);sign=rng.choice([-1.,1.]);bump=np.exp(-(np.sin(np.pi*(x-center[0]))**2+np.sin(np.pi*(y-center[1]))**2)/(np.pi*width)**2)
            initial+=sign*amplitude/np.sqrt(count)*bump
    if c.temporal_mode=='continuous':
        from pdeflow.heat_continuous import FourierHeat
        from .temporal import points
        generator=torch.Generator().manual_seed(int(np.random.SeedSequence([c.seed,split_index,index,319]).generate_state(1)[0]))
        context=points(c,c.encoder_points,generator,'encoder')
        solver=FourierHeat(initial,c.lengths,c.origins)
        return {'values':initial.astype(dtype).ravel(),'D':float(D),'family':family,'id':f'{split}:{index}:{identity(c)[:16]}','seed':[c.seed,split_index,index],
                'context':context.numpy(),'context_values':solver.evaluate(context.numpy(),D,c.interval).astype(np.float32),'solver':solver}
    values=heat_reference(initial,D,grid).astype(dtype)
    return {'values':values.reshape(-1),'D':float(D),'family':family,'id':f'{split}:{index}:{identity(c)[:16]}','seed':[c.seed,split_index,index]}


class Dataset:
    def __init__(self,c,root,create=True):
        from .training import prepare,data_config
        from .contracts import grid_queries
        self.c=c;self.root=Path(root);self.materialized=c.data_mode=='materialized'
        if self.materialized:
            if not (self.root/'dataset.json').exists():
                if not create:raise ValueError('Missing recorded training data; restore it before resuming')
                prepare(c,self.root)
            self.metadata=read_json(self.root/'dataset.json')
            if self.metadata['sha256']!=file_digest(self.root/'dataset.npz'):raise ValueError('Dataset checksum mismatch')
            with np.load(self.root/'dataset.npz',allow_pickle=False) as f:self.values=f['values'].copy();self.parameters=f['diffusivity'].copy();self.q=torch.from_numpy(f['queries'].copy())
        else:
            from .optimization import options
            self.q=grid_queries(data_config(c).grid) if c.temporal_mode!='continuous' else None
            prefix=identity(c)[:16];ids=[f'{split}:{i}:{prefix}' for split,n in zip(('train','validation','test'),(c.train_count,c.validation_count,c.test_count)) for i in range(n)]
            if (self.root/'dataset.json').exists():
                self.metadata=read_json(self.root/'dataset.json')
                if self.metadata['sha256']!=identity(c):raise ValueError('Procedural dataset identity changed; no silent regeneration on resume')
            else:
                if not create:raise ValueError('Missing dataset recipe; restore the complete run')
                count=min(c.train_count,options(c)['calibration_count']);total=total2=points=0.;started=time.perf_counter()
                for i in range(count):
                    a=self.get('train',i)['values'].astype(np.float64);total+=a.sum();total2+=np.square(a).sum();points+=a.size
                mean=total/points;scale=max(np.sqrt(max(0.,total2/points-mean*mean)),1e-6)
                self.metadata={'schema_version':2,'kind':'coordinate_training_data','mode':'procedural','recipe':recipe(c),'sha256':identity(c),'ids':ids,
                   'splits':[s for s,n in zip(('train','validation','test'),(c.train_count,c.validation_count,c.test_count)) for _ in range(n)],
                   'normalization':{'mean':float(mean),'scale':float(scale),'fitted_ids':ids[:count]},'calibration_count':count,'preparation_seconds':time.perf_counter()-started,
                   'distribution':'Deterministic periodic family mixture; reference regenerated on local CPU; no full GPU dataset or latent cache'}
                write_json(self.root/'dataset.json',self.metadata)
    @lru_cache(maxsize=8)
    def get(self,split,index):
        if not self.materialized:return example(self.c,split,int(index))
        offset=0 if split=='train' else self.c.train_count if split=='validation' else self.c.train_count+self.c.validation_count
        i=offset+int(index)
        return {'values':self.values[i],'D':float(self.parameters[i]),'family':'legacy_smooth_random','id':self.metadata['ids'][i]}
    def batch(self,indices,split='train',workers=0):
        if workers:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=workers) as pool:items=list(pool.map(lambda i:self.get(split,int(i)),indices))
        else:items=[self.get(split,int(i)) for i in indices]
        return torch.from_numpy(np.stack([x['values'] for x in items])),torch.tensor([x['D'] for x in items]),items


def query_subset(c,generator):
    """Unique, stratified space-time nodes. Each physical time has equal loss weight."""
    spatial=int(np.prod(c.reference_shape));nt=len(c.times);n=min(c.query_points,spatial*nt)
    if n<nt:raise ValueError('Query points must include at least one point per physical time')
    indices=[];weights=[]
    for t in range(nt):
        count=n//nt+(t<n%nt);indices.append(torch.randperm(spatial,generator=generator)[:count]+t*spatial);weights.extend([1/(nt*count)]*count)
    return torch.cat(indices),torch.tensor(weights)
