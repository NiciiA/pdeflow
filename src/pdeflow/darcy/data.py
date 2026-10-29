"""Independent Darcy problems, restartable shards and bounded-memory loading."""
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path
import time
import numpy as np
import torch
from pdeflow.core import Conditioning
from pdeflow.darcy import DarcyTransform,problem,solve
from pdeflow.numerics import smooth_initial
from pdeflow.observations import sensor_observations
from pdeflow.runtime import atomic_write,digest,file_digest,read_json,write_json


def draw_permeability(rng,grid,amplitude,bandwidth,offset=0.):
    return np.exp(offset+amplitude*np.tanh(smooth_initial(rng,grid,bandwidth)))


def identity(c):
    dc=c.data;seeds=np.random.SeedSequence(dc.seed).spawn(dc.count+1)
    counts=dc.split_counts or (int(.6*dc.count),max(1,int(.2*dc.count)),dc.count-int(.6*dc.count)-max(1,int(.2*dc.count)))
    order=np.random.default_rng(seeds[-1]).permutation(dc.count).tolist();a,b,_=counts
    key={'generator':'periodic_darcy_fv_v1','grid':asdict(c.grid),'data_config':asdict(dc)}
    return {**key,'schema_version':3,'family':'darcy','fields':['a','u'],'ids':[digest({**key,'index':j})[:20] for j in range(dc.count)],
            'record_seeds':[int.from_bytes(s.generate_state(4).tobytes(),'little') for s in seeds[:-1]],'splits':{'train':order[:a],'validation':order[a:a+b],'test':order[a+b:]}}


def generate(c,output):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);m=identity(c);start=time.perf_counter()
    if (out/'manifest.json').exists():raise FileExistsError('Completed Darcy data exists; choose a new directory')
    if (out/'generation.json').exists():
        if digest(read_json(out/'generation.json'))!=digest(m):raise ValueError('Darcy generation identity differs')
    else:
        if any(out.iterdir()):raise FileExistsError('Nonempty data directory without generation identity')
        write_json(out/'generation.json',m)
    held={'ids':[m['ids'][j] for j in m['splits']['test']]}
    if (out/'test_manifest.json').exists() and read_json(out/'test_manifest.json')!=held:raise ValueError('Immutable Darcy test manifest changed')
    if not (out/'test_manifest.json').exists():write_json(out/'test_manifest.json',held)
    shards=[]
    for first in range(0,c.data.count,c.data.shard_size):
        last=min(first+c.data.shard_size,c.data.count);name=f'shard-{first:06d}.npz';side=out/(name+'.json')
        if side.exists():
            record=read_json(side)
            if record['file']!=name or record['first']!=first or record['last']!=last or file_digest(out/name)!=record['sha256']:raise ValueError('Darcy shard integrity mismatch')
        else:
            values=[];forcing=[];diagnostics=[]
            for j in range(first,last):
                rng=np.random.default_rng(m['record_seeds'][j]);amp=float(rng.uniform(*c.data.amplitude_range));off=float(rng.uniform(*c.data.offset_range))
                a=draw_permeability(rng,c.grid,amp,c.data.bandwidth,off)
                f=smooth_initial(rng,c.grid,c.data.bandwidth);f-=f.mean()
                u,diagnostic=solve(a,f,c.grid);diagnostics.append(diagnostic)
                values.append(np.stack((a,u))[:,None].astype(c.data.storage_dtype));forcing.append(f) # preserve compatible forcing in CPU float64
            atomic_write(out/name,lambda stream:np.savez_compressed(stream,values=np.stack(values),forcing=np.stack(forcing)))
            record={'file':name,'first':first,'last':last,'sha256':file_digest(out/name),'solver_diagnostics':diagnostics};write_json(side,record)
        shards.append(record)
    m.update(shards=shards,sha256=digest(shards),generation_seconds=time.perf_counter()-start)
    data=DarcyData(out,m);j=m['splits']['test'][0];truth=data.values[[j]];f=data.forcing([j])[0].numpy();p=problem(c.grid,f)
    obs=sensor_observations(truth,c.grid,c.inference.sensors,c.inference.seed,c.inference.noise_std,c.inference.noise_floor)
    write_json(out/'test_problem.json',p.to_dict());write_json(out/'test_observations.json',obs.to_dict())
    atomic_write(out/'test_reference.npz',lambda stream:np.savez_compressed(stream,values=truth.numpy()))
    write_json(out/'manifest.json',m);return m


class DarcyValues:
    def __init__(self,path,m):self.path=path;self.metadata=m;self.cache=OrderedDict();self.verified=set();self.shape=(len(m['ids']),2,1,*m['grid']['shape'])
    def item(self,i):
        if not 0<=i<self.shape[0]:raise IndexError(i)
        shard=self.metadata['shards'][i//self.metadata['data_config']['shard_size']];name=shard['file']
        if name not in self.cache:
            if name not in self.verified:
                if file_digest(self.path/name)!=shard['sha256']:raise ValueError('Darcy shard hash mismatch')
                self.verified.add(name)
            with np.load(self.path/name,allow_pickle=False) as a:v,f=a['values'].copy(),a['forcing'].copy()
            if v.shape!=(shard['last']-shard['first'],*self.shape[1:]) or f.shape!=(len(v),*self.shape[3:]) or not np.isfinite(v).all() or not np.isfinite(f).all() or np.any(v[:,0]<=0):raise ValueError('Darcy array/layout/positivity mismatch')
            self.cache[name]=(v,f)
            if len(self.cache)>2:self.cache.popitem(last=False)
        self.cache.move_to_end(name);return self.cache[name],i-shard['first']
    def batch(self,indices,which=0):
        if isinstance(indices,slice):indices=list(range(*indices.indices(self.shape[0])))
        if isinstance(indices,torch.Tensor):indices=indices.cpu().tolist()
        result=[None]*len(indices)
        for j in sorted(range(len(indices)),key=lambda j:int(indices[j])):
            arrays,local=self.item(int(indices[j]));result[j]=arrays[which][local].copy()
        return torch.from_numpy(np.stack(result))
    def __getitem__(self,indices):return self.batch(indices)


class DarcyData:
    def __init__(self,path,m):self.metadata=m;self.values=DarcyValues(path,m)
    def indices(self,split):return self.metadata['splits'][split]
    def forcing(self,indices):return self.values.batch(indices,1)
    def conditioning(self,indices):return Conditioning('darcy',{}, {'f':self.forcing(indices)})
    def transform(self):
        original=self.indices('train');indices=sorted(original);n=0;mean=torch.zeros(2,dtype=torch.float64);m2=mean.clone()
        for j in range(0,len(indices),self.metadata['data_config']['shard_size']):
            v=self.values[indices[j:j+self.metadata['data_config']['shard_size']]].double();v[:,0]=v[:,0].log();axes=(0,*range(2,v.ndim));b=v.mean(axes);count=v.numel()//2
            centered=v-b.reshape(1,2,*([1]*(v.ndim-2)));delta=b-mean;total=n+count
            m2+=centered.square().sum(axes)+delta.square()*n*count/total;mean+=delta*count/total;n=total
        return DarcyTransform(tuple(mean.tolist()),tuple((m2/n).sqrt().clamp_min(1e-8).tolist()),tuple(self.metadata['ids'][i] for i in original))


def load(path,c):
    path=Path(path);m=read_json(path/'manifest.json');expected=identity(c)
    for k in expected:
        if digest(m.get(k))!=digest(expected[k]):raise ValueError('Darcy data/split/grid/configuration identity mismatch')
    if digest(m['shards'])!=m['sha256']:raise ValueError('Darcy shard manifest mismatch')
    for first,s in zip(range(0,c.data.count,c.data.shard_size),m['shards'],strict=True):
        if s['first']!=first or s['last']!=min(first+c.data.shard_size,c.data.count) or s['file']!=f'shard-{first:06d}.npz':raise ValueError('Darcy shard indexing mismatch')
    if read_json(path/'test_manifest.json')!={'ids':[m['ids'][j] for j in m['splits']['test']]}:raise ValueError('Darcy held-out manifest changed')
    return DarcyData(path,m)
