"""Restartable independent heat instances; bounded shard cache and streaming stats."""
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path
import time

import numpy as np
import torch

from pdeflow.core import GridSpec, TransformBundle
from pdeflow.equations import heat_problem
from pdeflow.numerics import heat_reference, smooth_initial
from pdeflow.observations import sensor_observations
from pdeflow.runtime import atomic_write, digest, file_digest, read_json, write_json


def plan(config):
    dc=config.data;seeds=np.random.SeedSequence(dc.seed).spawn(dc.count+1)
    counts=dc.split_counts or (int(.6*dc.count),max(1,int(.2*dc.count)),dc.count-int(.6*dc.count)-max(1,int(.2*dc.count)))
    order=np.random.default_rng(seeds[-1]).permutation(dc.count).tolist();a,b,_=counts
    identity={'generator':'diverse_fourier_heat_v2','grid':{k:v for k,v in asdict(config.grid).items() if k != "temporal_kind"},'data':asdict(dc)}
    return {'schema_version':2,'family':'heat',**identity,'data_config':asdict(dc),
            'ids':[digest({**identity,'index':j})[:20] for j in range(dc.count)],
            'record_seeds':[int.from_bytes(s.generate_state(4).tobytes(), 'little') for s in seeds[:-1]],
            'splits':{'train':order[:a],'validation':order[a:a+b],'test':order[a+b:]},
            'storage_dtype':dc.storage_dtype,'units':'nondimensional','fields':[{'name':'u','role':'unknown','meaning':'signed scalar heat field'}]}


def instance(config, seed):
    rng=np.random.default_rng(seed);dc=config.data
    diffusivity=float(rng.uniform(*dc.diffusivity_range))
    bandwidth=float(rng.uniform(*(dc.bandwidth_range or (dc.bandwidth,dc.bandwidth))))
    amplitude=float(rng.uniform(*dc.amplitude_range));offset=float(rng.uniform(*dc.offset_range))
    initial=amplitude*smooth_initial(rng,config.grid,bandwidth)+offset
    return heat_reference(initial,diffusivity,config.grid)[None].astype(dc.storage_dtype),diffusivity


def estimate(config, samples=8):
    if config.specialist != 'heat':
        raise ValueError('estimate-data implements the heat reference probe only; use Darcy generation/preflight diagnostics for Darcy costs')
    if not 1<=samples<=32:raise ValueError('Cost probe requires 1–32 independent instances')
    start=time.perf_counter();seeds=plan(config)['record_seeds'][:samples];planning_seconds=time.perf_counter()-start
    start=time.perf_counter();payload=[]
    for seed in seeds:payload.append(instance(config,seed)[0])
    reference_seconds=time.perf_counter()-start;compression_start=time.perf_counter()
    import io
    target=io.BytesIO();np.savez_compressed(target,values=np.stack(payload))
    compression_seconds=time.perf_counter()-compression_start
    seconds=reference_seconds+compression_seconds;raw=np.stack(payload).nbytes
    return {'samples':len(payload),'seconds':seconds,'seconds_per_instance':seconds/len(payload),
            'raw_bytes':int(config.data.count*raw/len(payload)),
            'estimated_compressed_bytes':int(config.data.count*len(target.getvalue())/len(payload)),
            'planning_seconds':planning_seconds,'reference_seconds':reference_seconds,'compression_seconds':compression_seconds,
            'projected_generation_seconds':planning_seconds+seconds*config.data.count/len(payload),
            'grid':list(config.grid.shape),'channels':1,'times':list(config.grid.times),'dtype':config.data.storage_dtype,
            'format':'compressed NPZ shards','scope':'Small-batch reference+compression probe; projection is not a runtime guarantee'}


def generate_shards(config, output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);metadata=plan(config)
    if (output/'manifest.json').exists():raise FileExistsError('Completed dataset exists; choose a new directory')
    identity=output/'generation.json'
    if identity.exists():
        if digest(read_json(identity))!=digest(metadata):raise ValueError('Interrupted generation configuration differs; choose a new directory')
    else:
        if any(output.iterdir()):raise FileExistsError('Nonempty destination without a generation identity')
        write_json(identity,metadata)
    test_identity={'ids':[metadata['ids'][i] for i in metadata['splits']['test']],
                   'record_seeds':[metadata['record_seeds'][i] for i in metadata['splits']['test']]}
    test_path=output/'test_manifest.json'
    if test_path.exists() and read_json(test_path)!=test_identity:raise ValueError('Held-out manifest changed')
    if not test_path.exists():write_json(test_path,test_identity)
    start=time.perf_counter();shards=[]
    for first in range(0,config.data.count,config.data.shard_size):
        last=min(first+config.data.shard_size,config.data.count);name=f'shard-{first:06d}.npz';side=output/(name+'.json');path=output/name
        if side.exists():
            record=read_json(side)
            if record['file']!=name or record['first']!=first or record['last']!=last or file_digest(path)!=record['sha256']:raise ValueError('Completed shard integrity mismatch')
        else:
            # An interrupted write without its sidecar is recomputed from immutable seeds.
            batch=[instance(config,metadata['record_seeds'][i]) for i in range(first,last)]
            atomic_write(path,lambda f:np.savez_compressed(f,values=np.stack([v for v,_ in batch]),diffusivity=np.array([d for _,d in batch])))
            record={'file':name,'first':first,'last':last,'sha256':file_digest(path)};write_json(side,record)
        shards.append(record)
    metadata.update(shards=shards,sha256=digest(shards),generation_seconds=time.perf_counter()-start)
    values=LazyValues(output,metadata);k=metadata['splits']['test'][0];truth=values[[k]]
    d=float(values.diffusivity([k])[0]);obs=sensor_observations(truth,config.grid,config.inference.sensors,config.inference.seed,config.inference.noise_std,config.inference.noise_floor)
    write_json(output/'test_problem.json',heat_problem(config.grid,d).to_dict());write_json(output/'test_observations.json',obs.to_dict())
    atomic_write(output/'test_reference.npz',lambda f:np.savez_compressed(f,values=truth.numpy()))
    write_json(output/'manifest.json',metadata)
    return metadata


class LazyValues:
    def __init__(self,path,metadata):
        self.path,self.metadata=path,metadata;self.cache=OrderedDict();self.verified=set()
        self.shape=(len(metadata['ids']),1,len(metadata['grid']['times']),*metadata['grid']['shape'])
    def locate(self,index):
        if not 0<=index<self.shape[0]:raise IndexError(index)
        # Fixed-size, ordered shards; no dataset-sized tensor or device allocation.
        shard=self.metadata['shards'][index//self.metadata['data_config']['shard_size']];name=shard['file']
        if name not in self.cache:
            path=self.path/name
            if name not in self.verified:
                if file_digest(path)!=shard['sha256']:raise ValueError('Shard integrity hash mismatch')
                self.verified.add(name)
            with np.load(path,allow_pickle=False) as a:values,d=a['values'].copy(),a['diffusivity'].copy()
            expected=(shard['last']-shard['first'],*self.shape[1:])
            if values.shape!=expected or d.shape!=(expected[0],) or not np.isfinite(values).all() or not np.isfinite(d).all() or np.any(d<=0):raise ValueError('Invalid shard values/layout')
            self.cache[name]=(values,d)
            if len(self.cache)>2:self.cache.popitem(last=False)
        self.cache.move_to_end(name)
        return self.cache[name],index-shard['first']
    def __getitem__(self,indices):
        if isinstance(indices,slice):indices=list(range(*indices.indices(self.shape[0])))
        if isinstance(indices,torch.Tensor):indices=indices.cpu().tolist()
        result=[None]*len(indices)
        for position in sorted(range(len(indices)),key=lambda j:int(indices[j])):
            arrays,local=self.locate(int(indices[position]));result[position]=arrays[0][local].copy()
        return torch.from_numpy(np.stack(result))
    def diffusivity(self,indices):
        return torch.tensor([float(self.locate(int(i))[0][1][self.locate(int(i))[1]]) for i in indices],dtype=torch.float64)


class ShardedData:
    def __init__(self,path,metadata):self.metadata=metadata;self.values=LazyValues(path,metadata)
    def indices(self,split):return self.metadata['splits'][split]
    def conditioning(self,indices):
        from pdeflow.core import Conditioning
        return Conditioning('heat',{'diffusivity':self.values.diffusivity(indices)})
    def transform(self):
        original=self.indices('train');indices=sorted(original);n=0;mean=torch.zeros(1,dtype=torch.float64);m2=mean.clone()
        for start in range(0,len(indices),self.metadata['data_config']['shard_size']):
            values=self.values[indices[start:start+self.metadata['data_config']['shard_size']]].double()
            axes=(0,*range(2,values.ndim));batch_mean=values.mean(axes);count=values.numel()//values.shape[1]
            batch_m2=((values-batch_mean.reshape(1,-1,*([1]*(values.ndim-2))))**2).sum(axes)
            delta=batch_mean-mean;total=n+count;m2+=batch_m2+delta.square()*(n*count/total);mean+=delta*(count/total);n=total
        return TransformBundle(tuple(mean.tolist()),tuple((m2/n).sqrt().clamp_min(1e-8).tolist()),tuple(self.metadata['ids'][i] for i in original))


def load_shards(path,config):
    m=read_json(Path(path)/'manifest.json');expected=plan(config)
    for key in ['schema_version','family','grid','data_config','ids','record_seeds','splits']:
        if digest(m[key])!=digest(expected[key]):raise ValueError('Sharded dataset configuration/instance/split mismatch')
    if digest(m['shards'])!=m['sha256']:raise ValueError('Shard manifest digest mismatch')
    for first,shard in zip(range(0,config.data.count,config.data.shard_size),m['shards'],strict=True):
        if shard['first']!=first or shard['last']!=min(first+config.data.shard_size,config.data.count) or shard['file']!=f'shard-{first:06d}.npz':raise ValueError('Invalid shard indexing')
    held=read_json(Path(path)/'test_manifest.json')
    if held!={'ids':[m['ids'][i] for i in m['splits']['test']],'record_seeds':[m['record_seeds'][i] for i in m['splits']['test']]}:raise ValueError('Held-out manifest changed')
    return ShardedData(Path(path),m)
