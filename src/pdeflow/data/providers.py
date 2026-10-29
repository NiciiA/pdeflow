"""Synchronous, bounded physical-data providers with explicit consumed state."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from typing import Protocol
import json
import time
import numpy as np
import torch
from pdeflow.core import ProblemSpec
from pdeflow.data.physical import PhysicalSample,combine,distribution,seed_for,solve_spec,specification
from pdeflow.observations import sensor_observations
from pdeflow.runtime import atomic_write,digest,file_digest,read_json,write_json
from pdeflow.specialists import get_specialist

class TrainingDataProvider(Protocol):
    transform: object
    metadata: dict
    def training_batch(self,rng,size): ...
    def validation_batches(self,size): ...
    def state_dict(self): ...
    def close(self): ...


def reference_sample(c,namespace,index,event=None):
    failures=[]
    for attempt in range(c.provider.max_attempts):
        spec,sampling=specification(c,namespace,index,attempt)
        solve_start=time.perf_counter()
        if event:event({'phase':'started','accepted':None,'specification':spec})
        try:
            sample=solve_spec(c,spec,sampling)
            if event:event({'phase':'finished','accepted':True,'specification':spec,'diagnostics':sample.diagnostics,'timings':sample.timings})
            return sample
        except (RuntimeError,ValueError,FloatingPointError) as error:
            failure={'phase':'finished','accepted':False,'specification':spec,'error':str(error),'timings':{'sampling_seconds':sampling,'reference_seconds':time.perf_counter()-solve_start}};failures.append(failure)
            if event:event(failure)
    raise RuntimeError(f'Reference generation exhausted {c.provider.max_attempts} attempts: {failures}')


def store_sample(path,sample,c,observations=False):
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    atomic_write(path/'target.npz',lambda f:np.savez_compressed(f,values=sample.values.numpy()))
    record={'specification':sample.specification,'problem':sample.problem.to_dict(),'diagnostics':sample.diagnostics,'timings':sample.timings,'target_sha256':file_digest(path/'target.npz')}
    if observations:
        seed=seed_for(c,sample.specification['namespace'],sample.specification['index'],sample.specification['attempt'],'observations')%(2**63)
        obs=sensor_observations(sample.values[None],c.grid,c.inference.sensors,seed,c.inference.noise_std,c.inference.noise_floor)
        record['observations']=obs.to_dict();write_json(path/'observations.json',obs.to_dict())
        write_json(path/'problem.json',sample.problem.to_dict())
        atomic_write(path/'reference.npz',lambda f:np.savez_compressed(f,values=sample.values[None].numpy()))
    write_json(path/'sample.json',record);return record


def load_sample(path,c):
    path=Path(path);r=read_json(path/'sample.json')
    if digest(r['specification']['distribution'])!=digest(distribution(c)) or file_digest(path/'target.npz')!=r['target_sha256']:raise ValueError('Cached physical target has incompatible specification or checksum')
    with np.load(path/'target.npz',allow_pickle=False) as a:values=torch.from_numpy(a['values'].copy())
    return PhysicalSample(r['specification'],ProblemSpec.from_dict(r['problem']),values,r['diagnostics'],r['timings'])


def prepare(c,output,training=False,resume=False):
    if c.provider is None:raise ValueError('A provider configuration is required')
    c.validate(check_device=False);output=Path(output)
    if output.exists() and any(output.iterdir()):
        if not resume:raise FileExistsError('Choose a fresh physical manifest directory')
        if (output/'manifest.json').exists():return check_manifest(output,c,'physical_training' if training else 'fixed_evaluation')
    output.mkdir(parents=True,exist_ok=True);events=[];start=time.perf_counter()
    namespaces={'train':c.provider.offline_count} if training else {'validation':c.provider.validation_count,'test':c.provider.test_count}
    records={}
    for namespace,count in namespaces.items():
        records[namespace]=[]
        for i in range(count):
            destination=output/namespace/str(i)
            if resume and (destination/'sample.json').exists():
                sample=load_sample(destination,c)
                if sample.specification['namespace']!=namespace or sample.specification['index']!=i:raise ValueError('Partial preparation record has wrong namespace/index')
                required=['target.npz'] if training else ['target.npz','problem.json','observations.json','reference.npz']
                record=read_json(destination/'sample.json') if all((destination/f).exists() for f in required) else store_sample(destination,sample,c,not training)
            else:
                sample=reference_sample(c,namespace,i,events.append);record=store_sample(destination,sample,c,not training)
            records[namespace].append({'id':sample.specification['id'],'index':i,'record_sha256':file_digest(output/namespace/str(i)/'sample.json')})
    manifest={'schema_version':2,'distribution':distribution(c),'records':records,'attempts':events,'seconds':time.perf_counter()-start,
              'identity':digest({'distribution':distribution(c),'records':{k:[{'id':r['id'],'index':r['index']} for r in rows] for k,rows in records.items()}}),'kind':'physical_training' if training else 'fixed_evaluation'}
    write_json(output/'manifest.json',manifest);return manifest


def check_manifest(path,c,kind):
    path=Path(path);m=read_json(path/'manifest.json')
    records=m['records'] if m.get('schema_version')==1 else {k:[{'id':r['id'],'index':r['index']} for r in rows] for k,rows in m['records'].items()}
    if m.get('schema_version') not in (1,2) or m['kind']!=kind or digest(m['distribution'])!=digest(distribution(c)) or m['identity']!=digest({'distribution':m['distribution'],'records':records}):raise ValueError('Incompatible/corrupt fixed physical manifest')
    for ns,records in m['records'].items():
        for r in records:
            if file_digest(path/ns/str(r['index'])/'sample.json')!=r['record_sha256']:raise ValueError('Immutable physical record changed')
    return m


class PhysicalProvider:
    """Online fresh stream, bounded round-robin replay, or ID-matched disk data.

    Single producer/rank only. Hybrid refreshes a slot after reuse_limit total
    presentations. Cache arrays are never checkpointed: IDs regenerate them.
    """
    def __init__(self,c,evaluation,output,data_path=None,saved=None,transform=None,cache_calibration=False,event=None):
        self.c=c;self.settings=c.provider;self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.evaluation=Path(evaluation) if evaluation else None
        if self.evaluation is None:raise ValueError('Online/hybrid training requires explicit --evaluation fixed manifests; prepare-evaluation first')
        self.eval_manifest=check_manifest(self.evaluation,c,'fixed_evaluation')
        self.cursor=0;self.presentations=0;self.reused=0;self.pool=[];self.closed=False;self.last_ids=[]
        self.accounting={'sampling_seconds':0.,'reference_seconds':0.,'cache_load_seconds':0.,'data_wait_seconds':0.,'reference_solve_calls':0,'accepted_reference_calls':0,'failed_attempts':0,'regenerated_cache_calls':0}
        self.calibration_ids=[];self.mode=self.settings.mode;self.offline=None
        if self.mode!='offline' and data_path is not None:raise ValueError('Online/hybrid providers do not consume --data; use the explicit fixed --evaluation manifest')
        if self.mode=='offline':
            if not data_path:raise ValueError('Offline provider requires --data from prepare-training')
            self.data_path=Path(data_path);self.offline=check_manifest(self.data_path,c,'physical_training')
        if saved:
            if saved['settings']!=asdict(self.settings) or saved['distribution_sha256']!=digest(distribution(c)) or saved['evaluation_identity']!=self.eval_manifest['identity']:raise ValueError('Resume provider/distribution/evaluation mismatch')
            self.cursor=saved['cursor'];self.presentations=saved['presentations'];self.reused=saved['reused'];self.accounting=saved['accounting'].copy();self.calibration_ids=saved['calibration_ids'];self.consumed_ids=list(saved.get('consumed_ids',[]));self.pool=[dict(slot) for slot in saved['pool']]
            if self.mode=='offline' and saved['offline_identity']!=self.offline['identity']:raise ValueError('Resume offline physical data changed')
            if transform is None:raise ValueError('Resume requires the frozen checkpoint transform')
            self.transform=transform
            for slot in self.pool:
                slot['sample']=self._regenerate(slot['specification'])
        elif cache_calibration and (self.output/'calibration.json').exists():
            frozen=read_json(self.output/'calibration.json')
            from pdeflow.runtime import code_identity
            if frozen['distribution']!=digest(distribution(c)) or frozen['count']!=self.settings.calibration_count or frozen['source']!=code_identity()['source_sha256']:raise ValueError('Frozen calibration incompatible; preserve this run and start a fresh run')
            self.calibration_ids=frozen['ids']
            self.transform=get_specialist(c.specialist).transform_type(**{k:tuple(v) for k,v in frozen['transform'].items()})
            self.accounting=frozen.get('accounting',self.accounting).copy()
        else:
            if event:event({'kind':'stage','stage':'Calibrating transforms','count':self.settings.calibration_count})
            calibration=[self._generate('calibration',i) for i in range(self.settings.calibration_count)]
            self.calibration_ids=[x.specification['id'] for x in calibration]
            self.transform=get_specialist(c.specialist).transform_type.fit(combine(calibration)[0],self.calibration_ids)
            if cache_calibration:
                from pdeflow.runtime import code_identity
                write_json(self.output/'calibration.json',{'distribution':digest(distribution(c)),'count':self.settings.calibration_count,'source':code_identity()['source_sha256'],'ids':self.calibration_ids,'transform':asdict(self.transform),'accounting':self.accounting.copy()})
        ids=self.calibration_ids+[r['id'] for ns in ('validation','test') for r in self.eval_manifest['records'][ns]]
        n=len(self.calibration_ids);v=len(self.eval_manifest['records']['validation'])
        self.metadata={'ids':ids,'splits':{'train':list(range(n)),'validation':list(range(n,n+v)),'test':list(range(n+v,len(ids)))},'sha256':digest({'distribution':distribution(c),'evaluation':self.eval_manifest['identity'],'offline':self.offline['identity'] if self.offline else None})}
    def _event(self,event):
        if event.get('phase')=='started':self.accounting['reference_solve_calls']+=1
        else:
            if event['accepted']:self.accounting['accepted_reference_calls']+=1
            else:self.accounting['failed_attempts']+=1
            for k,v in event.get('timings',{}).items():self.accounting[k]+=v
        with (self.output/'data-events.jsonl').open('a') as f:f.write(json.dumps(event,allow_nan=False)+'\n')
    def journal_audit(self):
        attempted=completed=failed=0;seconds=0.;unique=set();partial_lines=0
        file=self.output/'data-events.jsonl'
        if file.exists():
            for line in file.open():
                try:event=json.loads(line)
                except json.JSONDecodeError:partial_lines+=1;continue
                if event.get('phase')=='started':attempted+=1;continue
                if 'phase' not in event:attempted+=1 # pre-audit journals
                completed+=1;failed+=not event['accepted'];seconds+=event.get('timings',{}).get('reference_seconds',0.)
                if event['accepted']:unique.add(event['specification']['id'])
        return {'attempted_reference_calls':attempted,'completed_reference_calls':completed,'failed_reference_calls':failed,'incomplete_attempts':max(0,attempted-completed),'partial_journal_lines':partial_lines,'unique_accepted_physical_ids':len(unique),'completed_reference_seconds':seconds,'scope':'Entire journal including regeneration and discarded work after the last checkpoint; incomplete-call duration is unknown'}
    def _generate(self,ns,i):return reference_sample(self.c,ns,i,self._event)
    def _regenerate(self,spec):
        start=time.perf_counter();self._event({'phase':'started','accepted':None,'specification':spec});sample=solve_spec(self.c,spec);self.accounting['regenerated_cache_calls']+=1;self._event({'phase':'finished','accepted':True,'regeneration':True,'specification':spec,'diagnostics':sample.diagnostics,'timings':sample.timings});self.accounting['cache_load_seconds']+=time.perf_counter()-start;return sample
    def _fresh(self):
        sample=self._generate('train',self.cursor);self.cursor+=1;return sample
    def training_batch(self,rng,size):
        if self.closed:raise RuntimeError('Data provider is closed')
        start=time.perf_counter();samples=[]
        if self.mode=='offline':
            count=len(self.offline['records']['train']);ix=torch.randint(count,(size,),generator=rng).tolist()
            t=time.perf_counter();samples=[load_sample(self.data_path/'train'/str(i),self.c) for i in ix];self.accounting['cache_load_seconds']+=time.perf_counter()-t
        else:
            for _ in range(size):
                if self.mode=='online':sample=self._fresh()
                else:
                    if not self.pool:
                        for _ in range(self.settings.capacity):
                            new=self._fresh();self.pool.append({'specification':new.specification,'uses':0,'sample':new})
                    slot_index=(self.presentations+len(samples))%self.settings.capacity;slot=self.pool[slot_index]
                    if slot['uses']>=self.settings.reuse_limit:
                        new=self._fresh();slot={'specification':new.specification,'uses':0,'sample':new};self.pool[slot_index]=slot
                    if slot['uses']>0:self.reused+=1
                    slot['uses']+=1;sample=slot['sample']
                samples.append(sample)
        self.presentations+=len(samples);self.last_ids=[s.specification['id'] for s in samples]
        if self.mode=='offline':self.reused=self.presentations-len(set(self.consumed_ids+self.last_ids)) if hasattr(self,'consumed_ids') else 0
        self.consumed_ids=getattr(self,'consumed_ids',[])+self.last_ids
        self.accounting['data_wait_seconds']+=time.perf_counter()-start
        return combine(samples)
    def validation_batches(self,size):
        count=len(self.eval_manifest['records']['validation'])
        for first in range(0,count,size):yield combine([load_sample(self.evaluation/'validation'/str(i),self.c) for i in range(first,min(first+size,count))])
    def state_dict(self):
        return {'settings':asdict(self.settings),'distribution_sha256':digest(distribution(self.c)),'evaluation_identity':self.eval_manifest['identity'],'offline_identity':self.offline['identity'] if self.offline else None,'cursor':self.cursor,'presentations':self.presentations,'reused':self.reused,'calibration_ids':self.calibration_ids,'pool':[{k:v for k,v in s.items() if k!='sample'} for s in self.pool],'accounting':self.accounting.copy(),'consumed_ids':getattr(self,'consumed_ids',[])}
    def report(self):
        return {**self.state_dict(),'journal_audit':self.journal_audit(),'mode':self.mode,'unique_training_problems_solved':self.cursor if self.mode!='offline' else len(self.offline['records']['train']),'unique_training_problems_consumed':len(set(getattr(self,'consumed_ids',[]))),'generated_ahead':sum(s['uses']==0 for s in self.pool),'failure_rate':self.accounting['failed_attempts']/max(1,self.accounting['reference_solve_calls']),'normalization':'frozen_training_only_calibration','parallelism':'single producer, no workers/prefetch; reference FFT/reductions on CPU'}
    def close(self):self.pool.clear();self.closed=True


class LegacyOfflineProvider:
    def __init__(self,c,path):
        from pdeflow.data import load_dataset
        if path is None:raise ValueError('Offline training requires --data')
        self.c=c;self.data=load_dataset(path,c);self.metadata=self.data.metadata
        self.ids=[self.metadata['ids'][i] for i in self.data.indices('train')]
        self.transform=get_specialist(c.specialist).fit_transform(self.data,self.ids);self.presentations=0;self.last_ids=[]
    def training_batch(self,rng,size):
        ids=self.data.indices('train');ix=[ids[j] for j in torch.randint(len(ids),(size,),generator=rng).tolist()];self.last_ids=[self.metadata['ids'][j] for j in ix];self.presentations+=size
        return self.data.values[ix],self.data.conditioning(ix)
    def validation_batches(self,size):
        ids=self.data.indices('validation')
        for first in range(0,len(ids),size):
            ix=ids[first:first+size];yield self.data.values[ix],self.data.conditioning(ix)
    def state_dict(self):return {'mode':'legacy_offline','presentations':self.presentations}
    def report(self):return self.state_dict()
    def close(self):pass
