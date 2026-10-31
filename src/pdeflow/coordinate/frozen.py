"""Bounded frozen encoder cache. Immutable identity is hashed once per stage.

Only CPU reference preparation is prefetched; model execution stays on the owner
thread. At most one next reference chunk is in flight. No observation/flow RNG or
conditioning values enter this cache.
"""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from io import BytesIO
from pathlib import Path
import time
import torch
from pdeflow.runtime import digest,atomic_write


class FrozenTargets:
    def __init__(self,identity,limit,timings,root=None,prefetch=1):
        tick=time.perf_counter();self.prefix=identity if isinstance(identity,str) else digest(identity);self.identity_hash_seconds=time.perf_counter()-tick
        self.limit=limit;self.timings=timings;self.root=Path(root) if root else None;self.prefetch=prefetch
        self.memory=OrderedDict();self.stats={'memory_hits':0,'deduplicated_requests':0,'disk_hits':0,'misses':0,'reference_worker_seconds':0.,'reference_requests':0,'encoding_batches':0,'maximum_encoding_batch':0}
        inventory_started=time.perf_counter()
        with timings.measure('frozen_disk_inventory',host=True):
            self.disk=set(p.stem for p in self.root.glob('*.pt')) if self.root else set()
        self.inventory_seconds_setup=time.perf_counter()-inventory_started
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='frozen-reference') if prefetch else None
    def close(self):
        if self.pool:self.pool.shutdown(wait=True,cancel_futures=True)
    def key(self,item,t,amp):
        # Prefix covers checkpoint/source, dataset, coordinate sampling,
        # normalization and precision. Do NOT serialize it again here.
        return (self.prefix,item['id'],float(t).hex(),bool(amp))
    def _put(self,key,value):
        self.memory[key]=value;self.memory.move_to_end(key)
        if len(self.memory)>self.limit:self.memory.popitem(last=False)
    def resolve(self,requests,amp,size,prepare,encode,device=None):
        """requests=(item,exact physical time); return raw complete state latents."""
        with self.timings.measure('frozen_key_construction',host=True):keys=[self.key(item,t,amp) for item,t in requests]
        found={};missing=OrderedDict()
        with self.timings.measure('frozen_cache_lookup',host=True):
            for key,request in zip(keys,requests):
                if key in found or key in missing:self.stats['deduplicated_requests']+=1;continue
                if key in self.memory:
                    self.stats['memory_hits']+=1;self.memory.move_to_end(key);found[key]=self.memory[key]
                elif key not in missing:missing[key]=request
        for key in list(missing):
            if self.root is None or float.fromhex(key[2])!=0.:continue
            with self.timings.measure('frozen_key_construction',host=True):name=digest(key)
            with self.timings.measure('frozen_cache_lookup',host=True):exists=name in self.disk
            if exists:
                with self.timings.measure('frozen_disk_read',host=True):payload=(self.root/(name+'.pt')).read_bytes()
                with self.timings.measure('frozen_deserialization',host=True):value=torch.load(BytesIO(payload),map_location='cpu',weights_only=True)
                self.stats['disk_hits']+=1;found[key]=value;self._put(key,value);del missing[key]
        self.stats['misses']+=len(missing)
        rows=list(missing.items());chunks=[rows[i:i+size] for i in range(0,len(rows),size)]
        def make(chunk):
            tick=time.perf_counter();data=prepare([request for _,request in chunk]);return data,time.perf_counter()-tick
        future=self.pool.submit(make,chunks[0]) if self.pool and chunks else None
        for i,chunk in enumerate(chunks):
            with self.timings.measure('frozen_reference_wait' if self.pool else 'frozen_reference_evaluation',host=True):
                data,seconds=future.result() if self.pool else make(chunk)
            self.stats['reference_worker_seconds']+=seconds;self.stats['reference_requests']+=len(chunk)
            # Single next chunk overlaps owner-thread GPU work and CPU stores.
            future=self.pool.submit(make,chunks[i+1]) if self.pool and i+1<len(chunks) else None
            values=encode([request for _,request in chunk],data);self.stats['encoding_batches']+=1;self.stats['maximum_encoding_batch']=max(self.stats['maximum_encoding_batch'],len(chunk))
            resident=values.detach()
            with self.timings.measure('frozen_output_transfer',host=True):values=resident.cpu()
            for j,(key,_) in enumerate(chunk):
                value=values[j:j+1].clone();found[key]=resident[j:j+1];self._put(key,value)
                if self.root is not None and float.fromhex(key[2])==0. and len(self.disk)<self.limit:
                    with self.timings.measure('frozen_key_construction',host=True):name=digest(key)
                    with self.timings.measure('frozen_serialization',host=True):buffer=BytesIO();torch.save(value,buffer);payload=buffer.getvalue()
                    with self.timings.measure('frozen_disk_write',host=True):atomic_write(self.root/(name+'.pt'),lambda out:out.write(payload))
                    self.disk.add(name)
        destination=device or next(iter(found.values())).device
        with self.timings.measure('frozen_input_transfer',host=True):
            transfer=[key for key in found if (found[key].device.type!=destination.type or destination.index is not None and found[key].device.index!=destination.index)]
            if transfer:
                values=torch.cat([found[key] for key in transfer])
                if destination.type=='cuda' and values.device.type=='cpu':values=values.pin_memory()
                values=values.to(destination,non_blocking=destination.type=='cuda')
                for i,key in enumerate(transfer):found[key]=values[i:i+1]
        with self.timings.measure('frozen_output_pack',host=True):return torch.cat([found[key] for key in keys])
    def report(self):
        return {**self.stats,'items':len(self.memory),'bytes':sum(x.numel()*x.element_size() for x in self.memory.values()),'disk_items':len(self.disk),'identity_hash_seconds_setup':self.identity_hash_seconds,'disk_inventory_seconds_setup':self.inventory_seconds_setup,'counter_scope':'Cumulative stage totals; worker seconds can overlap wall time','worker_time_scope':'CPU reference work; overlaps owner compute when prefetch=1, do not add to update wall time'}
