"""Cold/warm reference-only control at the requested point/grid/loss counts.

No neural capacity is allocated. Not an end-to-end training/GPU speed estimate.
"""
import argparse
import json
from pathlib import Path
from dataclasses import replace
import time
import numpy as np
import torch
from pdeflow.coordinate.contracts import CoordinateConfig
from pdeflow.coordinate.dataset import example,state_samples,item_quantity
from pdeflow.field_losses import grouped_queries,AnalysisCache
from pdeflow.coordinate.profiling import distribution
from pdeflow.runtime import write_json


def run(c,repeats=8):
    result={'config':c.to_dict(),'repeats':repeats,'scope':'CPU reference/region supervision only, not neural training. Same requested reference, encoder/domain/region points and family/field loss. Unit normalization in both controls. Per-repeat time is a batch of requested batch_size solutions. Fresh solver/analysis cache for cold; exact same reference/time/config for warm. Full physical interval remains supported.'}
    fingerprints={}
    for backend in ('legacy','optimized'):
        bc=replace(c,training_options={**c.training_options,'performance_mode':backend});records={};values=[];norm={'mean':0.,'scale':1.};times=[.237139+.071371*(i%8) for i in range(c.batch_size)]
        items=[example(bc,'train',i) for i in range(c.batch_size)];cache=AnalysisCache()
        def batch(items,cache):
            out=[]
            for i,item in enumerate(items):
                t=times[i];out.append(state_samples(item,item['context'],t,bc).numpy())
                for j in ('u','v') if item['family']=='wave' else ('u',):
                    q,y,w,_=grouped_queries(bc,[item_quantity(item,j)],torch.Generator().manual_seed(71+i),norm,cache,[t]);out.extend([q.numpy(),y.numpy(),w.numpy()])
            return out
        for label in ('cold','warm'):
            elapsed=[]
            if label=='warm':batch(items,cache)
            for _ in range(repeats):
                tick=time.perf_counter()
                if label=='cold':items=[example(bc,'train',i) for i in range(c.batch_size)];cache=AnalysisCache()
                values=batch(items,cache);elapsed.append(time.perf_counter()-tick)
            records[label]=distribution(elapsed)
        fingerprints[backend]=values;result[backend]=records
    errors=[float(np.max(np.abs(a-b))) for a,b in zip(fingerprints['legacy'],fingerprints['optimized'])]
    result['maximum_array_difference']=max(errors);result['arrays_compared']=len(errors)
    for a,b in zip(fingerprints['legacy'],fingerprints['optimized']):np.testing.assert_allclose(a,b,rtol=2e-6,atol=2e-7)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--repeats',type=int,default=8);a=p.parse_args()
    c=CoordinateConfig(**json.loads(Path(a.config).read_text())).validate();torch.set_num_threads(c.runtime.get('threads',2));r=run(c,a.repeats);write_json(a.output,r);print(json.dumps({k:v for k,v in r.items() if k!='config'},indent=2))
