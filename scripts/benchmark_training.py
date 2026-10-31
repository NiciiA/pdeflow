"""Bounded SAME-config A1/A2/B timing, usable against before/after source trees.

Run this script in a fresh subprocess per source: --source controls imports,
not the architecture. It never starts Session.run or overrides a quality gate.
Disposable random models are used solely to profile each stage's execution.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--updates',type=int,default=10)
    p.add_argument('--warmup',type=int,default=2)
    args=p.parse_args()
    if args.updates<args.warmup+5 or args.warmup<1:raise ValueError('At least1 warmup and5 measured updates required')
    args.output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(args.source.resolve()/'src'))
    import torch
    import numpy as np
    from dataclasses import replace
    from pdeflow.coordinate.contracts import CoordinateConfig
    from pdeflow.coordinate.training import train,read_checkpoint
    from pdeflow.coordinate.optimization import scientific_source
    from pdeflow.runtime import RuntimeSpec,synchronize,write_json,file_digest
    requested=CoordinateConfig(**json.loads(args.config.read_text())).validate()
    # An identical explicit precision is ESSENTIAL for reference-path speedups.
    # Native precision-policy auto is checked separately by the updated preflight.
    if requested.training_options.get('precision','fp32')=='auto':raise ValueError('Use explicit fp16 on T4, or fp32 for the CPU control, in both benchmark configs')
    c=replace(requested,ae_steps=args.updates,evolution_steps=args.updates,flow_steps=args.updates,checkpoint_every=args.updates,
              training_options={**requested.training_options,'profile_components':True,'validation_every':args.updates})
    device,_=RuntimeSpec(**c.runtime).resolve()
    def stats(values):
        a=np.array(values);return {'count':len(a),'median':float(np.median(a)),'mean':float(a.mean()),'std':float(a.std()),'p10':float(np.quantile(a,.1)),'p90':float(np.quantile(a,.9)),'min':float(a.min()),'max':float(a.max())}
    result={'requested_config':requested.to_dict(),'executed_config':c.to_dict(),'source':scientific_source(),'config_file_sha256':file_digest(args.config),'warmup_updates':args.warmup,'stages':{},'device':str(device),'torch':torch.__version__,
            'meaning':'Fresh random benchmark models; no quality claim or production gate bypass. Only update budget, checkpoint/validation frequency and instrumentation differ from requested training. No architecture, point count, loss, family, observation or physical-time reduction.'}
    if device.type=='cuda':result['gpu']={'name':torch.cuda.get_device_name(device),'capability':list(torch.cuda.get_device_capability(device))}
    write_json(args.output/'benchmark.json',result)
    for stage,dep in [('autoencoder',None),('evolution',args.output/'autoencoder/autoencoder.pt'),('flow',args.output/'evolution/evolution.pt')]:
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        synchronize(device);tick=time.perf_counter();r=train(c.to_dict(),args.output/stage,stage,dep);synchronize(device);wall=time.perf_counter()-tick
        cp,_=read_checkpoint(args.output/stage/('coordinate.pt' if stage=='flow' else stage+'.pt'));rows=cp['history'];measured=rows[args.warmup:]
        update=stats([x['step_seconds'] for x in measured]);cache=args.output/stage/'latent-targets'
        result['stages'][stage]={'update_seconds':update,'examples_per_second':c.batch_size/update['mean'],'end_to_end_update_seconds':stats([x['end_to_end_seconds'] for x in measured]) if all('end_to_end_seconds' in x for x in measured) else None,
          'total_stage_wall_seconds':wall,'amortized_all_in_seconds_per_update':wall/args.updates,'setup_seconds':r.get('setup_seconds'),'final_persistence_and_calibration_seconds':r.get('final_persistence_and_calibration_seconds'),'dataset_preparation_seconds':json.loads((args.output/'autoencoder/dataset.json').read_text()).get('preparation_seconds') if stage=='autoencoder' else 0,
          'warmup_rows':rows[:args.warmup],'measured_rows':measured,'peak_memory':r.get('memory'),'cache_bytes':sum(f.stat().st_size for f in cache.rglob('*') if f.is_file()),'last_frozen_cache':rows[-1].get('frozen_cache'),'cache_population_and_io_seconds':sum(sum((x.get('component_timings') or {}).get('host_exclusive_seconds',{}).get(k,0.) for k in ('frozen_cache_store','target_cache_io')) for x in rows) if all('host_exclusive_seconds' in (x.get('component_timings') or {}) for x in rows) else None,
          'validation_seconds':sum(x.get('validation_seconds',0) for x in rows),'persistence_seconds':sum(x.get('persistence_seconds',0) for x in rows),'all_gradients_finite':all(x['gradients_finite'] if 'gradients_finite' in x else np.isfinite(x['gradient_norm']) for x in rows),'all_losses_finite':all(np.isfinite(x['training_loss']) for x in rows),'skipped_updates':sum(x.get('optimizer_update_skipped',False) for x in rows),
          'cache_scope':'Fresh stage/cache; warmup is NOT synonymous with cache warm. Actual cache hits/misses retained. Full dataset sweep not performed.'}
        write_json(args.output/'benchmark.json',result)
        print(stage,json.dumps({k:result['stages'][stage][k] for k in ('update_seconds','examples_per_second','total_stage_wall_seconds','cache_bytes')}),flush=True)


if __name__=='__main__':main()
