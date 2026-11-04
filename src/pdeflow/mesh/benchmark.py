"""Disposable actual-candidate/objective preflight; never changes a requested workload."""
from pathlib import Path
import tempfile,time,statistics,shutil,uuid,traceback
from pdeflow.runtime import digest,read_json,write_json
from .contracts import MeshConfig
from .training import train_stage
from .provenance import scientific_source


def profile(c,local_root,updates=8,warmup=2,*,max_seconds=600):
    if type(updates)!=int or not 7<=updates<=32 or updates<warmup+5 or warmup<1:raise ValueError('At least1 warmup and5 measured updates required')
    c.validate();Path(local_root).mkdir(parents=True,exist_ok=True)
    if not 1<=max_seconds<=3600:raise ValueError('Benchmark timeout must be 1..3600 seconds per stage')
    from .settings import resolved,summary
    from pdeflow.colab import hardware
    from pdeflow.coordinate.random_state import capture,restore
    import torch
    before=capture();cpu=torch.get_rng_state();cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    report={'requested_config':digest(c.to_dict()),'scientific_source':scientific_source()['sha256'],'hardware':hardware(c.runtime['device']),'warmup_updates':warmup,'measured_updates':updates-warmup,'stages':{},'success':False,'training_settings':summary(c),'scope':'Same candidate, architecture, points, objective, effective batch, LR, original schedule horizon/warmup and precision. Disposable successful-update limit; one end validation/checkpoint. No accuracy result.'}
    try:
        with tempfile.TemporaryDirectory(prefix='mesh-preflight-',dir=local_root) as directory:
            dependency=None
            for stage in c.kind['stages']:
                started=time.monotonic()
                try:
                    record=train_stage(c,directory,stage,dependency,benchmark_updates=updates,should_stop=lambda:time.monotonic()-started>max_seconds)
                    if record['progress']!=updates:raise TimeoutError('Benchmark time limit reached before enough successful updates; reduce benchmark scope explicitly or increase max_seconds (not the production workload)')
                except BaseException:
                    original_trace=traceback.format_exc()
                    try:
                        kept=Path(local_root)/('mesh-preflight-failed-'+uuid.uuid4().hex);shutil.copytree(directory,kept)
                        from pdeflow.source_bundle import package
                        identity=package(kept/'source/source.zip');identity['scientific_source']=scientific_source();write_json(kept/'source/identity.json',identity);write_json(kept/'preflight-error.json',{'requested_config':c.to_dict(),'profile':report,'stage':stage,'traceback':original_trace,'scope':'Disposable short preflight failed; original production run unchanged'})
                        print('Preflight failure diagnostics retained:',kept,flush=True)
                    except Exception as save_error:print('Preflight diagnostic retention also failed; original error retained:',save_error,flush=True)
                    raise
                rows=read_json(Path(directory)/'stages'/stage/'history.json')
                boundary=next(i for i,r in enumerate(rows) if r['successful_updates']==warmup and not r['optimizer_update_skipped'])+1
                measured=rows[boundary:];successful=[r for r in measured if not r['optimizer_update_skipped']]
                seconds=[r['step_seconds'] for r in successful];reference=[r['reference_host_seconds'] for r in successful]
                wall=sum(r['end_to_end_seconds'] for r in measured)
                validation=sum(r.get('validation_seconds',0.) for r in measured)
                # Include scaler retries and host bookkeeping; exclude the one forced end validation from steady throughput.
                steady=wall-validation
                settings=resolved(c,stage);batch=settings['effective_batch'];median=statistics.median(seconds)
                checkpoint_seconds=max(0.,record['training_seconds']-record['setup_seconds']-sum(r['end_to_end_seconds'] for r in rows))
                update_seconds=steady/len(successful)
                estimate=update_seconds*c.budgets[stage]+record['setup_seconds']+validation*__import__('math').ceil(c.budgets[stage]/c.validation_every)+checkpoint_seconds*__import__('math').ceil(c.budgets[stage]/c.checkpoint_every)
                report['stages'][stage]={'resolved_training_settings':settings,'warmup_rows':rows[:boundary],
                    'median_seconds_per_successful_update':median,'minimum':min(seconds),'maximum':max(seconds),
                    'p90':sorted(seconds)[min(len(seconds)-1,int(.9*len(seconds)))],
                    'end_to_end_seconds_per_successful_update':wall/len(successful),
                    'steady_seconds_per_successful_update_including_skips':update_seconds,
                    'examples_per_second':len(successful)*batch/steady,
                    'end_to_end_examples_per_second':len(successful)*batch/wall,
                    'median_reference_host_seconds':statistics.median(reference),
                    'cold_first_update_seconds':rows[0]['step_seconds'],'setup_seconds':record['setup_seconds'],
                    'training_wall_seconds':record['training_seconds'],'memory':record['memory'],
                    'actual_forward_microbatch':successful[0]['actual_forward_microbatch'],
                    'successful_updates':record['successful_updates'],'attempted_updates':record['attempted_updates'],
                    'skipped_updates':record['skipped_updates'],'measured_successful_updates':len(successful),
                    'measured_skipped_updates':sum(r['optimizer_update_skipped'] for r in measured),
                    'validation_seconds':validation,'final_checkpoint_and_teardown_seconds':checkpoint_seconds,
                    'fgn_members_per_example':2 if stage=='stochastic' and c.architecture=='mesh_m3_fgn_v1' else 0,
                    'estimated_total_training_seconds':estimate,
                    'estimate_scope':'Short-sample projection, including observed skip rate, setup and cadence-amortized validation/checkpoint costs. Not a completion promise; Drive sync/export excluded.',
                    'rough_training_seconds_excluding_setup_evaluation_export':update_seconds*c.budgets[stage],
                    'measured_rows':measured}
                dependency=Path(directory)/'stages'/stage/'resume.pt'
            report['success']=True
            report['estimated_total_training_seconds']=sum(s['estimated_total_training_seconds'] for s in report['stages'].values())
    finally:
        restore(before);torch.set_rng_state(cpu)
        if cuda:torch.cuda.set_rng_state_all(cuda)
    return report


def microbatch_benchmark(c, local_root, candidates=(1,2,4,8), updates=8, warmup=2, *, max_seconds=600):
    """Explicit bounded sweep. No automatic choice, host-RAM heuristic or session mutation."""
    from dataclasses import replace
    from .settings import resolved
    if not candidates or len(candidates)>8 or len(set(candidates))!=len(candidates):raise ValueError('Provide 1..8 distinct microbatch candidates')
    for value in candidates:
        if type(value)!=int or value<1 or any(value>resolved(c,s)['effective_batch'] for s in c.kind['stages']):
            raise ValueError('Every candidate must fit every active stage effective batch; change effective batch explicitly first')
    output=Path(local_root)/('microbatch-benchmark-'+uuid.uuid4().hex);output.mkdir(parents=True)
    result={'path':str(output/'report.json'),'requested_config':c.to_dict(),'policy':'Only microbatch changes. LR, effective batch, budgets, schedule, scientific objective and FGN members retained. Apply a measured choice explicitly to a NEW run; not an optimal-batch claim.','candidates':[]}
    for micro in candidates:
        overrides={s:{**v,'microbatch':micro} for s,v in c.stage_settings.items()}
        candidate=replace(c,microbatch=micro,stage_settings=overrides).validate()
        try:
            report=profile(candidate,output,updates,warmup,max_seconds=max_seconds)
            result['candidates'].append({'microbatch':micro,'state':'completed','report':report})
        except (KeyboardInterrupt,SystemExit):raise
        except Exception as error:
            result['candidates'].append({'microbatch':micro,'state':'failed','error':str(error),'traceback':traceback.format_exc()})
            # Release transient allocations after OOM; never substitute a smaller workload.
            import torch,gc
            gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()
        write_json(output/'report.json',result)
    return result


if __name__ == '__main__':
    import argparse,json
    parser=argparse.ArgumentParser(description='Bounded real mesh microbatch sweep; never applies a winner')
    parser.add_argument('--config',required=True,help='Exact MeshConfig JSON, or UI training plan JSON containing config')
    parser.add_argument('--output',required=True)
    parser.add_argument('--microbatches',default='1,2,4,8')
    parser.add_argument('--updates',type=int,default=8)
    parser.add_argument('--warmup',type=int,default=2)
    parser.add_argument('--max-seconds',type=int,default=600,help='Per-stage optimizer-boundary time limit, excluding setup')
    args=parser.parse_args();raw=read_json(args.config);config=MeshConfig(**raw.get('config',raw)).validate()
    report=microbatch_benchmark(config,args.output,[int(v.strip()) for v in args.microbatches.split(',')],args.updates,args.warmup,max_seconds=args.max_seconds)
    print(json.dumps(report,indent=2))
    if any(r['state']!='completed' for r in report['candidates']):raise SystemExit(1)
