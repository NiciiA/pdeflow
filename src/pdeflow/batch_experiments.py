"""Disposable, cancellable stage batch experiments shared by UI and notebook."""
from dataclasses import replace
from pathlib import Path
import argparse,json,math,os,signal,subprocess,sys,time,uuid,traceback
from pdeflow.runtime import read_json,write_json,digest,RuntimeSpec
from pdeflow.training_settings import STAGES,TOTALS,resolve_batches,configure,positive

CANDIDATES=[8,16,32,64,128,256,512,1024]
ACTIVE=('queued','running','stopping')

def science_identity(c):
    from .coordinate.optimization import model_contract,options,precision,scientific_source
    device,_=RuntimeSpec(**c.runtime).resolve();policy,_=precision(options(c)['precision'],device)
    from .colab import hardware
    host=hardware(str(device));gpu=host['gpu']
    identity={k:gpu[k] for k in ('name','total_bytes','capability')} if gpu else {'backend':str(device),'platform':host['platform'],'host_ram_bytes':host['host_ram_bytes']}
    return digest({'hardware':identity,'model_data_loss':model_contract(c),'optimizer':{s:options(c)[s] for s in STAGES},'source':scientific_source()['sha256'],'precision':policy,'device':str(device)})


def plan(c,mode='efficiency',stages=None,candidates='auto',effective_batch=None,effective_batches=None,budget_policy='examples',budget=64,warmup=2,measured=5,max_candidates=3,max_seconds=120,headroom=.15,cache_conditions=None,observed_seconds=None):
    stages=stages or ['autoencoder'];conditions=cache_conditions or ['cold','warm']
    if mode not in ('efficiency','learning'):raise ValueError('Experiment mode: efficiency or learning')
    if not stages or set(stages)-set(STAGES):raise ValueError('Select A1/A2/B stages by autoencoder/evolution/flow')
    if 'warm' in conditions and 'cold' not in conditions:raise ValueError('Warm replay requires a completed cold trial; choose cold or cold+warm')
    if set(conditions)-{'cold','warm'} or not conditions:raise ValueError('Cache conditions: cold and/or warm')
    if type(warmup)!=int or warmup<1 or type(measured)!=int or measured<5:raise ValueError('Use >=1 warmup and >=5 measured attempts')
    if not 0<=headroom<.8 or not 1<=max_candidates<=32 or not 1<=max_seconds<=86400:raise ValueError('Invalid headroom/candidate/runtime bounds')
    fixed=positive(effective_batch or c.batch_size,'Effective batch');automatic=candidates=='auto'
    if automatic:
        start=min(fixed,min(resolve_batches(c)[s]['microbatch'] for s in stages));sizes=[start]
        sizes+= [v for v in CANDIDATES if v>start and v<=fixed]
        sizes=sizes[:max_candidates]
    else:
        if not isinstance(candidates,list) or not candidates:raise ValueError('Candidates must be auto or a nonempty editable list')
        sizes=sorted(set(positive(v,'GPU microbatch candidate') for v in candidates))
        if len(sizes)>32:raise ValueError('Select at most32 practical candidates')
        if max(sizes)>fixed and mode=='efficiency':raise ValueError('Microbatch candidates must not exceed the fixed effective batch')
    effective_values=[fixed] if mode=='efficiency' else sorted(set(positive(x,'Effective-batch candidate') for x in (effective_batches or [c.batch_size,c.batch_size*2])))
    if mode=='learning' and budget_policy=='examples':positive(budget,'Examples budget')
    if mode=='learning' and budget_policy not in ('examples','wall','updates'):raise ValueError('Learning budget: examples, wall or updates')
    if mode=='learning' and (type(budget) not in (int,float) or not math.isfinite(budget) or budget<=0):raise ValueError('Learning budget must be finite positive')
    rows=[]
    for stage in stages:
        for effective in effective_values:
            values=sizes if mode=='efficiency' else [min(resolve_batches(c)[stage]['microbatch'],effective)]
            for micro in values:
                updates=warmup+measured
                if mode=='learning' and budget_policy=='examples':
                    updates=math.ceil(budget/effective)
                    if updates<warmup+measured:raise ValueError('Examples budget too small for warmup+measured iterations at every effective batch')
                elif mode=='learning' and budget_policy=='updates':
                    updates=positive(budget,'Fixed optimizer update budget')
                    if updates<warmup+measured:raise ValueError('Update budget too small for measurement')
                elif mode=='learning':updates=10**8
                for condition in conditions:
                    rows.append({'stage':stage,'microbatch':micro,'accumulation':math.ceil(effective/micro),'effective_batch':effective,'updates':updates,'cache':condition,
                      'planned_examples':None if mode=='learning' and budget_policy=='wall' else updates*effective,'status':'pending'})
    estimate={'seconds':None,'runtime_cap_seconds':max_seconds,'meaning':'No measured pilot for this exact job yet. First completed candidate supplies a measured estimate for the remainder; runtime cap/cancellation bound exploration. Includes requested calibration/anchors, not a promise to finish every candidate.'}
    if observed_seconds and all(stage in observed_seconds for stage in stages):
        for value in observed_seconds.values():
            if type(value) not in (int,float) or not math.isfinite(value) or value<=0:raise ValueError('Observed seconds must be finite positive measurements')
        predicted=sum((budget if mode=='learning' and budget_policy=='wall' else observed_seconds[r['stage']]*r['updates']*r['effective_batch']/c.batch_size) for r in rows)
        estimate.update(seconds=predicted,meaning='Rough prior measured baseline extrapolation; assumes no batching speedup. Setup/anchors and validation/export additional; kernel scaling can change. Actual pilot replaces this estimate.')
    return {'mode':mode,'stages':stages,'candidates':sizes,'automatic':automatic,'rows':rows,'warmup':warmup,'measured':measured,'budget_policy':budget_policy if mode=='learning' else 'fixed ordered examples/attempts','budget':budget if mode=='learning' else warmup+measured,'max_seconds':max_seconds,'headroom':headroom,
      'science_identity':science_identity(c),'requested_config':c.to_dict(),'estimate':estimate,
      'meaning':'Efficiency holds model, points, objective, precision, effective batch and draw order fixed; candidates with any skipped steps are not recommended. Learning deliberately changes effective batch and optimization; equal updates are unequal data exposure. No automatic LR scaling.'}


def recommend(rows,headroom=.15):
    groups={}
    for r in rows:
        if r.get('status')!='completed' or r.get('skipped_updates',0) or not r.get('numerically_finite',False):continue
        peak=max(r.get('peak_allocated') or 0,r.get('peak_reserved') or 0);total=r.get('gpu_total_bytes')
        if total and peak>total*(1-headroom):continue
        groups.setdefault((r['stage'],r['microbatch'],r['effective_batch']),[]).append(r)
    by_stage={}
    for (stage,micro,effective),values in groups.items():
        all_conditions=[r for r in rows if r['stage']==stage and r['microbatch']==micro and r['effective_batch']==effective]
        if any(r.get('status')!='completed' or r.get('skipped_updates',0) or not r.get('numerically_finite',False) for r in all_conditions):continue
        # Conservative across cold/replay conditions, never the fastest cached row.
        score=min(v['examples_per_second'] for v in values);cv=max(v['std_seconds']/max(v['mean_seconds'],1e-12) for v in values)
        if cv>.5:continue
        by_stage.setdefault(stage,[]).append({'microbatch':micro,'effective_batch':effective,'score':score,'cv':cv,'peak':max((v.get('peak_allocated') or 0) for v in values)})
    result={}
    for stage,values in by_stage.items():
        best=max(v['score'] for v in values);close=[v for v in values if v['score']>=best*.95]
        chosen=min(close,key=lambda v:(v['peak'],v['microbatch']))
        result[stage]={**chosen,'reason':'Measured throughput/stability; CUDA headroom filter where peak/total memory are available (CPU/MPS margin is unqualified); candidates within5% prefer lower memory/smaller microbatch. Skipped updates disqualify. Thresholds are a conservative heuristic, not a statistical significance claim.'}
    return result


def start(c,root,background=True,**kwargs):
    p=plan(c,**kwargs);root=Path(root);root.mkdir(parents=True,exist_ok=False)
    record={'id':root.name,'state':'queued','plan':p,'rows':p['rows'],'recommendation':{},'created_unix':time.time()};write_json(root/'experiment.json',record)
    if background:
        with (root/'worker.log').open('w') as log:
            process=subprocess.Popen([sys.executable,'-m','pdeflow.batch_experiments','run',str(root.resolve())],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        record['pid']=process.pid;write_json(root/'experiment.json',record)
    else:run(root)
    return status(root)


def status(root):
    root=Path(root);r=read_json(root/'experiment.json')
    if r['state'] in ACTIVE and r.get('pid'):
        try:os.kill(r['pid'],0)
        except ProcessLookupError:
            r.update(state='interrupted',error='Benchmark worker exited; partial results retained')
            for row in r['rows']:
                if row['status'] in ('pending','running'):row.update(status='interrupted',reason='Worker exited before measurement completed')
            write_json(root/'experiment.json',r)
    r['stop_requested']=(root/'stop-request.json').exists()
    r['result_state']='benchmark result; not applied and no actual run advanced'
    return r


def stop(root):
    write_json(Path(root)/'stop-request.json',{'requested_unix':time.time()});return status(root)


def selected_settings(c,record,selections=None,recommended=False):
    if record['plan']['science_identity']!=science_identity(c):raise ValueError('Benchmark is stale for this hardware/source/model/data/loss/optimizer; rerun or choose explicit manual settings')
    if recommended:
        if record['plan']['mode']!='efficiency':raise ValueError('Learning experiments have no automatic optimization recommendation; review validation and select manually')
        chosen=record.get('recommendation',{})
    else:
        chosen={}
        for stage,selection in (selections or {}).items():
            micro=selection.get('microbatch') if isinstance(selection,dict) else selection
            effective=selection.get('effective_batch') if isinstance(selection,dict) else None
            matches=[r for r in record['rows'] if r['stage']==stage and r['microbatch']==micro and (effective is None or r['effective_batch']==effective) and r['status']=='completed' and not r.get('skipped_updates')]
            if not matches:raise ValueError('Selected benchmark candidate did not complete stably')
            chosen[stage]={'microbatch':micro,'effective_batch':matches[0]['effective_batch']}
    if not chosen:raise ValueError('No eligible selected/recommended batch settings')
    for stage,value in chosen.items():
        conditions=[r for r in record['rows'] if r['stage']==stage and r['microbatch']==value['microbatch'] and r['effective_batch']==value['effective_batch']]
        if not conditions or any(r['status']!='completed' or r.get('skipped_updates',0) or not r.get('numerically_finite',False) for r in conditions):raise ValueError('Selected settings did not complete all requested cache conditions stably')
    values={x['effective_batch'] for x in chosen.values()}
    if len(values)!=1:raise ValueError('Selected stages must share an effective batch')
    effective=values.pop();base=c.training_options.get('microbatch',c.batch_size)
    if isinstance(base,int) and base>effective:base='auto'
    return {'effective_batch':effective,'microbatch':base,'stages':{**c.training_options.get('stage_batches',{}),**{s:{'microbatch':x['microbatch']} for s,x in chosen.items()}}}


def failure(error,root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    trace=traceback.format_exc();(root/'error-traceback.txt').write_text(trace)
    print(trace,file=sys.stderr,flush=True)
    result={'status':'oom' if isinstance(error,__import__('torch').OutOfMemoryError) else 'error',
            'error':str(error),'error_type':type(error).__name__,'traceback':trace,
            'traceback_path':str(root/'error-traceback.txt')}
    if (root/'run.json').exists():
        record=read_json(root/'run.json')
        for key in ('attempted_updates','successful_optimizer_updates','skipped_updates','history','normalizations'):
            if key in record:result[key]=record[key]
        result['successful_updates']=record.get('successful_optimizer_updates',0)
        rows=record.get('history',[])
        result['scales']=[r.get('scaler_scale') for r in rows]
        result['numerically_finite']=all(math.isfinite(r['training_loss']) and r['gradients_finite'] for r in rows) if rows else None
        result['nonfinite_gradient_attempts']=sum(not r['gradients_finite'] for r in rows)
    return result


def failure_details(record):
    """Plain text shared by the notebook; show failures even before any trial runs."""
    lines=[]
    if record.get('error'):lines.append(f"FAILED: {record.get('error_type','Error')}: {record['error']}")
    for stage,anchor in record.get('anchors',{}).items():
        lines.append(f"Anchor {stage}: {anchor.get('status')}; attempted={anchor.get('attempted_updates')}, successful={anchor.get('successful_updates')}, skipped={anchor.get('skipped_updates')}; scales={anchor.get('scales', [r.get('scaler_scale') for r in anchor.get('history',[])])}")
        if anchor.get('error'):lines.append(anchor['error'])
        invalid=[r['overflow_diagnostics'] for r in anchor.get('history',[]) if 'overflow_diagnostics' in r]
        if invalid:lines.append('Last overflow diagnostics: '+json.dumps(invalid[-1],sort_keys=True))
        if anchor.get('traceback'):lines.append(anchor['traceback'])
        if anchor.get('process_log'):lines.append('Process log: '+anchor['process_log'])
    for row in record.get('rows',[]):
        if row.get('error'):
            lines.append(f"{row['stage']} microbatch={row['microbatch']}: {row['error']}")
            if row.get('traceback'):lines.append(row['traceback'])
    if record.get('traceback'):lines.append(record['traceback'])
    return '\n'.join(lines)


def worker(job):
    try:_worker(job)
    except Exception as error:write_json(Path(job['root'])/'result.json',failure(error,job['root']))


def _worker(job):
    import torch,numpy as np
    from .coordinate.contracts import CoordinateConfig
    from .coordinate.training import train,read_checkpoint
    from .coordinate.profiling import distribution
    c=CoordinateConfig(**job['config']).validate();root=Path(job['root']);stage=job['stage'];begin=time.perf_counter();limit=job.get('wall_seconds')
    if job.get('cache_from'):
        import pdeflow.coordinate.engine as engine
        original=engine.FrozenTargets
        def cache(*args,**kwargs):
            target=args[3] if len(args)>3 else kwargs.get('root')
            if target:
                source=Path(job['cache_from'])/'latent-targets'/Path(target).name
                if source.exists():
                    import shutil
                    shutil.copytree(source,target,dirs_exist_ok=True)
            return original(*args,**kwargs)
        engine.FrozenTargets=cache
    try:
        record=train(c.to_dict(),root,stage,job.get('dependency'),stop_after=job.get('attempts'),should_stop=lambda:(Path(job['stop_file']).exists() or limit is not None and time.perf_counter()-begin>=limit))
        cp,_=read_checkpoint(root/'resume.pt');rows=cp['history'];measured=rows[job['warmup']:]
        if not measured:result={'status':'insufficient_measurements','error':'Budget/cancellation ended before enough measurements'}
        else:
            timing=distribution([r['end_to_end_seconds'] for r in measured]);skips=sum(r['optimizer_update_skipped'] for r in rows);wall=sum(r['end_to_end_seconds'] for r in measured)
            total=torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None
            result={'status':'completed' if len(measured)>=5 else 'insufficient_measurements','median_seconds':timing['median'],'mean_seconds':timing['mean'],'std_seconds':timing['std'],'distribution':timing,
              'examples_per_second':sum(c.batch_size for _ in measured)/wall,'successful_updates_per_second':sum(not r['optimizer_update_skipped'] for r in measured)/wall,'skipped_updates':skips,'numerically_finite':all(math.isfinite(r['training_loss']) and r['gradients_finite'] for r in rows),'nonfinite_gradient_attempts':sum(not r['gradients_finite'] for r in rows),'overflow_skips_handled':all(r['gradients_finite'] or r['optimizer_update_skipped'] for r in rows),
              'peak_allocated':record['memory'].get('cuda_peak_allocated'),'peak_reserved':record['memory'].get('cuda_peak_reserved'),'peak_host_rss':record['memory'].get('process_peak_host_rss_bytes'),'gpu_total_bytes':total,
              'actual_forward_sizes':sorted({n for r in rows for n in r['forward_microbatch_sizes']}),'attempted_updates':cp['attempted_updates'],'successful_updates':cp['progress'],'examples_processed':cp['effective_examples_processed'],
              'setup_seconds':record['setup_seconds'],'first_cold_update_seconds':rows[0]['end_to_end_seconds'],'warmup_seconds':sum(r['end_to_end_seconds'] for r in rows[:job['warmup']]),'final_persistence_seconds':record['final_persistence_and_calibration_seconds'],
              'preparation':rows[-1]['frozen_preparation'],'validation_seconds':sum(r.get('validation_seconds',0) for r in rows),'checkpoint_seconds':sum(r.get('persistence_seconds',0) for r in rows),'scales':[r['scaler_scale'] for r in rows],'history':rows,'validation_loss':rows[-1].get('validation_loss'),'all_in_seconds':time.perf_counter()-begin}
        if job.get('role')=='anchor':
            result.update(status='completed' if cp['completed'] and cp['progress']==job['successful_updates'] else 'insufficient_successful_updates',role='anchor',required_successful_updates=job['successful_updates'],maximum_attempts=job['attempts'])
            if result['status']!='completed':result['error']='Anchor stopped before required successful optimizer updates; inspect skips, cancellation and runtime cap'
    except Exception as error:result=failure(error,root)
    if job.get('learning') and result.get('status') in ('completed','insufficient_measurements') and (root/'resume.pt').exists():
        from .coordinate import validation
        validation_started=time.perf_counter()
        try:
            subset=min(c.validation_count,__import__('pdeflow.coordinate.optimization',fromlist=['options']).options(c)['validation_subset'])
            if stage=='flow':report=validation.flow(root/'resume.pt',root/'learning-validation.json',c.runtime['device'],subset=subset,counts=[max(1,c.observation_counts[0])],noises=[c.noise_range[0]],particles=1,steps=2,resolution=16,allow_partial=True)
            else:report=getattr(validation,'representation' if stage=='autoencoder' else 'evolution')(root/'resume.pt',root/'learning-validation.json',c.runtime['device'],subset=subset,allow_partial=True)
            result['learning_validation']=report;result['learning_validation_scope']='Same held-out validation cases; bounded16x16 physical evaluation, no test selection or accuracy claim.'
        except (ValueError,RuntimeError) as error:result['learning_validation_error']=str(error)
        result['learning_validation_seconds']=time.perf_counter()-validation_started
    write_json(root/'result.json',result)


def run(root):
    from .coordinate.contracts import CoordinateConfig
    from .coordinate.optimization import options,precision
    root=Path(root);r=read_json(root/'experiment.json');p=r['plan'];c=CoordinateConfig(**p['requested_config']).validate();o=options(c);device,_=RuntimeSpec(**c.runtime).resolve();policy,_=precision(o['precision'],device)
    started=time.time();r.update(state='running',pid=os.getpid());write_json(root/'experiment.json',r)
    def cancelled():return (root/'stop-request.json').exists() or time.time()-started>=p['max_seconds']
    def launch(job,path):
        path.mkdir(parents=True,exist_ok=True);job['root']=str(path.resolve());job['stop_file']=str((root/'stop-request.json').resolve());jobfile=path/'job.json';write_json(jobfile,job)
        # Trainer requires an empty root; keep the subprocess control file outside it.
        execution=path/'stage';job['root']=str(execution.resolve());write_json(jobfile,job)
        with (path/'process.log').open('w') as log:
            child=subprocess.Popen([sys.executable,'-m','pdeflow.batch_experiments','worker',str(jobfile.resolve())],stdout=log,stderr=subprocess.STDOUT)
            interrupted=False;deadline=None
            while child.poll() is None:
                if cancelled() and not interrupted:
                    child.send_signal(signal.SIGINT);interrupted=True;deadline=time.time()+10
                if interrupted and time.time()>deadline:child.terminate();break
                time.sleep(.15)
            child.wait()
        result=read_json(execution/'result.json') if (execution/'result.json').exists() else {'status':'cancelled' if interrupted else 'error','error':'Candidate process ended without result; exit='+str(child.returncode),'traceback':(path/'process.log').read_text()[-262144:]}
        result['process_log']=str(path/'process.log');return result
    try:
        dependencies={};anchor_opts={**c.training_options,'precision':policy,'execution':'batched'}
        anchor=replace(c,ae_steps=2,evolution_steps=2,flow_steps=2,training_options=anchor_opts)
        needed=['autoencoder'] if 'evolution' in p['stages'] else []
        if 'flow' in p['stages']:needed=['autoencoder','evolution']
        for stage in needed:
            if cancelled():break
            dep=dependencies.get('autoencoder') if stage=='evolution' else None
            # Before N successes, at most N*(overflow_limit-1) skips can occur
            # without hitting the trainer's existing consecutive-overflow failure.
            # Keep one scaler alive; stop on successes, cancel, wall cap or this bound.
            result=launch({'config':anchor.to_dict(),'stage':stage,'dependency':dep,'warmup':0,'role':'anchor','successful_updates':2,'attempts':2*o['max_consecutive_overflows']},root/'anchors'/stage)
            r.setdefault('anchors',{})[stage]=result;write_json(root/'experiment.json',r)
            if cancelled():break
            path=root/'anchors'/stage/'stage'/('autoencoder.pt' if stage=='autoencoder' else 'evolution.pt')
            if result['status']!='completed' or not path.exists():
                raise ValueError(f"Disposable {stage} anchor failed: {result.get('error',result['status'])}; successful={result.get('successful_updates',0)}/2, attempted={result.get('attempted_updates')}. See anchors in this report and {result['process_log']}")
            dependencies[stage]=str(path.resolve())
        failed={};cold={}
        for i,row in enumerate(r['rows']):
            if cancelled():break
            stage=row['stage'];signature=(stage,row['effective_batch'])
            if p['automatic'] and failed.get(signature,10**30)<=row['microbatch']:row.update(status='skipped',reason='At/above confirmed OOM/headroom boundary');continue
            cc,_=configure(c,{'effective_batch':row['effective_batch'],'microbatch':row['microbatch'],'stages':{stage:row['microbatch']}})
            extra={s:{**cc.training_options.get(s,{}),'horizon':o[s]['horizon']} for s in STAGES}
            cc=replace(cc,**{TOTALS[stage]:row['updates']},training_options={**cc.training_options,**extra,'precision':policy,'profile_components':True,'validation_every':row['updates']})
            # Flow minimum is a config invariant even while benchmarking another stage.
            cc=replace(cc,flow_steps=max(2,cc.flow_steps)).validate();key=(stage,row['effective_batch'],row['microbatch'])
            job={'config':cc.to_dict(),'stage':stage,'dependency':dependencies.get('autoencoder' if stage=='evolution' else 'evolution') if stage!='autoencoder' else None,'warmup':p['warmup'],'attempts':row['updates'] if p['mode']=='efficiency' or p['budget_policy']=='examples' else None,'learning':p['mode']=='learning',
              'wall_seconds':p['budget'] if p['mode']=='learning' and p['budget_policy']=='wall' else None,'cache_from':cold.get(key) if row['cache']=='warm' else None}
            row.update(status='running');write_json(root/'experiment.json',r)
            result=launch(job,root/f'candidate-{i:03d}');row.update(result)
            if row['cache']=='cold':cold[key]=str((root/f'candidate-{i:03d}'/'stage').resolve())
            peak=max(row.get('peak_allocated') or 0,row.get('peak_reserved') or 0);total=row.get('gpu_total_bytes')
            if row['status']=='oom' or total and peak>total*(1-p['headroom']):failed[signature]=row['microbatch']
            complete=[v for v in r['rows'] if v['status']=='completed']
            if complete:p['estimate']={'seconds':sum(v['all_in_seconds'] for v in complete)/len(complete)*sum(v['status']=='pending' for v in r['rows']),'source':'Measured completed-candidate mean; not memory inference'}
            r['recommendation']=recommend(r['rows'],p['headroom']) if p['mode']=='efficiency' else {};write_json(root/'experiment.json',r)
        for row in r['rows']:
            if row['status'] in ('pending','running'):row.update(status='cancelled',reason='Cancellation/runtime cap; partial results preserved')
        r['state']='cancelled' if cancelled() else 'completed'
    except Exception as error:
        r.update(state='failed',error=str(error),error_type=type(error).__name__,traceback=traceback.format_exc())
        (root/'error-traceback.txt').write_text(r['traceback'])
        for row in r['rows']:
            if row['status'] in ('pending','running'):row.update(status='blocked',reason='Experiment failed before this candidate completed: '+str(error))
    finally:r['elapsed_seconds']=time.time()-started;write_json(root/'experiment.json',r)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['run','worker']);parser.add_argument('path',type=Path);args=parser.parse_args()
    worker(read_json(args.path)) if args.action=='worker' else run(args.path)
