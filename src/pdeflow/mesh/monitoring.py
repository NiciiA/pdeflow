"""Bounded live records and read-only mesh/suite monitoring; no model imports."""
from pathlib import Path
import json,time,os,threading,socket,statistics,html,math
from collections import deque
from pdeflow.runtime import read_json,write_json
from .convergence import attach,successful,compact,recorded_validation,validation_summary


class Reporter:
    def __init__(self,root,c,stage=None,callback=None):
        from .settings import resolved
        settings=resolved(c,stage or c.kind['stages'][0])
        self.root=Path(root);self.started=time.monotonic();self.last=0.;self.callback=callback;self.rows=deque(maxlen=200);self.seen=0
        self.successes=deque(maxlen=c.training_smoothing_window+200)
        self.record={'validation_every':c.validation_every,'validation_protocol':c.validation_protocol,'family_weights':c.family_weights,'training_smoothing_window':c.training_smoothing_window,'validation_history':[],'schema':'mesh_progress_v1','run_id':self.root.name,'architecture':c.architecture,'stage':stage,'device':c.runtime['device'],'resolved_precision':'pending device resolution','autocast_dtype':None,'budget':c.budgets.get(stage),'all_budgets':{s:c.budgets[s] for s in c.kind['stages']},'batch_size':settings['effective_batch'],'microbatch':settings['microbatch'],'accumulation':settings['accumulation'],'learning_rate':settings['learning_rate'],'losses':{'field':c.field_loss,'objective':c.objective,'stages':list(c.kind['stages'])},'successful_updates':0,'attempted_updates':0,'skipped_updates':0,'failed_attempts':0,'pid':os.getpid(),'host':socket.gethostname()}
    def update(self,phase=None,force=False,**values):
        changed=phase is not None and phase!=self.record.get('phase');self.record.update(values)
        if phase is not None:self.record['phase']=phase
        now=time.monotonic()
        if not(force or changed or now-self.last>=5):return
        self.record.update(unix=time.time(),elapsed_seconds=now-self.started,history=list(self.rows),smoothing_history=list(self.successes));self.record=attach(self.record);write_json(self.root/'progress.json',self.record);self.last=now
        if self.callback:self.callback(dict(self.record))
    def restore(self,history,convergence_history,validation_history):
        self.rows.extend(history[-200:]);self.successes.extend(convergence_history)
        self.record['validation_history']=validation_history
    def validation(self,result):
        if not any(r.get('validation_id')==result.get('validation_id') for r in self.record['validation_history']):
            self.record['validation_history'].append(validation_summary(result))
        self.update('validation',force=True)
    def row(self,row):
        row=dict(row);row['monitor_warmup']=self.seen<2;self.seen+=1
        if successful(row):self.successes.append(compact(row))
        self.rows.append(row);recent=list(self.rows)[-25:];timed=[r for r in recent if 'step_seconds' in r and not r.get('monitor_warmup')];wall=sum(r['step_seconds']+r.get('validation_seconds',0)+r.get('persistence_seconds',0) for r in timed);success=sum(not r.get('optimizer_update_skipped',False) and not r.get('failed_attempt',False) for r in timed);rate=success/wall if success>=5 and wall>0 else None
        self.update('training',successful_updates=row['step'],attempted_updates=row['attempted_update'],skipped_updates=row.get('skipped_updates',0),failed_attempts=row.get('failed_attempts',0),learning_rate=row.get('learning_rate'),loss=row.get('training_loss'),moving_average=statistics.fmean(r['training_loss'] for r in recent if r.get('training_loss') is not None) if any(r.get('training_loss') is not None for r in recent) else None,components=row.get('components'),scaler_scale=row.get('scaler_scale'),seconds_per_successful_update=1/rate if rate else None,examples_per_second=self.record['batch_size']*rate if rate else None,eta_stage_seconds=max(0,self.record['budget']-row['step'])/rate if rate else None,eta_note='Recent wall includes measured validation/checkpoint/sync overhead; setup and two warmup attempts excluded; remaining unmeasured validation/save costs are uncertain' if rate else 'Waiting for five measured successful updates after two warmup attempts; setup excluded')


def read_optional(path):
    try:return read_json(path)
    except (OSError,ValueError):return {}


def owner_active(root):
    file=Path(root)/'.run.lock'
    if not file.exists():return False
    import fcntl
    with file.open('r') as handle:
        try:fcntl.flock(handle,fcntl.LOCK_SH|fcntl.LOCK_NB)
        except BlockingIOError:return True
        fcntl.flock(handle,fcntl.LOCK_UN);return False


def status(root):
    """Read a few bounded JSON files; no restore, mkdir, checkpoint load or owner lock."""
    root=Path(root)
    if (root/'.pdeflow-store.json').exists():return stored_status(root)
    suite=read_optional(root/'suite.json')
    if suite:
        active=suite.get('active');child=status(root/'runs'/active) if active else {};order=suite['plan']['order'];states={n:suite.get('candidates',{}).get(n,{}).get('state','current' if n==active else 'pending') for n in order}
        profiles=read_optional(root/'profiles.json');remaining=[name for name in order if states[name]!='completed' and name!=active];estimates=[]
        for name in remaining:
            report=profiles.get(name,{});cfg=suite['plan']['configs'].get(name)
            valid=cfg and report.get('success') and report.get('scientific_source')==suite.get('source',{}).get('sha256')
            estimates.append(sum(part['median_seconds_per_successful_update']*cfg['budgets'][stage]+part.get('setup_seconds',0) for stage,part in report['stages'].items()) if valid else None)
        active_eta=child.get('eta_model_seconds') if active else 0.
        eta=active_eta+sum(estimates) if active_eta is not None and all(v is not None for v in estimates) else None
        note='Provisional whole-suite estimate: pending candidates use preflight rates; unmeasured validation/Drive costs and quality-gate failures remain uncertain' if eta is not None else 'Whole-suite ETA unavailable: active rate or later candidate stages are unmeasured. Active-model estimate shown separately.'
        return {'kind':'suite','root':str(root),'run_id':root.name,'state':suite['state'],'candidates':states,'active':active,'active_model':child,'persistent':read_optional(root/'sync-status.json'),'eta_suite_seconds':eta,'eta_suite_note':note}
    session=read_optional(root/'session.json');r=read_optional(root/'progress.json');stage=r.get('stage') or session.get('stage');old={}
    if not r and stage:old=read_optional(root/'stages'/stage/'run.json');r={**old,'history':old.get('history',[]),'unix':(root/'stages'/stage/'run.json').stat().st_mtime if old else 0}
    if not r:r={'phase':'not started','successful_updates':0,'history':[]}
    age=max(0,time.time()-r.get('unix',time.time()));active=owner_active(root);terminal=session.get('state') in ('failed','completed','interrupted','needs_review');state=session.get('state',r.get('phase','not started')) if terminal else 'active' if active else 'stale / no local owner' if session else 'not started'
    failure=root/'stages'/str(stage)/'failure';latest=read_optional(failure/'latest.json');name=latest.get('directory');error_report=failure/name/'report.json' if name and Path(name).name==name else failure/'report.json'
    sync=read_optional(root/'sync-status.json')
    if sync.get('last_success_unix'):sync['age_seconds']=max(0,time.time()-sync['last_success_unix'])
    budgets=r.get('all_budgets',{});stages=list(budgets);future=stages[stages.index(stage)+1:] if stage in stages else [];profile=read_optional(root/'profile.json');eta=r.get('eta_stage_seconds');extra=[]
    for name in future:
        part=profile.get('stages',{}).get(name,{})
        extra.append(part.get('median_seconds_per_successful_update',0)*budgets[name]+part.get('setup_seconds',0) if part else None)
    model_eta=eta+sum(extra) if eta is not None and all(v is not None for v in extra) else None
    cfg=session.get('config',old.get('config',{}))
    r.setdefault('validation_every',cfg.get('validation_every'));r.setdefault('training_smoothing_window',cfg.get('training_smoothing_window',100))
    vh=r.get('validation_history')
    if vh is None:vh=read_optional(root/'stages'/str(stage)/'validation-history.json') or recorded_validation(r.get('history',[]),stage,read_optional(root/'stages'/str(stage)/'validation.json'))
    r=attach(r,vh)
    return {**r,'eta_model_seconds':model_eta,'eta_model_note':'Active-model ETA; later stages use provisional preflight rates and may stop on quality gates' if future else 'Active model has only this stage remaining','kind':'model','state':state,'record_age_seconds':age,'local_owner_active':active,'root':str(root),'persistent_directory':session.get('persistent'),'persistent':sync,'error':session.get('error') or old.get('error'),'traceback':session.get('traceback') or old.get('traceback'),'error_report':str(error_report),'log_paths':[str(root/'stages'/str(stage)/'history-live.jsonl'),str(root/'stages'/str(stage)/'error.json')],'status_note':'Record age is not an exact stall detector; long preparation/sync can run between progress writes. PID/remote activity is not inferred from a stale file.'}


def stored_status(root):
    """Read committed metadata objects only; no restore or full object rehash loop."""
    from pdeflow.runtime import digest,file_digest
    pointer=read_optional(root/'latest.json');identifier=pointer.get('id','')
    if len(identifier)!=32 or not identifier.isalnum():return {'kind':'snapshot','state':'latest pointer unavailable; use read-only collector for a full committed-manifest scan','root':str(root)}
    path=root/'snapshots'/(identifier+'.json');manifest=read_optional(path)
    try:committed=manifest.get('id')==identifier and path.with_suffix('.commit').read_text()==digest(manifest)
    except OSError:committed=False
    if not committed:return {'kind':'snapshot','state':'latest manifest is not committed/valid; use read-only collector','root':str(root)}
    def object_record(name):
        e=manifest.get('files',{}).get(name,{})
        sha=e.get('sha256','')
        if e.get('bytes',2**30)>2*1024**2 or len(sha)!=64 or any(x not in '0123456789abcdef' for x in sha):return {}
        obj=root/'objects'/sha
        try:return read_json(obj) if file_digest(obj)==sha else {}
        except (OSError,ValueError):return {}
    suite=object_record('suite.json');prefix='runs/'+suite['active']+'/' if suite.get('active') else '';session=object_record(prefix+'session.json');r=object_record(prefix+'progress.json')
    if not r and session.get('stage'):
        old=object_record(prefix+'stages/'+session['stage']+'/run.json');cfg=old.get('config',{});settings=old.get('resolved_training_settings',{});history=old.get('history',[]);last=history[-1] if history else {};r={**old,'run_id':root.name,'stage':session['stage'],'phase':old.get('state',session.get('state')),'successful_updates':old.get('progress',session.get('progress')),'failed_attempts':old.get('failed_attempts'),'resolved_precision':old.get('precision'),'device':cfg.get('runtime',{}).get('device'),'batch_size':settings.get('effective_batch',cfg.get('batch_size')),'microbatch':settings.get('microbatch',cfg.get('microbatch')),'accumulation':settings.get('accumulation'),'loss':last.get('training_loss'),'learning_rate':last.get('learning_rate'),'elapsed_seconds':old.get('training_seconds'),'history':history}
    cfg=session.get('config',{})
    r.setdefault('validation_every',cfg.get('validation_every'));r.setdefault('training_smoothing_window',cfg.get('training_smoothing_window',100))
    vh=r.get('validation_history')
    if vh is None:vh=object_record(prefix+'stages/'+str(session.get('stage'))+'/validation-history.json') or recorded_validation(r.get('history',[]),r.get('stage'),object_record(prefix+'stages/'+str(session.get('stage'))+'/validation.json'))
    r=attach(r,vh)
    saved={'snapshot':identifier,'successful_update':manifest.get('successful_update'),'age_seconds':max(0,time.time()-manifest['created_unix']),'status':'committed metadata; full object integrity not rechecked by monitor'}
    result={**r,'kind':'snapshot','run_id':root.name,'root':str(root),'state':session.get('state','unknown')+' (last saved; remote activity unknown)','phase':r.get('phase','last snapshot'),'persistent':saved,'error':session.get('error'),'local_owner_active':None,'status_note':'Read-only persisted metadata, not live worker status. Latest pointer can be stale: collector scans and verifies all committed objects.'}
    if suite:return {'kind':'suite','run_id':root.name,'state':suite['state']+' (last saved; remote activity unknown)','candidates':{name:suite.get('candidates',{}).get(name,{}).get('state','pending') for name in suite['plan']['order']},'active_model':result,'persistent':saved,'eta_suite_seconds':None,'eta_suite_note':'Persisted snapshot only: no live ETA or remote-worker liveness claim'}
    return result


def text_summary(record):
    if record.get('kind')=='suite':return f"Suite {record['run_id']} · {record['state']} · {record['candidates']}\nSuite ETA {record.get('eta_suite_seconds')}s · {record['eta_suite_note']}\nDrive {record.get('persistent')}\n"+text_summary(record.get('active_model') or {})
    r=record;eta=r.get('eta_stage_seconds');fmt=lambda x:'—' if x is None else f'{x:.3g}' if isinstance(x,(int,float)) else str(x)
    text=f"{r.get('run_id',r.get('root','run'))} · {r.get('architecture','')} · {r.get('stage','')} · {r.get('phase',r.get('state','initialization'))}\nDevice {r.get('device','pending')} · precision {r.get('resolved_precision',r.get('precision','pending'))} · batch/micro/accumulation {r.get('batch_size')}/{r.get('microbatch')}/{r.get('accumulation')}\nSuccessful {r.get('successful_updates',r.get('progress',0))}/{r.get('budget','—')} · attempted/skipped/failed {r.get('attempted_updates',0)}/{r.get('skipped_updates',0)}/{r.get('failed_attempts',0)} · elapsed {fmt(r.get('elapsed_seconds'))}s\nLoss {fmt(r.get('loss'))} · moving mean {fmt(r.get('moving_average'))} · LR {fmt(r.get('learning_rate'))} · {fmt(r.get('examples_per_second'))} examples/s · {fmt(r.get('seconds_per_successful_update'))}s/update\nActive-model ETA {fmt(r.get('eta_model_seconds'))}s ({r.get('eta_model_note','pending')}) · stage ETA {fmt(eta)}s · {r.get('eta_note','Waiting for enough measured updates')}\nActive losses {r.get('losses',{})}\nComponents {r.get('components',{})}\nLocal {r.get('root','')} · persistent {r.get('persistent_directory','disabled')}\nDrive {r.get('persistent',{})}\n{r.get('error') or ''}"

    return text+'\n'+convergence_text(r)


def run(session,refresh_seconds=5,plots=True,selected=None,**kwargs):
    """Primary blocking training call with a rate-limited live display thread."""
    if refresh_seconds<.05:raise ValueError('Positive bounded refresh interval required')
    root=session.root;stop=threading.Event();output=None
    try:
        from IPython import get_ipython
        from IPython.display import display,HTML
        if get_ipython() is not None:output=display(HTML('<pre>Initializing mesh training…</pre>'),display_id=True)
    except ImportError:pass
    def render():
        r=status(root)
        # New sessions have no session.json yet. Do not initialize any model to display their plan.
        if r.get('kind')!='suite' and not r.get('architecture') and hasattr(session,'c'):
            from .settings import resolved
            c=session.c;settings=resolved(c,c.kind['stages'][0]);r.update(run_id=root.name,architecture=c.architecture,device=c.runtime['device'],phase='initialization',all_budgets={s:c.budgets[s] for s in c.kind['stages']},budget=c.budgets[c.kind['stages'][0]],stage=c.kind['stages'][0],batch_size=settings['effective_batch'],microbatch=settings['microbatch'],accumulation=settings['accumulation'],learning_rate=settings['learning_rate'],losses={'field':c.field_loss,'objective':c.objective},root=str(root),persistent_directory=str(session.persistent) if session.persistent else None)
        message=text_summary(r)
        if output:
            body=html_summary(r,plots)
            output.update(HTML(body))
        else:print(message,flush=True)
    render()
    def watch():
        while not stop.wait(refresh_seconds):
            try:render()
            except Exception as error:print('Monitor read/display error (training unaffected):',error,flush=True)
    worker=threading.Thread(target=watch,daemon=True);worker.start()
    try:return session.run(selected=selected,**kwargs) if hasattr(session,'inspect') and (root/'suite.json').exists() else session.run(**kwargs)
    finally:
        stop.set();worker.join(2)
        try:render()
        except Exception as error:print('Final monitor read failed; training outcome is retained:',error,flush=True)


def phase(root,name,**updates):
    root=Path(root);r=read_optional(root/'progress.json');r.update(phase=name,unix=time.time(),**updates);write_json(root/'progress.json',r)


def sync_store(root,persistent,keep=3,tag=None):
    from pdeflow.persistence import publish
    root=Path(root);previous=read_optional(root/'progress.json').get('phase','initialization');phase(root,'Drive syncing');old=read_optional(root/'sync-status.json');write_json(root/'sync-status.json',{**old,'status':'syncing','attempt_unix':time.time()})
    try:snapshot=publish(root,persistent,keep=keep,tag=tag)
    except BaseException as error:
        write_json(root/'sync-status.json',{**old,'status':'failed','error':str(error),'failure_unix':time.time(),'local_diagnostics_retained':True});phase(root,'persistent save failed');raise
    write_json(root/'sync-status.json',{'status':'verified','snapshot':snapshot['id'],'successful_update':snapshot.get('successful_update'),'stage':snapshot.get('stage'),'last_success_unix':time.time(),'persistent':str(persistent),'uploaded_bytes':snapshot.get('uploaded_bytes')});phase(root,previous);return snapshot


def convergence_text(record):
    current=record.get('active_model',record)
    current=attach(current);summary=current['training_summary'];fmt=lambda x:'not available' if x is None else f'{x:.6g}'
    lines=[f"{summary['label']}: {fmt(summary['mean'])} (n={summary['count']}/{summary['window']}, partial window allowed)"]
    for family,part in summary['family_objective'].items():
        lines.append(f"E[L | {family}]: {fmt(part['mean'])}; {part['examples']} examples, {part['missing_updates']} updates lacking recorded family contributions")
    latest=current['latest_validation'];best=current['best_validation']
    if not latest:
        lines.append(f"No validation has been recorded. Next scheduled validation update: {current.get('next_validation_update') if current.get('next_validation_update') is not None else 'not available / no remaining update'}")
    else:
        lines.append(f"Latest fixed validation update {latest.get('update','unknown')}: normalized domain MSE {fmt(latest.get('uniform_domain_normalized_mse'))}; physical aggregate MSE {fmt(latest.get('decoded_field_mse'))} (mixed squared quantity units); {latest.get('prediction_scope','prediction scope unavailable')}")
        for name in ('heat/u','wave/u','wave/v'):
            group=latest.get('groups',{}).get(name,{})
            lines.append(f"{name}: physical MSE {fmt(group.get('physical_mse'))}, RMSE {fmt(group.get('physical_rmse'))}, normalized MSE {fmt(group.get('normalized_mse'))}; cases {group.get('case_count','not available')}, points {group.get('evaluated_points','not available')}")
        if best:lines.append(f"Best recorded normalized domain MSE {fmt(best['uniform_domain_normalized_mse'])} at update {best.get('update','unknown')}; diagnostic only, no new checkpoint-selection rule")
    lines.append(summary['quantity_objective'])
    return '\n'.join(lines)


def _metric_plot(title,series,ylabel):
    """SVG only; consumes actual recorded coordinates, never makes validation samples."""
    series=[(name,color,[(x,y) for x,y in points if isinstance(y,(float,int)) and math.isfinite(y)],opacity,markers)
            for name,color,points,opacity,markers in series]
    series=[s for s in series if s[2]]
    if not series:return ''
    all_points=[p for _,_,points,_,_ in series for p in points];xs=[x for x,_ in all_points];ys=[y for _,y in all_points]
    xmin,xmax=min(xs),max(xs);low,high=min(ys),max(ys);pad=max((high-low)*.08,abs(high)*.01,1e-8);low-=pad;high+=pad
    px=lambda x:65+500*(x-xmin)/max(xmax-xmin,1)
    py=lambda y:190-160*(y-low)/(high-low)
    body=f'<figure><h4>{html.escape(title)}</h4><svg role="img" aria-label="{html.escape(title)}" viewBox="0 0 600 245" style="max-width:900px;width:100%;background:#fff;color:#111"><path d="M65 25V190H575" fill="none" stroke="#777"/>'
    for value in (low,(high+low)/2,high):
        body+=f'<text x="59" y="{py(value)+4:.2f}" font-size="11" text-anchor="end">{value:.3g}</text>'
    body+=f'<text x="65" y="210" font-size="11">{xmin}</text><text x="565" y="210" text-anchor="end" font-size="11">{xmax}</text><text x="220" y="235" font-size="12">Successful optimizer update</text>'
    for name,color,points,opacity,markers in series:
        coordinates=' '.join(f'{px(x):.2f},{py(y):.2f}' for x,y in points)
        body+=f'<polyline fill="none" stroke="{color}" stroke-opacity="{opacity}" stroke-width="2" points="{coordinates}"/>'
        if markers:
            for x,y in points:body+=f'<circle class="validation-marker" cx="{px(x):.2f}" cy="{py(y):.2f}" r="4" fill="{color}" stroke="white"><title>{html.escape(name)}: update {x}, value {y:.7g}</title></circle>'
    body+='</svg><figcaption>'+html.escape(ylabel)+' · '+'; '.join(f'<span style="color:{color}">{html.escape(name)}</span>' for name,color,_,_,_ in series)+'</figcaption></figure>'
    return body


def html_summary(record,plots=True):
    """Existing live/reconnect monitor's HTML representation; no file writes or model calls."""
    from .convergence import success_rows,rolling_series
    current=attach(record.get('active_model',record));body='<pre style="white-space:pre-wrap">'+html.escape(text_summary(record))+'</pre>'
    if not plots:return body
    title=f"{current.get('architecture','recorded model')} · {current.get('stage','recorded stage')}"
    history=current.get('smoothing_history',current.get('history',[]));window=current.get('training_smoothing_window',100)
    raw=success_rows(current.get('history',[]))[-200:];smooth=rolling_series(history,window)[-200:]
    body+=_metric_plot(title+' · Training objective',[
        ('Raw optimized objective (successful updates)','#247b75',[(r['step'],r['training_loss']) for r in raw],.25,False),
        (f'Rolling mean, last {window} successful updates','#155a56',[(r['step'],r['mean']) for r in smooth],1.,False)],'Optimized objective; partial-window means before a full window')
    validations=current['validation_history'];colors={'heat/u':'#227b55','wave/u':'#456cc2','wave/v':'#a146a1'}
    aggregate=[(r['update'],r['uniform_domain_normalized_mse']) for r in validations if r.get('update') is not None and r.get('uniform_domain_normalized_mse') is not None]
    if aggregate:
        series=[('Aggregate normalized decoded-field MSE','#dd7d21',aggregate,1.,True)]
        for name,color in colors.items():series.append((name,color,[(r['update'],r['groups'][name]['normalized_mse']) for r in validations if name in r.get('groups',{}) and r.get('update') is not None],1.,True))
        body+=_metric_plot(title+' · Fixed-validation normalized decoded-field MSE',series,'Full recorded stage; fixed TRAIN quantity scale squared. Markers are actual evaluations; lines are reading aids.')
        physical=[(name,color,[(r['update'],r['groups'][name]['physical_mse']) for r in validations if name in r.get('groups',{}) and r.get('update') is not None],1.,True) for name,color in colors.items()]
        if any(s[2] for s in physical):body+='<details><summary>Physical-scale validation MSE (quantity units differ)</summary>'+_metric_plot(title+' · Physical-scale per-quantity validation',physical,'Separate quantities; raw squared displacement/velocity units are not directly comparable.')+'</details>'
    elif validations:body+='<p>A validation aggregate is recorded but its evaluation update is unavailable. No marker is inferred.</p>'
    else:body+='<p>No validation has been recorded. Next scheduled validation update: '+html.escape(str(current.get('next_validation_update') if current.get('next_validation_update') is not None else 'not available / no remaining update'))+'</p>'
    return body


def display_status(root,plots=True):
    """Read-only notebook/reconnect view using the same renderer as live training."""
    record=status(root)
    try:
        from IPython import get_ipython
        from IPython.display import display,HTML
        if get_ipython() is not None:display(HTML(html_summary(record,plots)))
        else:print(text_summary(record),flush=True)
    except ImportError:print(text_summary(record),flush=True)
    return record
