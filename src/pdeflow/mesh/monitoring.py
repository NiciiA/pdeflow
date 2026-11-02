"""Bounded live records and read-only mesh/suite monitoring; no model imports."""
from pathlib import Path
import json,time,os,threading,socket,statistics,html
from collections import deque
from pdeflow.runtime import read_json,write_json


class Reporter:
    def __init__(self,root,c,stage=None,callback=None):
        self.root=Path(root);self.started=time.monotonic();self.last=0.;self.callback=callback;self.rows=deque(maxlen=200);self.seen=0
        self.record={'schema':'mesh_progress_v1','run_id':self.root.name,'architecture':c.architecture,'stage':stage,'device':c.runtime['device'],'resolved_precision':'pending device resolution','autocast_dtype':None,'budget':c.budgets.get(stage),'all_budgets':{s:c.budgets[s] for s in c.kind['stages']},'batch_size':c.batch_size,'microbatch':c.microbatch,'accumulation':__import__('math').ceil(c.batch_size/c.microbatch),'losses':{'field':c.field_loss,'objective':c.objective,'stages':list(c.kind['stages'])},'successful_updates':0,'attempted_updates':0,'skipped_updates':0,'failed_attempts':0,'pid':os.getpid(),'host':socket.gethostname()}
    def update(self,phase=None,force=False,**values):
        changed=phase is not None and phase!=self.record.get('phase');self.record.update(values)
        if phase is not None:self.record['phase']=phase
        now=time.monotonic()
        if not(force or changed or now-self.last>=5):return
        self.record.update(unix=time.time(),elapsed_seconds=now-self.started,history=list(self.rows));write_json(self.root/'progress.json',self.record);self.last=now
        if self.callback:self.callback(dict(self.record))
    def row(self,row):
        row=dict(row);row['monitor_warmup']=self.seen<2;self.seen+=1
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
        old=object_record(prefix+'stages/'+session['stage']+'/run.json');cfg=old.get('config',{});history=old.get('history',[]);last=history[-1] if history else {};r={**old,'run_id':root.name,'stage':session['stage'],'phase':old.get('state',session.get('state')),'successful_updates':old.get('progress',session.get('progress')),'failed_attempts':old.get('failed_attempts'),'resolved_precision':old.get('precision'),'device':cfg.get('runtime',{}).get('device'),'batch_size':cfg.get('batch_size'),'microbatch':cfg.get('microbatch'),'loss':last.get('training_loss'),'learning_rate':last.get('learning_rate'),'elapsed_seconds':old.get('training_seconds'),'history':history}
    saved={'snapshot':identifier,'successful_update':manifest.get('successful_update'),'age_seconds':max(0,time.time()-manifest['created_unix']),'status':'committed metadata; full object integrity not rechecked by monitor'}
    result={**r,'kind':'snapshot','run_id':root.name,'root':str(root),'state':session.get('state','unknown')+' (last saved; remote activity unknown)','phase':r.get('phase','last snapshot'),'persistent':saved,'error':session.get('error'),'local_owner_active':None,'status_note':'Read-only persisted metadata, not live worker status. Latest pointer can be stale: collector scans and verifies all committed objects.'}
    if suite:return {'kind':'suite','run_id':root.name,'state':suite['state']+' (last saved; remote activity unknown)','candidates':{name:suite.get('candidates',{}).get(name,{}).get('state','pending') for name in suite['plan']['order']},'active_model':result,'persistent':saved,'eta_suite_seconds':None,'eta_suite_note':'Persisted snapshot only: no live ETA or remote-worker liveness claim'}
    return result


def text_summary(record):
    if record.get('kind')=='suite':return f"Suite {record['run_id']} · {record['state']} · {record['candidates']}\nSuite ETA {record.get('eta_suite_seconds')}s · {record['eta_suite_note']}\nDrive {record.get('persistent')}\n"+text_summary(record.get('active_model') or {})
    r=record;eta=r.get('eta_stage_seconds');fmt=lambda x:'—' if x is None else f'{x:.3g}' if isinstance(x,(int,float)) else str(x)
    return f"{r.get('run_id',r.get('root','run'))} · {r.get('architecture','')} · {r.get('stage','')} · {r.get('phase',r.get('state','initialization'))}\nDevice {r.get('device','pending')} · precision {r.get('resolved_precision',r.get('precision','pending'))} · batch/micro/accumulation {r.get('batch_size')}/{r.get('microbatch')}/{r.get('accumulation')}\nSuccessful {r.get('successful_updates',r.get('progress',0))}/{r.get('budget','—')} · attempted/skipped/failed {r.get('attempted_updates',0)}/{r.get('skipped_updates',0)}/{r.get('failed_attempts',0)} · elapsed {fmt(r.get('elapsed_seconds'))}s\nLoss {fmt(r.get('loss'))} · moving mean {fmt(r.get('moving_average'))} · LR {fmt(r.get('learning_rate'))} · {fmt(r.get('examples_per_second'))} examples/s · {fmt(r.get('seconds_per_successful_update'))}s/update\nActive-model ETA {fmt(r.get('eta_model_seconds'))}s ({r.get('eta_model_note','pending')}) · stage ETA {fmt(eta)}s · {r.get('eta_note','Waiting for enough measured updates')}\nActive losses {r.get('losses',{})}\nComponents {r.get('components',{})}\nLocal {r.get('root','')} · persistent {r.get('persistent_directory','disabled')}\nDrive {r.get('persistent',{})}\n{r.get('error') or ''}"


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
            c=session.c;r.update(run_id=root.name,architecture=c.architecture,device=c.runtime['device'],phase='initialization',all_budgets={s:c.budgets[s] for s in c.kind['stages']},budget=c.budgets[c.kind['stages'][0]],stage=c.kind['stages'][0],batch_size=c.batch_size,microbatch=c.microbatch,accumulation=__import__('math').ceil(c.batch_size/c.microbatch),losses={'field':c.field_loss,'objective':c.objective},root=str(root),persistent_directory=str(session.persistent) if session.persistent else None)
        message=text_summary(r)
        if output:
            body='<pre>'+html.escape(message)+'</pre>';current=r.get('active_model',r);rows=current.get('history',[])
            if plots and rows:
                finite=[(i,v['training_loss']) for i,v in enumerate(rows) if v.get('training_loss') is not None];vals=[v for _,v in finite]
                if vals:
                    extent=vals+[v['validation_loss'] for v in rows if v.get('validation_loss') is not None];low,high=min(extent),max(extent);scale=max(high-low,1e-12);points=' '.join(f'{10+580*i/max(1,len(rows)-1):.2f},{150-130*(v-low)/scale:.2f}' for i,v in finite);markers=''.join(f'<circle cx="{10+580*i/max(1,len(rows)-1):.2f}" cy="{150-130*(v["validation_loss"]-low)/scale:.2f}" r="4" fill="orange"/>' for i,v in enumerate(rows) if v.get('validation_loss') is not None)
                    body+='<svg viewBox="0 0 600 180"><polyline fill="none" stroke="teal" points="'+points+'"/>'+markers+'</svg><p>Recent raw objective; orange markers: fixed validation normalized MSE (distinct objective).</p>'
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
