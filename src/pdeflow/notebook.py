"""Rate-limited notebook progress; no model/data/optimizer implementation."""
from pathlib import Path
import threading,time
import numpy as np
from pdeflow.runtime import read_json


def full_history(root,active):
    """Merge committed history, append-only local journal and live tail by attempt."""
    root=Path(root);rows=[]
    if (root/'history.json').exists():rows.extend(read_json(root/'history.json'))
    if (root/'history-live.jsonl').exists():
        import json
        for line in (root/'history-live.jsonl').read_text().splitlines():
            try:rows.append(json.loads(line))
            except ValueError:continue  # interrupted trailing line is not a measurement
    rows.extend(active.get('history',[]));latest={}
    for row in rows:
        index=row.get('attempted_update',row['step'])
        if index<=active.get('attempted_updates',active.get('progress',index)):latest[index]=row
    return [latest[k] for k in sorted(latest)]


def history_figure(history):
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,3));x=[r['step'] for r in history];y=[r['training_loss'] for r in history]
    ax.plot(x,y,alpha=.35,label='training objective (full history)')
    window=min(25,len(y))
    if window:ax.plot(x[window-1:],np.convolve(y,np.ones(window)/window,mode='valid'),label=f'training moving average ({window} attempts)')
    val=[r for r in history if 'validation_loss' in r]
    for fixed,label in ((True,'fixed validation objective'),(False,'legacy resampled validation')):
        selected=[r for r in val if ('validation_policy' in r)==fixed]
        if selected:ax.plot([r['step'] for r in selected],[r['validation_loss'] for r in selected],marker='o',markersize=5,label=label)
    ax.legend();ax.set_xlabel('Successful optimizer updates');return fig


def status(session):
    root=session.root;record=read_json(root/'session.json') if (root/'session.json').exists() else {'state':'not started'}
    stages=[]
    for name in ('autoencoder','evolution','flow'):
        p=root/name/'run.json'
        if p.exists():stages.append(read_json(p))
    active=next((s for s in stages if s['state'] in ('preparing','running')),stages[-1] if stages else None)
    if active:
        rows=full_history(root/active['stage'],active);timed=[r['step_seconds'] for r in rows[-20:] if 'step_seconds' in r]
        mean=float(np.median(timed)) if len(timed)>=5 else None
        recent=rows[-20:];wall=sum(v['step_seconds'] for v in recent if 'step_seconds' in v);successful=sum(not v.get('optimizer_update_skipped',False) for v in recent);rate=successful/wall if len(timed)>=5 and wall>0 else None
        record={**record,'stage':active['stage'],'stage_state':active['state'],'attempted_updates':active.get('attempted_updates',active['progress']),'successful_updates':active['progress'],'skipped_updates':active.get('skipped_updates',0),'batch_settings':active.get('batch_settings'),'update':active['progress'],'planned':active['budget'],'elapsed_seconds':active.get('elapsed_seconds',active.get('training_seconds')),
          'examples_per_second':active['config']['batch_size']/mean if mean else None,'estimated_stage_seconds_remaining':(active['budget']-active['progress'])/rate if rate else None,
          'memory':active.get('memory'),'history':rows,'actual_precision':active.get('precision')}
    sync=root/'sync-status.json'
    if sync.exists():
        saved=read_json(sync)
        if saved.get('last_success_unix'):saved['persistent_age_seconds']=max(0,time.time()-saved['last_success_unix'])
        record['persistent_checkpoint']=saved
    if (root/'representation-quality.json').exists():record['representation_quality']=read_json(root/'representation-quality.json')
    record['local_bytes']=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    record['persistent_bytes']=sum(p.stat().st_size for p in session.persistent.rglob('*') if p.is_file()) if session.persistent.exists() else 0
    return record

def display_status(session,plots=True):
    from IPython.display import display,clear_output
    import matplotlib.pyplot as plt
    r=status(session);clear_output(wait=True)
    print(f"{session.name} · {r['state']} · {r.get('stage','')} · {r.get('update',0)}/{r.get('planned','—')} optimizer updates")
    print('ETA:',round(r['estimated_stage_seconds_remaining']), 's' if r.get('estimated_stage_seconds_remaining') is not None else '') if r.get('estimated_stage_seconds_remaining') is not None else print('ETA: waiting for at least five measured updates')
    print('Attempted / successful / skipped:',r.get('attempted_updates'),r.get('successful_updates'),r.get('skipped_updates'),'· batch settings:',r.get('batch_settings'))
    print('Throughput:',r.get('examples_per_second'),'examples/s · precision:',r.get('actual_precision'))
    print('Elapsed stage wall time:',r.get('elapsed_seconds'),'s')
    if r.get('representation_quality'):print('Representation gate:',r['representation_quality'],'· explicit override:',r.get('gate_override',False))
    print('GPU memory:',r.get('memory'),'· last persistent checkpoint:',r.get('persistent_checkpoint'))
    print('Local / persistent bytes:',r['local_bytes'],r['persistent_bytes'])
    h=r.get('history',[])
    if h and plots:
        fig=history_figure(h);display(fig);plt.close(fig)
    if r.get('stage')=='evolution':
        from pdeflow.coordinate.evolution_monitor import read_preview
        preview=read_preview(session.root/'evolution') if plots else None
        if preview:
            meta,frames=preview
            fig,axes=plt.subplots(len(frames),3,figsize=(10,3*len(frames)),squeeze=False)
            for row,(caption,values) in enumerate(zip(meta['rows'],frames)):
                lo,hi=float(values[:2].min()),float(values[:2].max())
                title=(f"A2 {caption['family']} {caption['quantity']} · physical t={caption['physical_time']:.4g} · {caption['id']}\n"
                       f"weights/checkpoint update {meta['checkpoint_update']}; preview generation update {meta['preview_generation_update']} · RMSE={caption['rmse']:.5g}")
                axes[row,0].text(0,1.25,title,transform=axes[row,0].transAxes,fontsize=8)
                for col,name in enumerate(('Reference','Current evolved prediction','Absolute error')):
                    axes[row,col].imshow(values[col].T,origin='lower',vmin=0 if col==2 else lo,vmax=max(float(values[2].max()),1e-12) if col==2 else hi)
                    axes[row,col].set_title(name);axes[row,col].axis('off')
            fig.tight_layout();display(fig);plt.close(fig)
        elif plots:print('No verified live A2 preview yet. A1/t=0 images are deliberately not used as A2 evidence.')
        return {k:v for k,v in r.items() if k not in ('history','config','validation')}
    candidates=[session.root/'flow-validation.npz',session.root/'flow/validation-preview.npz',session.root/'representation-validation.npz',session.root/'autoencoder/validation-preview.npz']
    p=next((p for p in candidates if p.exists()),None)
    if p and plots:
        with np.load(p,allow_pickle=False) as a:preview=a['preview']
        if preview.ndim==5:preview=preview[0]
        if preview.ndim==4:
            fig,axes=plt.subplots(1,3,figsize=(9,3));lo,hi=float(preview[:2].min()),float(preview[:2].max())
            for i,title in enumerate(('Reference','Reconstruction','Absolute error')):axes[i].imshow(preview[i,0].T,origin='lower',vmin=0 if i==2 else lo,vmax=float(preview[2].max()) if i==2 else hi);axes[i].set_title(title);axes[i].axis('off')
            display(fig);plt.close(fig)
    return {k:v for k,v in r.items() if k not in ('history','config','validation')}

def run(session,refresh_seconds=10,plot_seconds=120,**kwargs):
    if not 1<=refresh_seconds<=3600 or plot_seconds<refresh_seconds:raise ValueError("Notebook refresh must be1..3600 seconds and plot interval at least refresh interval")
    stop=threading.Event();last_plot=[0.];output_seconds=[0.]
    def watch():
        while not stop.wait(refresh_seconds):
            try:
                tick=time.perf_counter();plots=tick-last_plot[0]>=plot_seconds
                display_status(session,plots)
                if plots:last_plot[0]=tick
                output_seconds[0]+=time.perf_counter()-tick
                from pdeflow.runtime import write_json
                write_json(session.root/'notebook-output-timing.json',{'seconds':output_seconds[0],'refresh_seconds':refresh_seconds,'plot_seconds':plot_seconds,'meaning':'Host display overhead, separate from training step time'})
            except (OSError,ValueError):pass # transient display reads never stop science
    worker=threading.Thread(target=watch,daemon=True);worker.start()
    try:return session.run(**kwargs)
    finally:stop.set();worker.join(2);display_status(session)
