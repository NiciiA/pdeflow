"""Rate-limited notebook progress; no model/data/optimizer implementation."""
from pathlib import Path
import threading,time
import numpy as np
from pdeflow.runtime import read_json

def status(session):
    root=session.root;record=read_json(root/'session.json') if (root/'session.json').exists() else {'state':'not started'}
    stages=[]
    for name in ('autoencoder','flow'):
        p=root/name/'run.json'
        if p.exists():stages.append(read_json(p))
    active=next((s for s in stages if s['state'] in ('preparing','running')),stages[-1] if stages else None)
    if active:
        rows=active.get('history',[]);timed=[r['step_seconds'] for r in rows[-20:] if 'step_seconds' in r]
        mean=float(np.median(timed)) if len(timed)>=5 else None
        record={**record,'stage':active['stage'],'stage_state':active['state'],'update':active['progress'],'planned':active['budget'],'elapsed_seconds':active.get('elapsed_seconds',active.get('training_seconds')),
          'examples_per_second':active['config']['batch_size']/mean if mean else None,'estimated_stage_seconds_remaining':(active['budget']-active['progress'])*mean if mean else None,
          'memory':active.get('memory'),'history':rows,'actual_precision':active.get('precision')}
    sync=root/'sync-status.json'
    if sync.exists():record['persistent_checkpoint']=read_json(sync)
    if (root/'representation-quality.json').exists():record['representation_quality']=read_json(root/'representation-quality.json')
    record['local_bytes']=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    record['persistent_bytes']=sum(p.stat().st_size for p in session.persistent.rglob('*') if p.is_file()) if session.persistent.exists() else 0
    return record

def display_status(session):
    from IPython.display import display,clear_output
    import matplotlib.pyplot as plt
    r=status(session);clear_output(wait=True)
    print(f"{session.name} · {r['state']} · {r.get('stage','')} · {r.get('update',0)}/{r.get('planned','—')} optimizer updates")
    print('ETA:',round(r['estimated_stage_seconds_remaining']), 's' if r.get('estimated_stage_seconds_remaining') is not None else '') if r.get('estimated_stage_seconds_remaining') is not None else print('ETA: waiting for at least five measured updates')
    print('Throughput:',r.get('examples_per_second'),'examples/s · precision:',r.get('actual_precision'))
    print('Elapsed stage wall time:',r.get('elapsed_seconds'),'s')
    if r.get('representation_quality'):print('Representation gate:',r['representation_quality'],'· explicit override:',r.get('gate_override',False))
    print('GPU memory:',r.get('memory'),'· last persistent checkpoint:',r.get('persistent_checkpoint'))
    print('Local / persistent bytes:',r['local_bytes'],r['persistent_bytes'])
    h=r.get('history',[])
    if h:
        fig,ax=plt.subplots(figsize=(7,2));ax.plot([v['step'] for v in h],[v['training_loss'] for v in h],label='training objective')
        val=[v for v in h if 'validation_loss' in v]
        if val:ax.plot([v['step'] for v in val],[v['validation_loss'] for v in val],label='fixed validation objective')
        ax.legend();ax.set_xlabel('Optimizer updates');display(fig);plt.close(fig)
    candidates=[session.root/'flow-validation.npz',session.root/'flow/validation-preview.npz',session.root/'representation-validation.npz',session.root/'autoencoder/validation-preview.npz']
    p=next((p for p in candidates if p.exists()),None)
    if p:
        with np.load(p,allow_pickle=False) as a:preview=a['preview']
        if preview.ndim==5:preview=preview[0]
        if preview.ndim==4:
            fig,axes=plt.subplots(1,3,figsize=(9,3));lo,hi=float(preview[:2].min()),float(preview[:2].max())
            for i,title in enumerate(('Reference','Reconstruction','Absolute error')):axes[i].imshow(preview[i,0].T,origin='lower',vmin=0 if i==2 else lo,vmax=float(preview[2].max()) if i==2 else hi);axes[i].set_title(title);axes[i].axis('off')
            display(fig);plt.close(fig)
    return {k:v for k,v in r.items() if k not in ('history','config','validation')}

def run(session,**kwargs):
    stop=threading.Event()
    def watch():
        while not stop.wait(10):
            try:display_status(session)
            except (OSError,ValueError):pass # transient display reads never stop science
    worker=threading.Thread(target=watch,daemon=True);worker.start()
    try:return session.run(**kwargs)
    finally:stop.set();worker.join(2);display_status(session)
