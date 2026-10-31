"""Notebook-facing orchestration. Learning stays in coordinate.training.train."""
from pathlib import Path
from dataclasses import replace
import gc,os,time,tempfile,json,shutil,uuid,hashlib
import numpy as np
import torch
from pdeflow.runtime import read_json,write_json,atomic_write,file_digest,digest,environment,RuntimeSpec
from pdeflow.coordinate.contracts import CoordinateConfig
from pdeflow.coordinate.optimization import options,precision,scientific_source,model_contract,cuda_precision_support


def preset(name='large',device='cuda',precision_policy='auto',overrides=None):
    c=CoordinateConfig(dimension=2,reference_shape=(64,64),times=(0.,.05,.1,.15),width=512,tokens=128,layers=8,heads=8,
       train_count=65536,validation_count=2048,test_count=2048,ae_steps=100000,evolution_steps=100000,flow_steps=100000,encoder_points=1024,query_points=512,
       batch_size=8,learning_rate=.0003,checkpoint_every=500,observation_counts=(1,128),unconditional_probability=0.,data_mode='procedural',temporal_mode='continuous',time_interval=(0.,1.),
       distribution={'max_wave':4,'amplitude':[.4,1.2],'background':[-.5,.5],'width':[.08,.18]},
       runtime={'device':device,'dtype':'float32','threads':2},training_options={'precision':precision_policy,'calibration_count':512,
           'validation_every':500,'validation_subset':12,'broad_every':5000,'broad_subset':48,'observation_locations':'scattered',
           'autoencoder':{'warmup':1000,'schedule':'cosine'},'flow':{'warmup':1000,'schedule':'cosine'}})
    if name in ('pipeline-check','smoke'):
        c=replace(c,reference_shape=(8,8),train_count=12,validation_count=12,test_count=12,
          ae_steps=2,evolution_steps=2,flow_steps=2,encoder_points=32,query_points=32,batch_size=2,checkpoint_every=1,observation_counts=(1,8),
          distribution={'max_wave':1,'width':[.4,.5]},training_options={'precision':precision_policy,'calibration_count':6,'validation_every':1,'validation_subset':2,'broad_every':20,'broad_subset':6,'observation_locations':'scattered'})
    elif name in ('xsmall','small','medium'):
        c=replace(c,reference_shape=(32,32),train_count=8192,validation_count=256,test_count=256,ae_steps=10000,evolution_steps=10000,flow_steps=10000,encoder_points=512,query_points=256,observation_counts=(1,64),distribution={**c.distribution,'width':[.1,.2]})
    elif name in ('extra-large','xlarge'):
        c=replace(c,train_count=131072,validation_count=4096,test_count=4096,ae_steps=200000,evolution_steps=200000,flow_steps=200000,encoder_points=2048,query_points=1024,training_options={**c.training_options,'microbatch':4,'accumulation':2,'activation_checkpointing':True})
    from pdeflow.coordinate.presets import capacity
    c=capacity(c,name)
    raw=c.to_dict()
    for k,v in (overrides or {}).items():
        if k not in raw:raise ValueError('Unknown configuration override: '+k)
        raw[k]={**raw[k],**v} if isinstance(v,dict) and isinstance(raw[k],dict) else v
    return CoordinateConfig(**raw).validate()


def training_preset(name='large',device='cuda',precision_policy='auto',overrides=None):
    """Updated NEW-run presets. preset() retains historical ordinary MSE semantics."""
    from pdeflow.field_losses import recommended
    overrides=overrides or {}
    return preset(name,device,precision_policy,{**overrides,'field_loss':{**recommended(),**overrides.get('field_loss',{})}})


def hardware(device='cuda',disk_root=None):
    import platform,sys
    ram=None
    try:ram=os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES')
    except (ValueError,OSError):pass
    if Path('/proc/meminfo').exists():
        entries=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines());available=int(entries['MemAvailable'].split()[0])*1024
    else:available=None
    gpu=None
    if torch.cuda.is_available():
        p=torch.cuda.get_device_properties(0);free,total=torch.cuda.mem_get_info();gpu={'name':p.name,'total_bytes':total,'free_bytes':free,'bf16':cuda_precision_support()['bf16_native'],**cuda_precision_support(),'capability':list(torch.cuda.get_device_capability())}
    return {'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'numpy':np.__version__,
            'gpu':gpu,'host_ram_bytes':ram,'available_host_ram_bytes':available,'local_disk_free_bytes':shutil.disk_usage(disk_root or Path.cwd()).free,'selected_device':device}


def guard(c,allow_cpu_check=False):
    report=hardware(c.runtime['device'])
    if c.runtime['device']=='cuda' and not torch.cuda.is_available():raise ValueError('No CUDA GPU. Select Runtime → Change runtime type → GPU, or explicitly select CPU with the pipeline-check preset.')
    device,_=RuntimeSpec(**c.runtime).resolve()
    small=c.width<=32 and c.train_count<=64 and max(c.ae_steps,c.evolution_steps,c.flow_steps)<=20
    if c.data_mode=='materialized' and np.prod(c.reference_shape)*len(c.times)*4*(c.train_count+c.validation_count+c.test_count)>512*1024**2:
        raise ValueError('This substantial Colab run must use procedural data; materialized legacy data would exceed the bounded host-memory policy')
    if device.type!='cuda' and not (allow_cpu_check and small):raise ValueError('Substantial Colab runs require CUDA. Choose Runtime → Change runtime type → GPU. CPU is only allowed explicitly for a small pipeline check.')
    precision(options(c)['precision'],device)
    return report


def summary(c):
    from pdeflow.coordinate.presets import counts as parameter_counts
    counts=parameter_counts(c)
    o=options(c);nodes=int(np.prod(c.reference_shape));bytes_example=nodes*(c.animation_frames if c.temporal_mode=='continuous' else len(c.times))*4
    return {'batch_settings':__import__('pdeflow.training_settings',fromlist=['resolve_batches']).resolve_batches(c),'architecture':'Operator-Aware Physical Latent v2','parameters':counts,
      'field_reconstruction_loss':__import__('dataclasses').asdict(__import__('pdeflow.field_losses',fromlist=['settings']).settings(c.field_loss)),
      'supervision_query_cost':{'domain_queries':c.query_points,'additional_region_queries':__import__('pdeflow.field_losses',fromlist=['settings']).settings(c.field_loss).region_queries if __import__('pdeflow.field_losses',fromlist=['settings']).settings(c.field_loss).type=='region_balanced_mse' else 0},
      'config':c.to_dict(),'training_examples':c.train_count,'validation_examples':c.validation_count,'untouched_test_examples':c.test_count,
      'reference_shape':list(c.reference_shape),'temporal_mode':c.temporal_mode,'time_interval':list(c.interval),'animation_frames':c.animation_frames,'legacy_snapshot_times':list(c.times) if c.temporal_mode!='continuous' else None,'effective_batch':c.batch_size,'microbatch':o['microbatch'],'accumulation':o['accumulation'],
      'updates':{'representation':c.ae_steps,'evolution':c.evolution_steps,'flow':c.flow_steps},'hypothetical_materialized_float32_bytes':bytes_example*(c.train_count+c.validation_count+c.test_count),
      'bounded_reference_cache_bytes':o['reference_cache_items']*(nodes*52+c.encoder_points*16) if c.temporal_mode=='continuous' else bytes_example*8,'cache_estimate_definition':'Bounded reference_cache_items initial grids/spectral state/context sets; coefficient/frame LRUs and current batch/temporary chunks additional' if c.temporal_mode=='continuous' else 'Eight full legacy examples','maximum_latent_cache_bytes':4*o['frozen_cache_items']*c.tokens*c.width,'latent_cache_scope':'CPU frozen encoder LRU plus separately bounded local disk target cache; two such bounds, metadata overhead additional. Recomputable caches excluded from persistent snapshots.','estimated_weight_bytes':4*counts['total_training'],'rough_adam_weights_moments_bytes':12*counts['total_training'],
      'estimates':'Parameter/storage arithmetic only; activations, attention, AMP workspaces and allocator overhead need actual profiling. No fit or speed guarantee.'}


def gallery(c,output):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pdeflow.coordinate.dataset import example,recipe
    n=min(12,c.train_count);fig,axes=plt.subplots(2,n,figsize=(2.5*n,5),squeeze=False)
    records=[]
    for i in range(n):
        a=example(c,'train',i,dtype=np.float64);u=(np.stack([a['solver'].frame(t,a['D'],c.interval) for t in c.interval]) if c.temporal_mode=='continuous' else a['values'].reshape(len(c.times),*c.reference_shape));fine=np.stack([a['solver'].frame(t,a['parameter'],c.interval,tuple(2*x for x in c.reference_shape)) for t in c.interval])
        error=float(np.sqrt(np.mean((u-fine[:,::2,::2])**2)));records.append({'family':a['family'],'id':a['id'],'parameter':a['parameter'],'case':a['case'],'native_vs_2x_reference_rmse':error})
        lo,hi=float(u.min()),float(u.max())
        for j,t in enumerate((0,-1)):axes[j,i].imshow(u[t].T,origin='lower',vmin=lo,vmax=hi,cmap='viridis');axes[j,i].set_title(f'{a["family"]}\nt={(c.interval if c.temporal_mode=='continuous' else c.times)[t]:g}');axes[j,i].axis('off')
    fig.tight_layout();Path(output).parent.mkdir(parents=True,exist_ok=True);fig.savefig(output);plt.close(fig)
    write_json(Path(output).with_suffix('.json'),{'records':records,'meaning':'Same deterministic underlying field at native / twice-finer FFT reference. Discretization discrepancy, not learned error; inspect before substantial training.'});return records


def profile(c,local_root,allow_cpu_check=False,updates=10,warmup=2):
    if updates<warmup+5 or warmup<1:raise ValueError('Profile requires >=1 warmup and >=5 measured updates')
    from pdeflow.coordinate.profiling import distribution
    guard(c,allow_cpu_check);requested=digest(c.to_dict());o=options(c);report={'scientific_source':scientific_source()['sha256'],'requested_config':requested,'hardware':hardware(c.runtime['device']),'success':False}
    from pdeflow.coordinate.training import train
    from pdeflow.coordinate.validation import representation,flow as validate_flow
    dev,_=RuntimeSpec(**c.runtime).resolve();devices=list(range(torch.cuda.device_count())) if dev.type=='cuda' else []
    try:
        with torch.random.fork_rng(devices=devices),tempfile.TemporaryDirectory(prefix='pdeflow-profile-',dir=local_root) as temporary:
            p=Path(temporary);pc=replace(c,ae_steps=updates,evolution_steps=updates,flow_steps=max(2,updates),checkpoint_every=updates,
              training_options={**c.training_options,'validation_every':updates,'profile_components':True})
            records={}
            for stage in ('autoencoder','evolution','flow'):
                stage_root=p/stage
                r=train(pc.to_dict(),stage_root,stage,p/'evolution/evolution.pt' if stage=='flow' else p/'autoencoder/autoencoder.pt' if stage=='evolution' else None)
                rows=read_json(stage_root/'history.json');measured=rows[warmup:];seconds=float(np.median([v['step_seconds'] for v in measured]));records[stage]={'step_seconds':seconds,
                    'update_wall_seconds':distribution([v['step_seconds'] for v in measured]),'end_to_end_update_seconds':distribution([v['end_to_end_seconds'] for v in measured]),'examples_per_second':pc.batch_size/seconds,'warmup_rows':rows[:warmup],
                    'setup_seconds':r['setup_seconds'],'final_persistence_and_calibration_seconds':r['final_persistence_and_calibration_seconds'],'latent_calibration_seconds':r.get('latent_calibration_seconds',0.),'cache_bytes':sum(f.stat().st_size for f in stage_root.rglob('*') if f.is_file() and 'latent-targets' in f.parts),'frozen_cache':rows[-1]['frozen_cache'],'cache_population_and_io_seconds':sum(sum((v.get('component_timings') or {}).get('host_exclusive_seconds',{}).get(k,0.) for k in ('frozen_cache_store','target_cache_io')) for v in rows),'skipped_updates':sum(v['optimizer_update_skipped'] for v in rows),
                    'data_seconds':float(np.mean([v['data_seconds'] for v in rows])),'validation_seconds':sum(v.get('validation_seconds',0) for v in rows),'component_timings_after_warmup':[v.get('component_timings') for v in measured],'memory':r.get('memory',{'note':'Device peak allocation unavailable for this backend'}),'checkpoint_bytes':sum(f.stat().st_size for f in stage_root.glob('*.pt')),
                    'successful_updates_per_second':sum(not v['optimizer_update_skipped'] for v in measured)/sum(v['step_seconds'] for v in measured),'rough_training_seconds':getattr(c,{'autoencoder':'ae_steps','evolution':'evolution_steps','flow':'flow_steps'}[stage])*sum(v['step_seconds'] for v in measured)/sum(not v['optimizer_update_skipped'] for v in measured) if any(not v['optimizer_update_skipped'] for v in measured) else None,'cost_scope':'Warmup excluded from update distribution. End-to-end updates include scheduled validation/persistence. Setup and final calibration/export separate. Duration estimate excludes these costs; not a deadline guarantee.'}
            t=time.perf_counter();representation(p/'autoencoder/autoencoder.pt',p/'representation-validation.json',c.runtime['device'],subset=min(2,c.validation_count));report['physical_validation_seconds']=time.perf_counter()-t
            t=time.perf_counter();validate_flow(p/'flow/coordinate.pt',p/'flow-validation.json',c.runtime['device'],subset=1,counts=[max(1,c.observation_counts[0])],noises=[0.],particles=1,steps=2,resolution=16)
            report['flow_reconstruction_validation_seconds']=time.perf_counter()-t
            report.update(success=True,stages=records)
    except torch.OutOfMemoryError as error:
        gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
        micro=max(1,o['microbatch']//2);report.update(error='Out of memory profiling '+stage,suggestion={'microbatch':micro,'accumulation':int(np.ceil(c.batch_size/micro)),'activation_checkpointing':True},
           meaning='No capacity/grid/budget reduction applied. Explicitly apply this proposal and profile again; effective batch remains unchanged.')
    report['warning']='Disposable training, same requested architecture, points and microbatch. Only probe update budget and checkpoint/validation frequency change; calibration, architecture, points and scientific objective stay requested. Not an accuracy run or a full-run fit guarantee.'
    return report


def _name(name):
    import re
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,79}',name) or '..' in name:raise ValueError('Use a plain run name (letters, numbers, dash/underscore), no paths')
    return name


class Session:
    def __init__(self,config,name,local_root,persistent_root,allow_cpu_check=False,sync_seconds=300,keep_snapshots=3):
        self.c=config if isinstance(config,CoordinateConfig) else CoordinateConfig(**config).validate();self.name=_name(name)
        self.root=Path(local_root).expanduser().resolve()/self.name;self.persistent=Path(persistent_root).expanduser().resolve()/self.name
        if self.root==self.persistent or self.root.is_relative_to(self.persistent) or self.persistent.is_relative_to(self.root):raise ValueError('Fast local and persistent run roots must be separate')
        self.root.mkdir(parents=True,exist_ok=True);self.allow_cpu_check=allow_cpu_check;self.last_sync=0.;self.snapshot=None
        if not 1<=sync_seconds<=86400 or type(keep_snapshots)!=int or not 1<=keep_snapshots<=20:raise ValueError('Sync seconds must be 1–86400; retain 1–20 persistent snapshots')
        self.sync_seconds=sync_seconds;self.keep_snapshots=keep_snapshots
    def plan_batch_experiment(self,**kwargs):
        from .batch_experiments import plan
        profile=self.root/'profile.json'
        if profile.exists():
            measured=read_json(profile)
            if measured.get('success') and measured.get('requested_config')==digest(self.c.to_dict()) and measured.get('scientific_source')==scientific_source()['sha256'] and all((measured['hardware']['gpu'] or {}).get(k)==(hardware(self.c.runtime['device'])['gpu'] or {}).get(k) for k in ('name','total_bytes','capability','bf16_native')):
                kwargs.setdefault('observed_seconds',{s:r['step_seconds'] for s,r in measured['stages'].items()})
        return plan(self.c,**kwargs)
    def benchmark_batch_sizes(self,background=True,**kwargs):
        from .batch_experiments import start
        from .local import is_locked
        if is_locked(self.root):raise ValueError('Stop the active training worker before benchmarking batch sizes')
        directory=self.root/'batch-experiments';directory.mkdir(exist_ok=True)
        for path in directory.glob('*/experiment.json'):
            if read_json(path)['state'] in ('queued','running','stopping'):raise ValueError('A batch experiment is already active; inspect/cancel it first')
        profile=self.root/'profile.json'
        if profile.exists():
            measured=read_json(profile)
            if measured.get('success') and measured.get('requested_config')==digest(self.c.to_dict()) and measured.get('scientific_source')==scientific_source()['sha256']:kwargs.setdefault('observed_seconds',{s:r['step_seconds'] for s,r in measured['stages'].items()})
        return start(self.c,directory/uuid.uuid4().hex,background=background,**kwargs)
    def batch_experiment(self,identifier):
        from .batch_experiments import status
        if not identifier.isalnum() or len(identifier)!=32:raise ValueError('Invalid experiment ID')
        return status(self.root/'batch-experiments'/identifier)
    def stop_batch_experiment(self,identifier):
        from .batch_experiments import stop
        if not identifier.isalnum() or len(identifier)!=32:raise ValueError('Invalid experiment ID')
        return stop(self.root/'batch-experiments'/identifier)
    def apply_batch_result(self,identifier,recommended=True,selections=None,budget_policy='keep_updates',continuation=False):
        from .batch_experiments import selected_settings
        record=self.batch_experiment(identifier);settings=selected_settings(self.c,record,selections,recommended)
        return apply_training_settings(self,settings,budget_policy,continuation,selection={'method':'benchmark recommendation' if recommended else 'manual benchmark selection','experiment_id':identifier,'science_identity':record['plan']['science_identity'],'mode':record['plan']['mode']})
    def apply_training_settings(self,settings,budget_policy='keep_updates',continuation=False):
        return apply_training_settings(self,settings,budget_policy,continuation)
    def training_summary(self):
        from .training_settings import resolve_batches
        result=resolve_batches(self.c);device,_=RuntimeSpec(**self.c.runtime).resolve();actual,_=precision(options(self.c)['precision'],device)
        for stage,row in result.items():
            path=self.root/stage/'run.json'
            saved=read_json(path) if path.exists() else {};done=saved.get('progress',0);history=saved.get('history',[])
            processed=history[-1].get('effective_examples',0) if history else 0
            row.update(successful_updates_to_date=done,remaining_successful_updates=max(0,row['planned_successful_updates']-done),examples_processed_to_date=processed,approximate_examples=processed+max(0,row['planned_successful_updates']-done)*self.c.batch_size)
        record=read_json(self.root/'session.json') if (self.root/'session.json').exists() else {}
        return {'run_name':self.name,'configuration_state':'applied' if (self.root/'applied-settings.json').exists() else 'configured','training_state':record.get('state','not started'),'next_action':'Configuration application does not launch training','precision':actual,'stages':result}
    def inspect(self):return {'hardware':hardware(self.c.runtime['device'],self.root),'configuration':summary(self.c),'persistent':str(self.persistent)}
    def preflight(self,updates=10,warmup=2):
        from .local import is_locked
        if is_locked(self.root):raise ValueError('Stop the active training worker before resource preflight')
        result=profile(self.c,self.root.parent,self.allow_cpu_check,updates,warmup);write_json(self.root/'profile.json',result);return result
    def sync(self,force=False):
        from .persistence import publish
        if force or time.time()-self.last_sync>=self.sync_seconds:
            started=time.perf_counter();self.snapshot=publish(self.root,self.persistent,keep=self.keep_snapshots);self.last_sync=time.time()
            write_json(self.root/'sync-status.json',{'last_success_unix':self.last_sync,'snapshot':self.snapshot['id'],'bytes':self.snapshot['unique_bytes'],'sync_seconds':time.perf_counter()-started})
        return self.snapshot
    def restore(self):
        from .persistence import restore
        return restore(self.persistent,self.root)
    def stop(self):
        write_json(self.root/'stop-request.json',{'requested_unix':time.time(),'meaning':'Stop at the next complete optimizer boundary and persist'})
    def run(self,*args,**kwargs):
        from pdeflow.local import run_lock
        with run_lock(self.root):return self._run_owned(*args,**kwargs)
    def _run_owned(self,stages='both',mode='new',representation_run='',gate_ratio=1.,override_gate=False,stop_after=None):
        from .batch_experiments import status,ACTIVE
        for file in (self.root/'batch-experiments').glob('*/experiment.json'):
            if status(file.parent)['state'] in ACTIVE:raise ValueError('Finish/cancel the batch experiment before actual training')
        from .training_workflow import coordinate_stages
        applied=self.root/'applied-settings.json'
        if applied.exists() and read_json(applied)['config_identity']!=digest(self.c.to_dict()):raise ValueError('This Session has stale settings; rebind the result of apply_training_settings before training')
        print('Authoritative training settings:',json.dumps(self.training_summary(),indent=2))
        if mode in ('resume','extend'):
            if not (self.root/'session.json').exists():self.restore()
            previous=read_json(self.root/'session.json')
            recorded=previous['config'].get('field_loss',{})
            if self.c.field_loss!=recorded:
                raise ValueError('Field reconstruction loss/settings differ from the recorded run; use its recorded configuration or start a new run. No automatic objective migration.')
        guard(self.c,self.allow_cpu_check)
        if mode not in ('new','resume','extend') or stages not in ('both','representation','evolution','flow'):raise ValueError('Choose supported stage and resume mode')
        if not np.isfinite(gate_ratio) or not 0<gate_ratio<=1:raise ValueError('Gate ratio must be in (0,1]')
        manifest=self.root/'session.json'
        if mode=='new' and manifest.exists():raise FileExistsError('Run already exists; choose resume/extend or another name')
        if mode!='new' and not manifest.exists():self.restore()
        if mode!='new':(self.root/'stop-request.json').unlink(missing_ok=True)
        if max(self.c.ae_steps,self.c.evolution_steps,self.c.flow_steps)>20:
            if not (self.root/'profile.json').exists():raise ValueError('Run the real preflight profile first')
            prof=read_json(self.root/'profile.json')
            stable=lambda g:{k:g[k] for k in ('name','total_bytes','capability','bf16')} if g else None
            if not prof['success'] or prof.get('scientific_source')!=scientific_source()['sha256'] or prof['requested_config']!=digest(self.c.to_dict()) or stable(prof['hardware']['gpu'])!=stable(hardware(self.c.runtime['device'])['gpu']):
                raise ValueError('Profile does not match this configuration/GPU; profile the current settings before substantial training')
        dependency=None
        if representation_run:
            other=Session(self.c,representation_run,self.root.parent,self.persistent.parent,self.allow_cpu_check)
            if not (other.root/'autoencoder/autoencoder.pt').exists():other.restore()
            source=other.root/'autoencoder';target=self.root/'autoencoder'
            if not target.exists():shutil.copytree(source,target,ignore=shutil.ignore_patterns('checkpoints','best.pt','resume.pt'))
            dependency=target/'autoencoder.pt'
        previous=read_json(manifest) if manifest.exists() else {}
        if mode=='extend' and (self.root/'export').is_dir():
            archive=self.root/'budgets'/uuid.uuid4().hex[:12];archive.mkdir(parents=True)
            shutil.copytree(self.root/'export',archive/'export');write_json(archive/'session.json',previous)
        attempts=previous.get('attempts',[])+[{'mode':mode,'stages':stages,'gate_override':override_gate,'gate_ratio':gate_ratio,'config_identity':digest(self.c.to_dict()),'time':time.time()}]
        record={'initial_config':previous.get('initial_config',previous.get('config',self.c.to_dict())),'batch_settings':self.training_summary(),'kind':'pdeflow_colab_session','name':self.name,'config':self.c.to_dict(),'stages':stages,'mode':mode,'state':'starting','gate_ratio':gate_ratio,'gate_override':override_gate,'attempts':attempts,
                'scientific_source':scientific_source(),'environment':environment(RuntimeSpec(**self.c.runtime)),'started_unix':time.time(),
                'persistence_policy':{'sync_seconds':self.sync_seconds,'keep_snapshots':self.keep_snapshots,'boundary':'after completed optimizer-step checkpoints; plus final sync'}}
        write_json(manifest,record)
        def event(e):
            record.update(stage=e['stage'],progress=e['record']['progress'],state='saving/syncing');write_json(manifest,record);self.sync()
            record['state']='running' if e.get('continuing') else 'evaluating' if e['record']['state']=='completed' else 'interrupted'
            write_json(manifest,record)
        try:
            result=coordinate_stages(self.c.to_dict(),self.root,stages,mode,dependency,gate_ratio,override_gate,stop_after,event)
            record.update(result,state=result['state']);write_json(manifest,record)
            if result['state']=='completed' and stages not in ('representation','evolution'):self.export(persist=False)
            return record
        except BaseException as error:
            record.update(state='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error));write_json(manifest,record);raise
        finally:self.sync(True)
    def export(self,persist=True):
        from .coordinate.training import read_checkpoint
        # An interrupted extension may retain an older completed export. The
        # latest optimizer boundary is the authoritative inference state.
        path=self.root/'flow/resume.pt'
        if not path.exists():path=self.root/'flow/coordinate.pt'
        if not path.exists():raise ValueError('No valid conditional-flow state to export yet')
        cp,c=read_checkpoint(path,'flow')
        if cp['progress']<1:raise ValueError('At least one completed optimizer update is required')
        from .coordinate.training import portable_checkpoint
        exported=portable_checkpoint(cp);exported['inference_ready']=True
        exported['training_status']='completed budget' if cp['completed'] else 'partial training; latest saved optimizer boundary'
        destination=self.root/'export/coordinate.pt';atomic_write(destination,lambda f:torch.save(exported,f))
        report={'checkpoint':'coordinate.pt','sha256':file_digest(destination),'config':c.to_dict(),'completed_updates':cp['progress'],'training_status':exported['training_status'],
              'capabilities':cp['capabilities'],'normalization':cp['normalization'],'scientific_source':cp['scientific_source'],'environment':cp['environment'],
              'validation':read_json(self.root/'flow-validation.json') if (self.root/'flow-validation.json').exists() else None,'portable':'Includes decoder, flow, conditioning and latent/physical transforms; no training encoder, optimizer, data or external AE required'}
        write_json(destination.with_suffix('.json'),report)
        if persist:self.sync(True)
        return destination


def discover(persistent_root):
    from .persistence import latest
    result=[]
    for p in sorted(Path(persistent_root).glob('*/snapshots')):
        record=latest(p.parent)
        if record:result.append({'name':p.parent.name,'snapshot':record['id'],'created_unix':record['created_unix'],'progress':record.get('progress')})
    return result


from .training_settings import apply_training_settings
