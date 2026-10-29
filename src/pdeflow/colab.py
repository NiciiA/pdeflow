"""Notebook-facing orchestration. Learning stays in coordinate.training.train."""
from pathlib import Path
from dataclasses import replace
import gc,os,time,tempfile,json,shutil,uuid,hashlib
import numpy as np
import torch
from pdeflow.runtime import read_json,write_json,atomic_write,file_digest,digest,environment,RuntimeSpec
from pdeflow.coordinate.contracts import CoordinateConfig
from pdeflow.coordinate.optimization import options,precision,scientific_source,model_contract


def preset(name='large',device='cuda',precision_policy='auto',overrides=None):
    c=CoordinateConfig(dimension=2,reference_shape=(64,64),times=(0.,.05,.1,.15),width=512,tokens=128,layers=8,heads=8,
       train_count=65536,validation_count=2048,test_count=2048,ae_steps=100000,flow_steps=100000,encoder_points=1024,query_points=512,
       batch_size=8,learning_rate=.0003,checkpoint_every=500,observation_counts=(1,128),unconditional_probability=0.,data_mode='procedural',temporal_mode='continuous',time_interval=(0.,1.),
       distribution={'max_wave':4,'amplitude':[.4,1.2],'background':[-.5,.5],'width':[.08,.18]},
       runtime={'device':device,'dtype':'float32','threads':2},training_options={'precision':precision_policy,'calibration_count':512,
           'validation_every':500,'validation_subset':12,'broad_every':5000,'broad_subset':48,'observation_locations':'scattered',
           'autoencoder':{'warmup':1000,'schedule':'cosine'},'flow':{'warmup':1000,'schedule':'cosine'}})
    if name=='pipeline-check':c=replace(c,reference_shape=(8,8),width=16,tokens=4,layers=1,heads=2,train_count=12,validation_count=6,test_count=6,
          ae_steps=2,flow_steps=2,encoder_points=32,query_points=32,batch_size=2,checkpoint_every=1,observation_counts=(1,8),
          distribution={'max_wave':1,'width':[.4,.5]},training_options={'precision':precision_policy,'calibration_count':6,'validation_every':1,'validation_subset':2,'broad_every':20,'broad_subset':6,'observation_locations':'scattered'})
    elif name=='medium':c=replace(c,reference_shape=(32,32),width=128,tokens=32,layers=4,heads=8,train_count=8192,validation_count=256,test_count=256,
          ae_steps=10000,flow_steps=10000,encoder_points=512,query_points=256,observation_counts=(1,64),distribution={'max_wave':4,'width':[.1,.2]})
    elif name=='extra-large':c=replace(c,width=768,tokens=192,layers=12,heads=12,train_count=131072,validation_count=4096,test_count=4096,
          ae_steps=200000,flow_steps=200000,encoder_points=2048,query_points=1024,training_options={**c.training_options,'microbatch':4,'accumulation':2,'activation_checkpointing':True})
    elif name not in ('large','custom'):raise ValueError('Preset: pipeline-check, medium, large, extra-large or custom')
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
        p=torch.cuda.get_device_properties(0);free,total=torch.cuda.mem_get_info();gpu={'name':p.name,'total_bytes':total,'free_bytes':free,'bf16':torch.cuda.is_bf16_supported(),'fp16':True,'capability':list(torch.cuda.get_device_capability())}
    return {'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'numpy':np.__version__,
            'gpu':gpu,'host_ram_bytes':ram,'available_host_ram_bytes':available,'local_disk_free_bytes':shutil.disk_usage(disk_root or Path.cwd()).free,'selected_device':device}


def guard(c,allow_cpu_check=False):
    report=hardware(c.runtime['device'])
    if c.runtime['device']=='cuda' and not torch.cuda.is_available():raise ValueError('No CUDA GPU. Select Runtime → Change runtime type → GPU, or explicitly select CPU with the pipeline-check preset.')
    device,_=RuntimeSpec(**c.runtime).resolve()
    small=c.width<=32 and c.train_count<=64 and max(c.ae_steps,c.flow_steps)<=20
    if c.data_mode=='materialized' and np.prod(c.reference_shape)*len(c.times)*4*(c.train_count+c.validation_count+c.test_count)>512*1024**2:
        raise ValueError('This substantial Colab run must use procedural data; materialized legacy data would exceed the bounded host-memory policy')
    if device.type!='cuda' and not (allow_cpu_check and small):raise ValueError('Substantial Colab runs require CUDA. Choose Runtime → Change runtime type → GPU. CPU is only allowed explicitly for a small pipeline check.')
    precision(options(c)['precision'],device)
    return report


def summary(c):
    from pdeflow.coordinate.model import Autoencoder,Velocity
    with torch.device('meta'):
        ae=Autoencoder(c);flow=Velocity(c)
    counts={'representation':sum(p.numel() for p in ae.parameters()),'flow':sum(p.numel() for p in flow.parameters())}
    o=options(c);nodes=int(np.prod(c.reference_shape));bytes_example=nodes*(c.animation_frames if c.temporal_mode=='continuous' else len(c.times))*4
    return {'architecture':'Set encoder / frozen latent representation / conditional latent transformer / coordinate decoder','parameters':counts,
      'field_reconstruction_loss':__import__('dataclasses').asdict(__import__('pdeflow.field_losses',fromlist=['settings']).settings(c.field_loss)),
      'supervision_query_cost':{'domain_queries':c.query_points,'additional_region_queries':__import__('pdeflow.field_losses',fromlist=['settings']).settings(c.field_loss).region_queries if __import__('pdeflow.field_losses',fromlist=['settings']).settings(c.field_loss).type=='region_balanced_mse' else 0},
      'config':c.to_dict(),'training_examples':c.train_count,'validation_examples':c.validation_count,'untouched_test_examples':c.test_count,
      'reference_shape':list(c.reference_shape),'temporal_mode':c.temporal_mode,'time_interval':list(c.interval),'animation_frames':c.animation_frames,'legacy_snapshot_times':list(c.times) if c.temporal_mode!='continuous' else None,'effective_batch':c.batch_size,'microbatch':o['microbatch'],'accumulation':o['accumulation'],
      'updates':{'representation':c.ae_steps,'flow':c.flow_steps},'hypothetical_materialized_float32_bytes':bytes_example*(c.train_count+c.validation_count+c.test_count),
      'bounded_reference_cache_bytes':8*(nodes*52+c.encoder_points*16) if c.temporal_mode=='continuous' else bytes_example*8,'cache_estimate_definition':'Eight initial grids, Fourier coefficients/waves and fixed encoder sets; current batches/temporary Fourier chunks additional' if c.temporal_mode=='continuous' else 'Eight full legacy examples','estimated_weight_bytes':4*sum(counts.values()),'rough_adam_weights_moments_bytes':12*sum(counts.values()),
      'estimates':'Parameter/storage arithmetic only; activations, attention, AMP workspaces and allocator overhead need actual profiling. No fit or speed guarantee.'}


def gallery(c,output):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pdeflow.coordinate.dataset import example,recipe
    n=min(len(recipe(c)['families']),c.train_count);fig,axes=plt.subplots(2,n,figsize=(2.5*n,5),squeeze=False)
    records=[]
    for i in range(n):
        a=example(c,'train',i,dtype=np.float64);u=(np.stack([a['solver'].frame(t,a['D'],c.interval) for t in c.interval]) if c.temporal_mode=='continuous' else a['values'].reshape(len(c.times),*c.reference_shape));f=example(c,'train',i,tuple(2*x for x in c.reference_shape),dtype=np.float64);fine=np.stack([f['solver'].frame(t,f['D'],c.interval) for t in c.interval]) if c.temporal_mode=='continuous' else f['values'].reshape(len(c.times),*[2*x for x in c.reference_shape])
        error=float(np.sqrt(np.mean((u-fine[:,::2,::2])**2)));records.append({'family':a['family'],'id':a['id'],'D':a['D'],'native_vs_2x_reference_rmse':error})
        lo,hi=float(u.min()),float(u.max())
        for j,t in enumerate((0,-1)):axes[j,i].imshow(u[t].T,origin='lower',vmin=lo,vmax=hi,cmap='viridis');axes[j,i].set_title(f'{a["family"]}\nt={(c.interval if c.temporal_mode=='continuous' else c.times)[t]:g}');axes[j,i].axis('off')
    fig.tight_layout();Path(output).parent.mkdir(parents=True,exist_ok=True);fig.savefig(output);plt.close(fig)
    write_json(Path(output).with_suffix('.json'),{'records':records,'meaning':'Same deterministic underlying field at native / twice-finer FFT reference. Discretization discrepancy, not learned error; inspect before substantial training.'});return records


def profile(c,local_root,allow_cpu_check=False,updates=3):
    guard(c,allow_cpu_check);requested=digest(c.to_dict());o=options(c);report={'requested_config':requested,'hardware':hardware(c.runtime['device']),'success':False}
    from pdeflow.coordinate.training import train
    from pdeflow.coordinate.validation import representation,flow as validate_flow
    dev,_=RuntimeSpec(**c.runtime).resolve();devices=list(range(torch.cuda.device_count())) if dev.type=='cuda' else []
    try:
        with torch.random.fork_rng(devices=devices),tempfile.TemporaryDirectory(prefix='pdeflow-profile-',dir=local_root) as temporary:
            p=Path(temporary);pc=replace(c,ae_steps=updates,flow_steps=max(2,updates),checkpoint_every=updates,
              training_options={**c.training_options,'calibration_count':min(8,c.train_count),'validation_every':updates,'validation_subset':min(2,c.validation_count),'broad_every':999999})
            records={}
            for stage in ('autoencoder','flow'):
                stage_root=p/stage
                r=train(pc.to_dict(),stage_root,stage,p/'autoencoder/autoencoder.pt' if stage=='flow' else None)
                rows=read_json(stage_root/'history.json');measured=rows[1:] or rows;seconds=float(np.median([v['step_seconds'] for v in measured]));records[stage]={'step_seconds':seconds,
                    'data_seconds':float(np.mean([v['data_seconds'] for v in rows])),'validation_seconds':sum(v.get('validation_seconds',0) for v in rows),'memory':r['memory'],'checkpoint_bytes':r['checkpoint_bytes'],
                    'rough_training_seconds':seconds*(c.ae_steps if stage=='autoencoder' else c.flow_steps),'cost_scope':'Warmup first optimizer step excluded from median; validation/checkpoint/sync costs reported separately, not in duration estimate'}
            t=time.perf_counter();representation(p/'autoencoder/autoencoder.pt',p/'representation-validation.json',c.runtime['device'],subset=min(2,c.validation_count));report['physical_validation_seconds']=time.perf_counter()-t
            t=time.perf_counter();validate_flow(p/'flow/coordinate.pt',p/'flow-validation.json',c.runtime['device'],subset=1,counts=[max(1,c.observation_counts[0])],noises=[0.],particles=1,steps=2,resolution=16)
            report['flow_reconstruction_validation_seconds']=time.perf_counter()-t
            report.update(success=True,stages=records)
    except torch.OutOfMemoryError as error:
        gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
        micro=max(1,o['microbatch']//2);report.update(error='Out of memory profiling '+stage,suggestion={'microbatch':micro,'accumulation':int(np.ceil(c.batch_size/micro)),'activation_checkpointing':True},
           meaning='No capacity/grid/budget reduction applied. Explicitly apply this proposal and profile again; effective batch remains unchanged.')
    report['warning']='Disposable training, same requested architecture, points and microbatch. Reduced profiling update/calibration/validation counts are recorded; not model training or a guarantee of full-run fit.'
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
    def inspect(self):return {'hardware':hardware(self.c.runtime['device'],self.root),'configuration':summary(self.c),'persistent':str(self.persistent)}
    def preflight(self):
        result=profile(self.c,self.root.parent,self.allow_cpu_check);write_json(self.root/'profile.json',result);return result
    def sync(self,force=False):
        from .persistence import publish
        if force or time.time()-self.last_sync>=self.sync_seconds:
            self.snapshot=publish(self.root,self.persistent,keep=self.keep_snapshots);self.last_sync=time.time()
            write_json(self.root/'sync-status.json',{'last_success_unix':self.last_sync,'snapshot':self.snapshot['id'],'bytes':self.snapshot['unique_bytes']})
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
        from .training_workflow import coordinate_stages
        if mode in ('resume','extend'):
            if not (self.root/'session.json').exists():self.restore()
            previous=read_json(self.root/'session.json')
            recorded=previous['config'].get('field_loss',{})
            if self.c.field_loss!=recorded:
                raise ValueError('Field reconstruction loss/settings differ from the recorded run; use its recorded configuration or start a new run. No automatic objective migration.')
        guard(self.c,self.allow_cpu_check)
        if mode not in ('new','resume','extend') or stages not in ('both','representation','flow'):raise ValueError('Choose supported stage and resume mode')
        if not np.isfinite(gate_ratio) or not 0<gate_ratio<=1:raise ValueError('Gate ratio must be in (0,1]')
        manifest=self.root/'session.json'
        if mode=='new' and manifest.exists():raise FileExistsError('Run already exists; choose resume/extend or another name')
        if mode!='new' and not manifest.exists():self.restore()
        if mode!='new':(self.root/'stop-request.json').unlink(missing_ok=True)
        if max(self.c.ae_steps,self.c.flow_steps)>20:
            if not (self.root/'profile.json').exists():raise ValueError('Run the real preflight profile first')
            prof=read_json(self.root/'profile.json')
            stable=lambda g:{k:g[k] for k in ('name','total_bytes','capability','bf16')} if g else None
            if not prof['success'] or prof['requested_config']!=digest(self.c.to_dict()) or stable(prof['hardware']['gpu'])!=stable(hardware(self.c.runtime['device'])['gpu']):
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
        record={'kind':'pdeflow_colab_session','name':self.name,'config':self.c.to_dict(),'stages':stages,'mode':mode,'state':'starting','gate_ratio':gate_ratio,'gate_override':override_gate,'attempts':attempts,
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
            if result['state']=='completed' and stages!='representation':self.export(persist=False)
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
        exported={k:v for k,v in cp.items() if k not in ('optimizer','scheduler','scaler','rng','torch_rng','cuda_rng')};exported['inference_ready']=True
        exported['training_status']='completed budget' if cp['completed'] else 'partial training; latest saved optimizer boundary'
        destination=self.root/'export/coordinate.pt';atomic_write(destination,lambda f:torch.save(exported,f))
        report={'checkpoint':'coordinate.pt','sha256':file_digest(destination),'config':c.to_dict(),'completed_updates':cp['progress'],'training_status':exported['training_status'],
              'capabilities':cp['capabilities'],'normalization':cp['normalization'],'scientific_source':cp['scientific_source'],'environment':cp['environment'],
              'validation':read_json(self.root/'flow-validation.json') if (self.root/'flow-validation.json').exists() else None,'portable':'Includes frozen representation, flow and latent/physical transforms; optimizer/data/external AE not required'}
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
