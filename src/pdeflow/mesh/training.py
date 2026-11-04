"""Shared bounded-batch trainer; all targets are decoded fields or physical nodes."""
from contextlib import nullcontext
from pathlib import Path
import time,math,os,signal,threading,hashlib
import torch
from pdeflow.runtime import RuntimeSpec,write_json,read_json,atomic_write,digest,file_digest,synchronize,environment
from pdeflow.local import run_lock
from pdeflow.coordinate.optimization import precision,optimizer_step,gradient_diagnostics
from pdeflow.coordinate.random_state import capture,restore
from .contracts import MeshConfig,SCHEMA,model_contract,capabilities
from .model import MeshModel
from .data import Data
from .sampling import fgn_noise
from .objectives import loss
from .provenance import scientific_source,read_checkpoint,counts
from .validation import validate
from .monitoring import Reporter
from .diagnostics import require_finite,invalid_tensors,capture_failure,compare_replay,checkpoint_health


def ownership(model,stage):
    c=model.c;model.requires_grad_(True)
    if stage=='dynamics':
        model.conditioner.requires_grad_(False)
        if model.generative is not None:model.generative.requires_grad_(False)
    elif stage in ('flow','diffusion') or stage=='stochastic' and c.freeze_dynamics:
        model.decoder.requires_grad_(False);model.evolution.requires_grad_(False)
    return {'trainable':[name for name,p in model.named_parameters() if p.requires_grad],'frozen':[name for name,p in model.named_parameters() if not p.requires_grad]}


def tensor_digest(module):
    h=hashlib.sha256()
    for name,value in module.state_dict().items():h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def draw(c,b,g,device):
    if c.architecture=='mesh_m3_fgn_v1':noise=(fgn_noise(c,b,g,device),fgn_noise(c,b,g,device))
    else:noise=torch.randn(b,c.nodes+1,2,generator=g).to(device)
    s=torch.rand(b,generator=g).to(device);k=torch.randint(1,c.sampler['diffusion_steps']+1,(b,),generator=g).to(device);q=torch.rand(b,int(c.objective.get('physics_queries',4)),3,generator=g);q[...,:2]=q[...,:2]*q.new_tensor(c.domain['lengths'])+q.new_tensor(c.domain['origins']);q[...,-1]=c.interval[0]+q[...,-1]*(c.interval[1]-c.interval[0])
    return noise,s,k,q.to(device)


def slice_noise(noise,start,end):
    return tuple(tuple(x[start:end] for x in draw) for draw in noise) if isinstance(noise,tuple) else noise[start:end]


def train_stage(config,root,stage,dependency=None,resume=False,stop_after=None,should_stop=None,event=None,extend=False,fixed_subset=False,checkpoint_policy=None,progress=None,benchmark_updates=None):
    c=(config if isinstance(config,MeshConfig) else MeshConfig(**config)).validate();root=Path(root).resolve();path=root/'stages'/stage
    if stage not in c.kind['stages']:raise ValueError('Stage does not belong to this architecture')
    from .settings import resolved
    settings=resolved(c,stage);batch_size=settings['effective_batch'];microbatch=settings['microbatch'];opt=settings['optimizer']
    if benchmark_updates is not None and (resume or type(benchmark_updates)!=int or not 1<=benchmark_updates<=32):raise ValueError('Disposable benchmark requires 1..32 successful updates and cannot resume')
    path.mkdir(parents=True,exist_ok=True);reporter=Reporter(root,c,stage,progress);reporter.update('initialization',force=True);source=scientific_source();device,_=RuntimeSpec(**c.runtime).resolve();policy,amp=precision(c.precision,device);autocast=lambda:torch.autocast(device.type,dtype=amp) if policy!='fp32' else nullcontext()
    reporter.update('initialization',force=True,resolved_precision=policy,autocast_dtype=str(amp),grad_scaler_enabled=policy=='fp16')
    total=c.budgets[stage];limit=benchmark_updates if benchmark_updates is not None else total;started=time.perf_counter();previous=None;old=None
    with run_lock(path):
        if (path/'resume.pt').exists() and not resume:raise FileExistsError('Stage already exists; use explicit resume or a NEW experiment')
        if resume:
            health=checkpoint_health(path/'resume.pt')
            if not health['healthy']:raise ValueError('Saved model/optimizer is nonfinite; select an earlier verified healthy snapshot, never resume this state')
            old,prior=read_checkpoint(path/'resume.pt')
            if old.get('precision',policy)!=policy:raise ValueError('Resolved autocast policy differs from recorded checkpoint; use the same device/precision policy, not a silent precision fallback')
            if model_contract(prior)!=model_contract(c):raise ValueError('Incompatible mesh architecture/objective/data/observation change on resume; start a new experiment')
            if old['scientific_source']['sha256']!=source['sha256']:raise ValueError('Mesh scientific source changed; reload recorded source before resume')
            if prior.budgets[stage]!=total and not (extend and total>prior.budgets[stage]):raise ValueError('Changed budget requires deliberate extension')
            if old.get('dependency'):dependency=(path/old['dependency']['path']).resolve()
        reporter.update('reference/data preparation',force=True)
        data=Data(__import__('dataclasses').replace(c,batch_size=batch_size,microbatch=microbatch),root/'data',create=not resume);torch.manual_seed(c.seed);model=MeshModel(c).to(device);dep=None
        if dependency is not None:
            parent,prior=read_checkpoint(dependency)
            if parent['architecture']!=c.architecture or model_contract(prior)!=model_contract(c) or not (parent['completed'] or benchmark_updates is not None and parent.get('benchmark',{}).get('successful_updates')==benchmark_updates):raise ValueError('Incompatible/incomplete previous mesh stage')
            if parent['scientific_source']['sha256']!=source['sha256']:raise ValueError('Previous mesh stage source changed')
            model.load_state_dict(parent['model'],strict=True);dep={'path':os.path.relpath(dependency,path),'sha256':file_digest(dependency),'stage':parent['stage']}
            if old and dep!=old.get('dependency'):raise ValueError('Frozen prerequisite checkpoint changed')
        warm=(old or {}).get('warm_start') or (parent if dependency is not None else {}).get('warm_start')
        if c.warm_start and not old and dependency is None:
            from .warm_start import initialize
            warm=initialize(c,root,model,data.dataset.metadata)
        owned=ownership(model,stage);parameters=[p for p in model.parameters() if p.requires_grad];optimizer=torch.optim.AdamW(parameters,lr=settings['learning_rate'],weight_decay=0.);scaler=torch.amp.GradScaler('cuda',enabled=policy=='fp16',init_scale=opt.get('initial_scale',65536.));g=torch.Generator().manual_seed(c.seed+431);step=attempted=skipped_total=failed_total=streak=examples=0;history=[];schedule={'schedule':opt.get('schedule','constant'),'warmup':opt.get('warmup',0),'horizon':total,'minimum_ratio':opt.get('minimum_ratio',.1),'learning_rate':settings['learning_rate']}
        if old:
            model.load_state_dict(old['model'],strict=True);optimizer.load_state_dict(old['optimizer']);scaler.load_state_dict(old['scaler']);g.set_state(old['generator']);restore(old['ambient_rng']);torch.set_rng_state(old['torch_rng']);
            if device.type=='cuda' and old.get('cuda_rng'):torch.cuda.set_rng_state_all(old['cuda_rng'])
            failed_total=old.get('failed_attempts',0);step=old['successful_updates'];attempted=old['attempted_updates'];skipped_total=old['skipped_updates'];examples=old['effective_examples'];history=old['history'];schedule=old['scheduler']['settings'];streak=old.get('overflow_streak',0)
        settings['schedule_horizon']=schedule['horizon']
        freeze={name:tensor_digest(module) for name,module in model.named_children() if all(not p.requires_grad for p in module.parameters())}
        setup=time.perf_counter()-started;stop=[False]
        if threading.current_thread() is threading.main_thread():previous=signal.signal(signal.SIGINT,lambda *_:stop.__setitem__(0,True))
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        record={'kind':'mesh_stage_run','architecture':c.architecture,'stage':stage,'config':c.to_dict(),'config_identity':digest(c.to_dict()),'resolved_training_settings':settings,'scientific_source':source,'dataset_identity':data.dataset.metadata['sha256'],'state':'running','pid':os.getpid(),'progress':step,'budget':total,'ownership':owned,'precision':policy,'autocast_dtype':str(amp),'parameter_dtype':'float32','scaler_enabled':policy=='fp16','numerical_policy':'mesh_fp32_products_and_reductions_v1','dependency':dep,'setup_seconds':setup,'parameters':sum(p.numel() for p in model.parameters()),'scope':'Physical decoded targets; no full-state encoder/AE latent targets','fixed_subset_diagnostic':fixed_subset}
        print('Resolved mesh optimizer/batch:',stage,__import__('json').dumps(settings),flush=True)
        write_json(path/'run.json',record);last_validation=None;batch=None;draws=None;rng=None;pending_attempt=False
        reporter.update('training',force=True,successful_updates=step,attempted_updates=attempted,skipped_updates=skipped_total,failed_attempts=failed_total,setup_seconds=setup)
        if event:event({'stage':stage,'record':record,'path':str(path),'initialization':True})
        def save(completed,continuing=False):
            nonlocal last_validation
            require_finite({'model':model.state_dict(),'optimizer':optimizer.state_dict()},'checkpoint model/optimizer state')
            reporter.update('checkpointing',force=True,successful_updates=step)
            for name,sha in freeze.items():
                if tensor_digest(getattr(model,name))!=sha:raise RuntimeError('Frozen module changed: '+name)
            cp={'schema':SCHEMA,'state_contract':'normalized_physical_nodes_u0_v0_log_parameter_v1','architecture':c.architecture,'config':c.to_dict(),'config_identity':digest(c.to_dict()),'resolved_training_settings':settings,'benchmark':{'successful_updates':step if benchmark_updates is not None else None,'scope':'Disposable measurement only; never a production completed stage'} if benchmark_updates is not None else {},'stage':stage,'completed':completed,'inference_ready':benchmark_updates is None and stage==c.kind['stages'][-1] and step>0,'model':{k:v for k,v in model.state_dict().items() if not dep or k.split('.')[0] not in freeze},'model_partial_frozen_modules':list(freeze) if dep else [],'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'scheduler':{'settings':schedule,'completed_updates':step},'generator':g.get_state(),'ambient_rng':capture(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else [],'successful_updates':step,'attempted_updates':attempted,'skipped_updates':skipped_total,'failed_attempts':failed_total,'overflow_streak':streak,'effective_examples':examples,'counter_semantics':'successful_skipped_failed_v2','normalizations':data.norms,'normalization_fitted_ids':data.dataset.metadata['fitted_ids'],'training_ids':[i for i in data.dataset.metadata['ids'] if i.startswith('train:')],'dataset_identity':data.dataset.metadata['sha256'],'dataset_root':os.path.relpath(root/'data',path),'scientific_source':source,'environment':environment(RuntimeSpec(**c.runtime)),'precision':policy,'autocast_dtype':str(amp),'parameter_dtype':'float32','scaler_enabled':policy=='fp16','numerical_policy':'mesh_fp32_products_and_reductions_v1','history':history[-2000:],'history_total':attempted,'dependency':dep,'ownership':owned,'warm_start':warm,'frozen_hashes':freeze,'validation':last_validation,'capabilities':capabilities(c),'training_seconds':time.perf_counter()-started,'setup_seconds':setup,'noise_and_reference_policy':'Reference/observation seed depends on declared split and successful stage update; candidate order independent. Stochastic generator/scaler/RNG states persisted separately.','sampling':c.sampler,'resolved_field_loss':__import__('dataclasses').asdict(__import__('pdeflow.field_losses',fromlist=['settings']).settings(c.field_loss)),'resolved_reference_configuration':c.reference_config().to_dict(),'training_only_information':'Dynamics stage uses true clean physical initial node values; production generation receives sparse observations and disclosed physics only.'}
            # Keep the immediately preceding validated checkpoint independently.
            if (path/'resume.pt').exists():
                import shutil
                previous_good=path/'healthy/previous.pt';previous_good.parent.mkdir(exist_ok=True);atomic_write(previous_good,lambda f:f.write((path/'resume.pt').read_bytes()))
            atomic_write(path/'resume.pt',lambda f:torch.save(cp,f))
            # Standalone export is built once by the session/export layer. Stage
            # resume references frozen prerequisite modules instead of copying them.
            record.update(state='completed' if completed else 'running' if continuing else 'interrupted',progress=step,successful_updates=step,attempted_updates=attempted,skipped_updates=skipped_total,effective_examples=examples,history=history[-200:],training_seconds=time.perf_counter()-started,validation=last_validation,memory={'cuda_peak_allocated':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,'cuda_peak_reserved':torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None,'mps_current_allocated':torch.mps.current_allocated_memory() if device.type=='mps' else None,'scope':'CUDA peak; MPS current allocation is not peak'})
            write_json(path/'history.json',history[-2000:]);write_json(path/'run.json',record)
            if event:event({'stage':stage,'record':record,'path':str(path)})
            return cp
        try:
            while step<limit:
                if stop[0] or should_stop and should_stop() or stop_after is not None and step>=stop_after:break
                attempt_started=time.perf_counter();attempted+=1;pending_attempt=True;batch=None;draws=None
                rng={'ambient':capture(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if device.type=='cuda' else [],'generator':g.get_state()}
                tick=time.perf_counter();batch=data.prepare(0 if fixed_subset else step,fixed=fixed_subset);reference_seconds=time.perf_counter()-tick;tick=time.perf_counter();batch=batch.to(device);
                if fixed_subset:g.manual_seed(c.seed+431)
                noise,s,k,q=draw(c,batch_size,g,device);draws=(noise,s,k,q)
                require_finite({key:value for key,value in batch.__dict__.items() if torch.is_tensor(value)},'reference targets / queries / physical inputs')
                points,features,valid,quantities=batch.observations;require_finite((points[valid],features[valid]),'valid observations')
                transfer_seconds=time.perf_counter()-tick;synchronize(device);update_started=time.perf_counter();optimizer.zero_grad(set_to_none=True);parts={};objective=0.;lr=schedule['learning_rate'];warmup=schedule['warmup']
                if warmup and step<warmup:lr*=float(step+1)/warmup
                elif schedule['schedule']=='cosine':
                    phase=min(1.,max(0.,(step-warmup)/max(1,schedule['horizon']-warmup)));lr*=schedule['minimum_ratio']+(1-schedule['minimum_ratio'])*(1+math.cos(math.pi*phase))/2
                for group in optimizer.param_groups:group['lr']=lr
                gpu_start=gpu_end=None
                if device.type=='cuda':gpu_start=torch.cuda.Event(enable_timing=True);gpu_end=torch.cuda.Event(enable_timing=True);gpu_start.record()
                for start in range(0,batch_size,microbatch):
                    end=min(batch_size,start+microbatch);part=batch.slice(start,end)
                    with autocast():values,components=loss(model,part,stage,data.norms,slice_noise(noise,start,end),s[start:end],k[start:end],q[start:end])
                    require_finite({'objective':values,'components':components},'forward objective/components before backward')
                    scaler.scale(values.float().sum()/batch_size).backward();objective+=float(values.detach().float().sum())/batch_size
                    for key,value in components.items():parts[key]=parts.get(key,0.)+float(value.detach().float().sum())/batch_size
                if not math.isfinite(objective):raise FloatingPointError('Nonfinite mesh objective; optimizer not stepped')
                grad,finite,skipped=optimizer_step(optimizer,scaler,parameters,opt.get('clip',10.),policy)
                if gpu_end is not None:gpu_end.record()
                synchronize(device);compute_seconds=time.perf_counter()-update_started;pending_attempt=False
                if finite:step+=1;examples+=batch_size;streak=0
                else:skipped_total+=1;streak+=1
                row={'step':step,'attempted_update':attempted,'skipped_updates':skipped_total,'failed_attempts':failed_total,'training_loss':objective,'components':parts,'learning_rate':lr,'gradients_finite':finite,'gradient_norm':float(grad) if finite else None,'optimizer_update_skipped':not finite,'scaler_scale':scaler.get_scale(),'successful_updates':step,'effective_examples':examples,'family_example_counts':{name:int(batch.families.eq(i).sum()) for i,name in enumerate(('heat','wave'))},'actual_forward_microbatch':min(microbatch,batch_size),'effective_batch':batch_size,'forward_microbatches':math.ceil(batch_size/microbatch),'reference_host_seconds':reference_seconds,'transfer_enqueue_host_seconds':transfer_seconds,'synchronized_compute_update_seconds':compute_seconds,'gpu_forward_backward_optimizer_ms':gpu_start.elapsed_time(gpu_end) if gpu_start is not None else None,'step_seconds':reference_seconds+transfer_seconds+compute_seconds,'timing_scope':'Per-attempt synchronized wall time; validation/persistence separate. GPU events only at profiling boundaries.','physical_target_times':batch.times.detach().cpu().tolist(),'observation_times_minmax':[[float(x[:,2].min()),float(x[:,2].max())] if len(x) else [] for x in [points[valid] for points,valid in zip(batch.observations[0].detach().cpu(),batch.observations[2].detach().cpu())]]}
                if not finite:row['overflow_diagnostics']=gradient_diagnostics(model.named_parameters())
                if finite and (step==limit if benchmark_updates is not None else step%c.validation_every==0 or step==total):
                    reporter.update('validation',force=True,successful_updates=step)
                    tick=time.perf_counter();last_validation=validate(model,data,stage,preview_root=path,update=step);row['validation_seconds']=time.perf_counter()-tick;row['validation_loss']=last_validation['uniform_domain_normalized_mse'];write_json(path/'validation.json',last_validation)
                history.append(row)
                import json
                with (path/'history-live.jsonl').open('a') as journal:journal.write(json.dumps(row,allow_nan=False)+'\n')
                if benchmark_updates is None and finite and (step%c.checkpoint_every==0 or checkpoint_policy and checkpoint_policy(stage,step)):
                    tick=time.perf_counter();save(False,True);row['persistence_seconds']=time.perf_counter()-tick
                reporter.row(row)
                row['end_to_end_seconds']=time.perf_counter()-attempt_started
                if streak>=opt.get('max_consecutive_overflows',8):raise FloatingPointError('Persistent nonfinite FP16 gradients after bounded scaler recovery; inspect overflow diagnostics')
            completed=benchmark_updates is None and step>=total;save(completed);reporter.update('completed' if completed else 'interrupted',force=True,successful_updates=step,attempted_updates=attempted,skipped_updates=skipped_total,failed_attempts=failed_total)
            return record
        except BaseException as error:
            import traceback
            if pending_attempt:failed_total+=1
            gradients=gradient_diagnostics(model.named_parameters())
            optimizer.zero_grad(set_to_none=True)
            context={'config':c.to_dict(),'source':source,'stage':stage,'normalizations':data.norms,'gradient_diagnostics_before_clear':gradients,'successful_updates':step,'attempted_updates':attempted,'skipped_updates':skipped_total,'failed_attempts':failed_total,'precision':policy,'autocast_dtype':str(amp),'scaler':scaler.state_dict(),'phase':reporter.record.get('phase'),'batch_generation_index':0 if fixed_subset else step,'fixed_subset':fixed_subset,'data_seed_formula':'seed + generation_index*100003; TRAIN split offset0','dataset_identity':data.dataset.metadata['sha256']}
            diagnostics_error=None
            try:
                capture_failure(path,model,optimizer,batch,draws,rng,context,error,lambda:compare_replay(model,batch,draws,data.norms,stage,policy,amp,rng,scaler.state_dict()))
            except BaseException as capture_error:diagnostics_error=str(capture_error)
            record.update(state='failed',error=str(error),traceback=traceback.format_exc(),progress=step,attempted_updates=attempted,skipped_updates=skipped_total,failed_attempts=failed_total,diagnostics_error=diagnostics_error,last_checkpoint_preserved=True)
            try:
                write_json(path/'error.json',{'error':str(error),'traceback':record['traceback'],'diagnostics_error':diagnostics_error});write_json(path/'run.json',record)
                reporter.update('failed',force=True,error=str(error),diagnostics_error=diagnostics_error,successful_updates=step,attempted_updates=attempted,skipped_updates=skipped_total,failed_attempts=failed_total)
            except Exception as reporting_error:print('Failure reporting also failed; original exception retained:',reporting_error,flush=True)
            raise
        finally:
            if previous is not None:signal.signal(signal.SIGINT,previous)
