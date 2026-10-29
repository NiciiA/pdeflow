"""One shared bounded trainer for local/CLI/Colab coordinate models.

Updates are completed optimizer steps; checkpoints never contain partial gradients.
Full fields are reference-only AE targets, never flow conditioning features.
"""
from pathlib import Path
from contextlib import nullcontext
import math,time,signal,threading
import numpy as np
import torch
from pdeflow.runtime import RuntimeSpec,atomic_write,write_json,read_json,digest,file_digest,code_identity,environment,synchronize
from pdeflow.local import run_lock,now
from .contracts import CoordinateConfig,padded_observations
from .model import Autoencoder,Velocity
from .dataset import Dataset,query_subset
from .optimization import options,precision,learning_rate,scientific_source,model_contract,resume_contract


def _observations(c,dataset,y,parameters,g,force_count=None):
    from .training import training_observations
    o=options(c);norm=dataset.metadata['normalization']
    if c.temporal_mode=='continuous':
        from pdeflow.heat_continuous import FourierHeat
        from pdeflow.heat_placement import place
        from .temporal import sample_times
        items=[];mixture=c.observation_placement or {'random':.5,'even':.25,'heat':.25}
        names=sorted(mixture);prob=torch.tensor([mixture[k] for k in names])
        for initial,D in zip(y,parameters):
            n=int(torch.randint(c.observation_counts[0],c.observation_counts[1]+1,(),generator=g))
            if float(torch.rand((),generator=g))<c.unconditional_probability:n=0
            if force_count is not None:n=force_count
            solver=FourierHeat(initial.numpy().reshape(c.reference_shape),c.lengths,c.origins)
            t=sample_times(c,n,g,'observation');design=float(sample_times(c,1,g,'observation')[0])
            if float(torch.rand((),generator=g))<.5:t=torch.full_like(t,design)
            policy=names[int(torch.multinomial(prob,1,generator=g))];seed=int(torch.randint(2**31,(),generator=g))
            xy=place(n,policy,seed,c.lengths,c.origins,field=solver.frame(design,float(D),c.interval,shape=(32,32)) if policy=='heat' else None)[0] if n else np.empty((0,2))
            q=torch.tensor(np.c_[xy,t.numpy()],dtype=torch.float32)
            sigma=float(torch.empty(()).uniform_(*c.noise_range,generator=g));clean=torch.tensor(solver.evaluate(q.numpy(),float(D),c.interval),dtype=torch.float32)
            items.append((q,clean+sigma*torch.randn(n,generator=g),torch.full((n,),sigma)))
        return padded_observations(items,norm,'cpu')
    if o['observation_locations']=='grid':return training_observations(c,dataset.q,y,g,norm,'cpu',force_count=force_count)
    items=[];shape=c.reference_shape;space=int(np.prod(shape));q=dataset.q
    k=[2*np.pi*np.fft.fftfreq(n,d=L/n) for n,L in zip(shape,c.lengths)]
    for values,D in zip(y,parameters):
        n=int(torch.randint(c.observation_counts[0],c.observation_counts[1]+1,(),generator=g))
        if float(torch.rand((),generator=g))<c.unconditional_probability:n=0
        if force_count is not None:n=force_count
        xy=torch.rand(n,2,generator=g)*torch.tensor(c.lengths)+torch.tensor(c.origins)
        ti=torch.randint(len(c.times),(n,),generator=g);coords=torch.cat([xy,torch.tensor(c.times)[ti,None]],1)
        clean=np.empty(n);coeff=np.fft.fftn(values[:space].numpy().reshape(shape))/space
        for t in ti.unique().tolist():
            ix=np.flatnonzero(ti.numpy()==t);points=xy[ix].numpy()-np.asarray(c.origins)
            evolved=coeff*np.exp(-float(D)*(k[0][:,None]**2+k[1][None,:]**2)*c.times[t])
            clean[ix]=np.einsum('ni,ij,nj->n',np.exp(1j*points[:,0,None]*k[0]),evolved,np.exp(1j*points[:,1,None]*k[1]),optimize=True).real
        sigma=float(torch.empty(()).uniform_(*c.noise_range,generator=g));noisy=torch.tensor(clean,dtype=torch.float32)+sigma*torch.randn(n,generator=g)
        items.append((coords,noisy,torch.full((n,),sigma)))
    return padded_observations(items,norm,'cpu')


def _moments(ae,data,c,device,autocast):
    from .training import select_context,normalized_parameter
    n=data.metadata['normalization'];s=s2=None;count=0
    ae.eval()
    with torch.no_grad():
        for start in range(0,min(c.train_count,options(c)['calibration_count']),options(c)['microbatch']):
            y,d,items=data.batch(range(start,min(start+options(c)['microbatch'],c.train_count,options(c)['calibration_count'])))
            if c.temporal_mode=='continuous':
                cq=torch.tensor(np.stack([i['context'] for i in items]));cv=(torch.tensor(np.stack([i['context_values'] for i in items]))-n['mean'])/n['scale']
            else:
                cq,cv=select_context(data.q,(y-n['mean'])/n['scale'],c);cq=cq[None].expand(len(y),-1,-1)
            with autocast():z=ae.encode(cq.to(device),cv.to(device),normalized_parameter(c,d).to(device)).float()
            # Train-only calibration statistics are explicit CPU float64 work;
            # MPS does not support float64 model-device tensors.
            statistics=z.cpu().double()
            a=statistics.sum((0,1),keepdim=True);b=statistics.square().sum((0,1),keepdim=True)
            s=a if s is None else s+a;s2=b if s2 is None else s2+b;count+=z.shape[0]*z.shape[1]
    mean=(s/count).float();scale=(s2/count-(s/count).square()).clamp_min(0).sqrt().float().clamp_min(.01)
    return mean.cpu(),scale.cpu()


def train(config,output,stage,dependency=None,resume=False,stop_after=None,should_stop=None,event=None,extend=False):
    setup_started=time.perf_counter()
    from .training import read_checkpoint,select_context,normalized_parameter,SCHEMA
    c=CoordinateConfig(**config).validate();o=options(c)
    from pdeflow.field_losses import settings as loss_settings,AnalysisCache,grouped_queries,differentiable_loss
    field_loss=loss_settings(c.field_loss);analysis_cache=AnalysisCache();query_stats=[]
    if stage not in ('autoencoder','flow'):raise ValueError('Stage must be autoencoder or flow')
    if stop_after is not None and (type(stop_after)!=int or stop_after<1):raise ValueError('stop_after must be positive')
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=True);runtime=RuntimeSpec(**c.runtime);device,dtype=runtime.resolve()
    policy,amp_dtype=precision(o['precision'],device)
    autocast=lambda:torch.autocast(device.type,dtype=amp_dtype) if policy!='fp32' else nullcontext()
    total=c.ae_steps if stage=='autoencoder' else c.flow_steps;settings=o[stage];saved=None;source_override=False
    with run_lock(root):
        record_path=root/'run.json';checkpoint=root/'resume.pt'
        if not resume and any(p.name!='.run.lock' for p in root.iterdir()):raise FileExistsError('Fresh coordinate stage output must be empty; choose a new directory or explicitly resume')
        if resume and checkpoint.exists():
            saved,_=read_checkpoint(checkpoint,stage)
            if saved.get('engine_version')!=2:raise ValueError('Legacy resume checkpoint needs its original trainer/source; portable inference remains supported')
            settings,source_override=resume_contract(saved,c,stage,extend)
        elif resume and record_path.exists():
            prior=read_json(record_path)
            if prior.get('config_identity')!=digest(c.to_dict()) or prior.get('scientific_source')!=scientific_source():raise ValueError('Preparation restart requires its recorded config/scientific source')
        elif resume:raise ValueError('No saved stage to resume')
        record={'schema_version':1,'kind':'coordinate_stage_run','engine_version':2,'id':root.name,'stage':stage,'specialist':'heat','state':'preparing','pid':__import__('os').getpid(),
                'config':c.to_dict(),'config_identity':digest(c.to_dict()),'source':code_identity(),'scientific_source':scientific_source(),'progress':saved['progress'] if saved else 0,'budget':total,
                'created_at':read_json(record_path).get('created_at',now()) if record_path.exists() else now(),'export':str(root/('autoencoder.pt' if stage=='autoencoder' else 'coordinate.pt')),'checkpoint':str(checkpoint),'precision':policy}
        write_json(record_path,record);write_json(root/'config.json',c.to_dict())
        if stage=='autoencoder':data=Dataset(c,root,create=not bool(saved));dep=None;ae_cp=None
        else:
            if saved:dependency=(root/saved['dependency']['path']).resolve()
            if not dependency:raise ValueError('Flow requires a compatible frozen representation')
            dependency=Path(dependency).resolve();ae_cp,ae_c=read_checkpoint(dependency,'autoencoder')
            if not ae_cp['completed'] or model_contract(c)!=model_contract(ae_c):raise ValueError('Autoencoder must be completed with matching architecture and data recipe')
            if ae_cp.get('scientific_source',{}).get('sha256')!=scientific_source()['sha256'] and not o['allow_source_change']:raise ValueError('Representation scientific source differs; use the recorded source or an explicit recorded source-change override')
            data=Dataset(c,dependency.parent,create=False)
            dep={'path':__import__('os').path.relpath(dependency,root),'sha256':file_digest(dependency),'stage':'autoencoder','progress':ae_cp['progress']}
            if saved and dep['sha256']!=saved['dependency']['sha256']:raise ValueError('Frozen representation changed; existing flow targets cannot silently change')
        metadata=data.metadata;norm=metadata['normalization']
        if saved and (saved['dataset_identity']!=metadata['sha256'] or saved['normalization']!=norm):raise ValueError('Dataset, split or normalization identity changed')
        torch.manual_seed(c.seed);rng=torch.Generator().manual_seed(c.seed+7+(stage=='flow'))
        ae=Autoencoder(c).to(device);model=ae if stage=='autoencoder' else Velocity(c).to(device)
        latent_mean=latent_scale=None
        if ae_cp:
            ae.load_state_dict(ae_cp['autoencoder']);ae.eval().requires_grad_(False)
            latent_mean=ae_cp['latent_mean'];latent_scale=ae_cp['latent_scale']
        optimizer=torch.optim.Adam(model.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
        scaler=torch.amp.GradScaler('cuda',enabled=policy=='fp16');step=0;history=[];best=math.inf
        if saved:
            ae.load_state_dict(saved['autoencoder'])
            if stage=='flow':model.load_state_dict(saved['velocity'])
            optimizer.load_state_dict(saved['optimizer'])
            if saved['scaler']:scaler.load_state_dict(saved['scaler'])
            rng.set_state(saved['rng']);torch.set_rng_state(saved['torch_rng'])
            if device.type=='cuda' and saved.get('cuda_rng'):torch.cuda.set_rng_state_all(saved['cuda_rng'])
            step=saved['progress'];history=saved['history'];best=saved.get('best_validation',math.inf)
        started=time.perf_counter();before=saved.get('training_seconds',0.) if saved else 0.;since=0;last_publish=0.;stop=[False];previous=None
        record['setup_seconds']=time.perf_counter()-setup_started
        processed_before=saved.get('effective_examples_processed',saved['progress']*saved['config']['batch_size']) if saved else 0
        zero_seen=saved.get('zero_observation_examples_seen',int(saved['capabilities']['unconditional_trained'])) if saved else 0
        overflows=0
        if threading.current_thread() is threading.main_thread():previous=signal.signal(signal.SIGINT,lambda *_:stop.__setitem__(0,True))
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)

        def encode(y,d,context=None):
            if context is not None:cq,cv=context
            else:
                cq,cv=select_context(data.q,(y-norm['mean'])/norm['scale'],c);cq=cq[None].expand(len(y),-1,-1)
            return ae.encode(cq.to(device),cv.to(device),normalized_parameter(c,d).to(device))

        def batch_loss(y,d,qi,qw,target=None,source=None,s=None,obs=None,context=None,query_coordinates=None):
            if stage=='autoencoder':
                def forward(v,parameter):return ae.decode(encode(v,parameter,context),query_coordinates.to(device) if query_coordinates is not None else data.q[qi].to(device),normalized_parameter(c,parameter).to(device))
                if o['activation_checkpointing'] and model.training:
                    from torch.utils.checkpoint import checkpoint as recompute
                    decoded=recompute(forward,y,d,use_reentrant=False,preserve_rng_state=True)
                else:decoded=forward(y,d)
                truth=((y[:,qi]-norm['mean'])/norm['scale']).to(device)
                weights=qw.to(device)
                if weights.ndim==1:weights=weights[None].expand(len(y),-1)
                return differentiable_loss(decoded,truth,weights,field_loss.type)
            prediction=model((1-s[:,None,None])*source+s[:,None,None]*target,s,normalized_parameter(c,d).to(device),tuple(v.to(device) for v in obs))
            return (prediction.float()-(target-source).float()).square().flatten(1).mean(1).mean()

        def inputs(indices,g,split='train'):
            loading=time.perf_counter();y,d,items=data.batch(indices,split,o['workers'])
            if device.type=='cuda' and o['pin_memory']:y=y.pin_memory();d=d.pin_memory()
            context=query_coordinates=None
            if c.temporal_mode=='continuous':
                from .temporal import points
                query_coordinates=points(c,c.query_points,g,'query');qi=torch.arange(c.query_points);qw=torch.full((c.query_points,),1/c.query_points)
                context=(torch.tensor(np.stack([i['context'] for i in items])),(torch.tensor(np.stack([i['context_values'] for i in items]))-norm['mean'])/norm['scale'])
            else:qi,qw=query_subset(c,g)
            grouped=stage=='autoencoder' and field_loss.estimator=='grouped_stratified_v1'
            if grouped:
                query_coordinates,supervision,qw,stats=grouped_queries(c,items,g,norm,analysis_cache)
                qi=torch.arange(query_coordinates.shape[1]);query_stats[:]=stats
                if c.temporal_mode!='continuous':
                    from .training import select_context
                    cq,cv=select_context(data.q,(y-norm['mean'])/norm['scale'],c)
                    context=(cq[None].expand(len(y),-1,-1),cv)
                y=supervision
            target=source=s=obs=None
            if stage=='flow':
                chunks=[]
                with torch.no_grad():
                    for start in range(0,len(y),o['microbatch']):
                        with autocast():chunks.append(encode(y[start:start+o['microbatch']],d[start:start+o['microbatch']],tuple(v[start:start+o['microbatch']] for v in context) if context else None).float().cpu())
                target=(torch.cat(chunks)-latent_mean)/latent_scale
                source=torch.randn(target.shape,generator=g);s=torch.rand(len(y),generator=g)
                prior_enabled=c.unconditional_probability>0 or c.observation_counts[0]==0
                force=(0 if step==0 else max(1,c.observation_counts[0])) if split=='train' and prior_enabled and step<2 else None
                obs=_observations(c,data,y,d,g,force)
            elif c.temporal_mode=='continuous' and not grouped:
                y=torch.tensor(np.stack([i['solver'].evaluate(query_coordinates.numpy(),i['D'],c.interval) for i in items]),dtype=torch.float32)
            return (y,d,qi,qw,target,source,s,obs,context,query_coordinates),time.perf_counter()-loading

        def objective(parts,start,end):
            y,d,qi,qw,target,source,s,obs,context,query_coordinates=parts
            if qw.ndim==2:qw=qw[start:end]
            if query_coordinates is not None and query_coordinates.ndim==3:query_coordinates=query_coordinates[start:end]
            with autocast():return batch_loss(y[start:end],d[start:end],qi,qw,
                target[start:end].to(device) if target is not None else None,source[start:end].to(device) if source is not None else None,
                s[start:end].to(device) if s is not None else None,tuple(v[start:end] for v in obs) if obs is not None else None,
                tuple(v[start:end] for v in context) if context else None,query_coordinates)

        def validation(subset):
            model.eval();values=[];g=torch.Generator().manual_seed(c.seed+999)
            with torch.no_grad():
                for start in range(0,min(c.validation_count,subset),o['microbatch']):
                    end=min(start+o['microbatch'],c.validation_count,subset);parts,_=inputs(range(start,end),g,'validation')
                    values.append((float(objective(parts,0,end-start)),end-start))
            return sum(v*n for v,n in values)/sum(n for _,n in values)

        def save(completed=False,validation_value=None,final=False):
            nonlocal latent_mean,latent_scale,best
            writing_started=time.perf_counter()
            if stage=='autoencoder' and completed:latent_mean,latent_scale=_moments(ae,data,c,device,autocast)
            unconditional=zero_seen>0
            cp={'schema':SCHEMA,'engine_version':2,'architecture':'set_latent_coordinate','stage':stage,'config':c.to_dict(),'config_identity':digest(c.to_dict()),
                'completed':completed,'progress':step,'autoencoder':ae.state_dict() if stage=='autoencoder' else None,'velocity':model.state_dict() if stage=='flow' else None,
                'latent_mean':latent_mean,'latent_scale':latent_scale,'normalization':norm,'training_ids':metadata['ids'][:c.train_count],'split_ids':metadata['ids'],
                'dataset_identity':metadata['sha256'],'dependency':dep,'optimizer':optimizer.state_dict(),'scheduler':{'completed_updates':step,'settings':settings,'extension':'Keep original horizon; hold terminal LR beyond it'},
                'scaler':scaler.state_dict(),'rng':rng.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else [],'history':history,
                'field_loss':__import__('dataclasses').asdict(field_loss),'field_loss_units':'global training-normalized field; physical validation after inverse transform','latent_flow_loss':'unchanged mean squared CFM velocity error','source':code_identity(),'scientific_source':scientific_source(),'source_change_override':source_override,'environment':environment(runtime),'actual_precision':policy,
                'training_seconds':before+time.perf_counter()-started,'best_validation':min(best,validation_value) if validation_value is not None else best,
                'capabilities':{'family':'heat','dimension':c.dimension,'fields':['u'],'conditioning':'constant known diffusivity; sparse scalar observations only',
                    'query':'continuous coordinates inside trained physical contract; query flexibility is not accuracy','unconditional_trained':unconditional,'temporal':__import__('pdeflow.coordinate.temporal',fromlist=['capability']).capability(c)},
                'update_semantics':'Completed optimizer step; effective examples=batch_size; checkpoint at accumulation boundary','effective_examples_processed':processed_before+since*c.batch_size,
                'zero_observation_examples_seen':zero_seen,
                'runtime_changes':{'previous':saved['config']['runtime'],'current':c.runtime,'previous_precision':saved.get('actual_precision'),'current_precision':policy} if saved else None}
            atomic_write(checkpoint,lambda f:torch.save(cp,f));read_checkpoint(checkpoint,stage)
            periodic=root/'checkpoints'/f'update-{step:09d}.pt';atomic_write(periodic,lambda f:torch.save(cp,f))
            for p in sorted(periodic.parent.glob('update-*.pt'))[:-o['keep_periodic']]:p.unlink()
            if validation_value is not None and validation_value<best:
                best=validation_value;atomic_write(root/'best.pt',lambda f:torch.save(cp,f))
            if completed:
                portable={k:v for k,v in cp.items() if k not in ('optimizer','scheduler','scaler','rng','torch_rng','cuda_rng')}
                portable['autoencoder']=ae.state_dict();atomic_write(Path(record['export']),lambda f:torch.save(portable,f))
            record.update(progress=step,state='completed' if completed else 'interrupted',history=history[-200:],history_total=len(history),history_file=str(root/'history.json'),
                dependency=dep,dataset_identity=metadata['sha256'],normalization=norm,updated_at=now(),training_seconds=cp['training_seconds'],seconds_this_session=time.perf_counter()-started,
                parameters=sum(x.numel() for x in model.parameters()),effective_examples_processed=cp['effective_examples_processed'],checkpoint_bytes=checkpoint.stat().st_size,
                checkpoint_seconds=time.perf_counter()-writing_started,
                memory={'cuda_peak_allocated':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,'cuda_peak_reserved':torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None})
            write_json(root/'history.json',history);write_json(record_path,record)
            if event:event({'stage':stage,'kind':'checkpoint','path':str(root),'record':record,'continuing':not final and not completed})

        try:
            while step<total:
                if stop[0] or should_stop and should_stop():break
                tick=time.perf_counter();rng_before=rng.get_state();torch_before=torch.get_rng_state();cuda_before=torch.cuda.get_rng_state_all() if device.type=='cuda' else None
                indices=torch.randint(c.train_count,(c.batch_size,),generator=rng);parts,loading=inputs(indices,rng)
                model.train();optimizer.zero_grad(set_to_none=True);lr=learning_rate(settings,step)
                for group in optimizer.param_groups:group['lr']=lr
                loss_value=0.
                for start in range(0,c.batch_size,o['microbatch']):
                    end=min(c.batch_size,start+o['microbatch']);loss=objective(parts,start,end)
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss; checkpointing last completed update')
                    weight=(end-start)/c.batch_size;scaler.scale(loss*weight).backward();loss_value+=float(loss.detach())*weight
                scaler.unscale_(optimizer);gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),settings['gradient_clip'],error_if_nonfinite=False)
                if not torch.isfinite(gradient):
                    if policy=='fp16' and overflows<8:
                        scaler.step(optimizer);scaler.update();overflows+=1;rng.set_state(rng_before);torch.set_rng_state(torch_before)
                        torch.cuda.set_rng_state_all(cuda_before);continue # same batch retried; no optimizer update counted
                    raise FloatingPointError('Nonfinite gradients; last complete checkpoint retained')
                scaler.step(optimizer);scaler.update();step+=1;since+=1;synchronize(device)
                if stage=='flow':zero_seen+=int((parts[7][2].sum(1)==0).sum())
                row={'step':step,'training_loss':loss_value,'learning_rate':lr,'gradient_norm':float(gradient),'step_seconds':time.perf_counter()-tick,'data_seconds':loading,
                    'elapsed_seconds':time.perf_counter()-started,'effective_examples':processed_before+since*c.batch_size,'equivalent_dataset_passes':(processed_before+since*c.batch_size)/c.train_count,'amp_overflow_retries':overflows}
                if stage=='autoencoder':row.update(field_loss=__import__('dataclasses').asdict(field_loss),query_estimator=query_stats.copy(),loss_units='global training-normalized field')
                val=None
                if step%o['validation_every']==0 or step==total:
                    t=time.perf_counter();val=validation(o['validation_subset']);row.update(validation_loss=val,validation_seconds=time.perf_counter()-t)
                if step%o['broad_every']==0:
                    row['broad_validation_loss']=validation(o['broad_subset'])
                history.append(row)
                if time.perf_counter()-last_publish>2 or val is not None:
                    record.update(state='running',progress=step,history=history[-200:],history_total=len(history),updated_at=now(),precision=policy,
                        elapsed_seconds=time.perf_counter()-started,memory={'cuda_allocated':torch.cuda.memory_allocated(device),'cuda_reserved':torch.cuda.memory_reserved(device)} if device.type=='cuda' else None);write_json(record_path,record);last_publish=time.perf_counter()
                if step%c.checkpoint_every==0 or step%o['broad_every']==0:
                    save(False,val)
                    if step%o['broad_every']==0:
                        from .validation import representation as physical_ae,flow as physical_flow
                        target=root/'validation-preview.json';target.unlink(missing_ok=True)
                        devices=list(range(torch.cuda.device_count())) if device.type=='cuda' else [];threads=torch.get_num_threads()
                        try:
                            with torch.random.fork_rng(devices=devices):
                                if stage=='autoencoder':
                                    physical=physical_ae(checkpoint,target,str(device),subset=o['broad_subset'],allow_partial=True)
                                    row.update(broad_field_rmse=float(np.sqrt(physical['validation_mse'])),broad_initial_rmse=float(np.mean([v['initial_rmse'] for v in physical['records']])),broad_spectral_error=float(np.mean([v['spectral_power_relative_l2'] for v in physical['records']])))
                                else:
                                    physical=physical_flow(checkpoint,target,str(device),subset=min(6,o['broad_subset']),counts=sorted(set([max(1,c.observation_counts[0]),min(16,c.observation_counts[1])])),particles=2,steps=4,allow_partial=True)
                                    row['broad_ensemble_mean_rmse']=float(np.mean([v['ensemble_mean_rmse'] for v in physical['records']]))
                        finally:torch.set_num_threads(threads)
                    record['state']='running';write_json(record_path,record)
                if stop_after and since>=stop_after:break
            save(step>=total,history[-1].get('validation_loss') if history else None,final=True)
        except BaseException as error:
            # Discard partial gradients: the saved RNG/weights correspond to the last
            # completed boundary, not the partially consumed failing batch.
            if checkpoint.exists():
                record.update(state='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error),updated_at=now());write_json(record_path,record)
            else:
                record.update(state='failed',error=str(error));write_json(record_path,record)
            raise
        finally:
            if previous is not None:signal.signal(signal.SIGINT,previous)
        return record
