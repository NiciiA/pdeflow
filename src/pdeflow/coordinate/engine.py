"""Shared A1/A2/B implementation. Real references are targets, never inference context."""
from pathlib import Path
from contextlib import nullcontext
import os,time,signal,threading,math
import numpy as np
import torch
from pdeflow.runtime import RuntimeSpec,write_json,read_json,atomic_write,digest,file_digest,code_identity,environment,synchronize
from pdeflow.local import run_lock,now
from .contracts import CoordinateConfig
from .joint import ARCHITECTURE,schema,layout,options as joint_options,path as flow_path,flow_loss,active_mask
from .optimization import options,precision,learning_rate,scientific_source,model_contract,resume_contract,representation_contract
from .dataset import Dataset,state_samples,item_quantity
from .model import Autoencoder,Velocity
from .training import normalized_parameter,read_checkpoint,portable_checkpoint,SCHEMA
from .physics import family_index
from pdeflow.field_losses import grouped_queries,AnalysisCache,differentiable_loss,settings as loss_settings


def observation_batch(c,items,g,norms,device):
    from .contracts import padded_observations
    rows=[];ids=[]
    for item in items:
        n=int(torch.randint(*[c.observation_counts[0],c.observation_counts[1]+1],(),generator=g))
        if float(torch.rand((),generator=g))<c.unconditional_probability:n=0
        q=torch.rand(n,3,generator=g);q[:,:2]=q[:,:2]*torch.tensor(c.lengths)+torch.tensor(c.origins);q[:,-1]*=c.interval[1]
        from .temporal import sample_times
        q[:,-1]=sample_times(c,n,g,'observation').float()
        if n:
            from pdeflow.heat_placement import place
            placement=c.observation_placement or {'random':.5,'even':.25,'heat':.25};policies=list(placement);choice=int(torch.multinomial(torch.tensor(list(placement.values())),1,generator=g));policy=policies[choice];seed=int(torch.randint(2**31,(),generator=g));field=item['solver'].frame(.237,item['parameter'],c.interval,(32,32)) if policy=='heat' else None
            xy,_=place(n,policy,seed,c.lengths,c.origins,field=field);q[:,:2]=torch.tensor(xy,dtype=torch.float32)
        quantities=torch.randint(2,(n,),generator=g) if item['family']=='wave' else torch.zeros(n,dtype=torch.long)
        sigma=float(torch.empty(()).uniform_(*c.noise_range,generator=g));values=torch.zeros(n);ns=norms[item['family']]
        f=torch.zeros(n,3)
        for j,name in enumerate(('u','v')):
            indices=quantities.eq(j)
            if indices.any():
                clean=torch.tensor(item['solver'].evaluate(q[indices].numpy(),item['parameter'],c.interval,name),dtype=torch.float32);noisy=clean+sigma*torch.randn(len(clean),generator=g)
                f[indices,0]=(noisy-ns[j]['mean'])/ns[j]['scale'];f[indices,1]=sigma/ns[j]['scale'];f[indices,2]=1
        rows.append((q,f,quantities));ids.append(n)
    count=max(1,max(ids));points=torch.zeros(len(items),count,3,device=device);features=torch.zeros_like(points);valid=torch.zeros(len(items),count,dtype=torch.bool,device=device);quantity=torch.zeros_like(valid,dtype=torch.long)
    for b,(q,f,k) in enumerate(rows):n=len(q);points[b,:n]=q.to(device);features[b,:n]=f.to(device);valid[b,:n]=True;quantity[b,:n]=k.to(device)
    return points,features,valid,quantity


def train(config,output,stage,dependency=None,resume=False,stop_after=None,should_stop=None,event=None,extend=False):
    c=CoordinateConfig(**config).validate();o=options(c);jo=joint_options(c)
    if stage not in ('autoencoder','evolution','flow'):raise ValueError('Stages: autoencoder(A1), evolution(A2), flow(B)')
    from .physics import training_spec
    physics=training_spec(c)
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=True);device,_=RuntimeSpec(**c.runtime).resolve();policy,amp=precision(o['precision'],device)
    autocast=lambda:torch.autocast(device.type,dtype=amp) if policy!='fp32' else nullcontext()
    total=getattr(c,{'autoencoder':'ae_steps','evolution':'evolution_steps','flow':'flow_steps'}[stage]);saved=None;source_override=False;effective_examples=0;dep=None;dependency_cp=None;settings=o[stage]
    with run_lock(root):
        checkpoint=root/'resume.pt'
        if not resume and any(p.name!='.run.lock' for p in root.iterdir()):raise FileExistsError('Fresh stage requires an empty directory; choose a new run or explicit resume')
        if resume:
            saved,_=read_checkpoint(checkpoint,stage);settings,source_override=resume_contract(saved,c,stage,extend)
            if saved.get('dependency'):dependency=(root/saved['dependency']['path']).resolve()
        if stage!='autoencoder':
            if dependency is None:raise ValueError('Stage '+stage+' requires its frozen preceding stage')
            expected='autoencoder' if stage=='evolution' else 'evolution';dependency=Path(dependency).resolve();dependency_cp,dc=read_checkpoint(dependency,expected)
            if not dependency_cp['completed'] or representation_contract(c)!=representation_contract(dc):raise ValueError('Incompatible/uncompleted representation/evolution dependency')
            if dependency_cp['scientific_source']['sha256']!=scientific_source()['sha256'] and not o['allow_source_change']:raise ValueError('Frozen dependency scientific source differs')
            dep={'path':os.path.relpath(dependency,root),'sha256':file_digest(dependency),'stage':expected,'progress':dependency_cp['progress']}
            if saved and saved['dependency']!=dep:raise ValueError('Frozen stage dependency changed')
        data=Dataset(c,root if stage=='autoencoder' else (dependency.parent/dependency_cp['dataset_root']).resolve() if dependency_cp.get('dataset_root') else dependency.parent,create=stage=='autoencoder' and not resume)
        norms=data.metadata['normalizations'];torch.manual_seed(c.seed);g=torch.Generator().manual_seed(c.seed+{'autoencoder':7,'evolution':11,'flow':17}[stage]);ae=Autoencoder(c).to(device)
        if dependency_cp:ae.load_state_dict(dependency_cp['autoencoder'])
        ae.requires_grad_(False)
        if stage=='autoencoder':ae.encoder.requires_grad_(True);ae.decoder.requires_grad_(True);ae.coefficient.requires_grad_(True)
        if stage=='flow':ae.eval()
        if stage=='evolution':ae.evolution.requires_grad_(True);ae.graph_encoder.requires_grad_(True)
        model=Velocity(c).to(device) if stage=='flow' else ae
        optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=settings['learning_rate'],weight_decay=settings['weight_decay']);scaler=torch.amp.GradScaler('cuda',enabled=policy=='fp16');step=0;history=[]
        mean=dependency_cp.get('latent_mean') if dependency_cp else None;scale=dependency_cp.get('latent_scale') if dependency_cp else None
        if saved:
            ae.load_state_dict(saved['autoencoder'])
            if stage=='flow':model.load_state_dict(saved['velocity'])
            optimizer.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler']);g.set_state(saved['rng']);torch.set_rng_state(saved['torch_rng']);step=saved['progress'];history=saved['history'];effective_examples=saved.get('effective_examples_processed',step*c.batch_size);mean=saved['latent_mean'];scale=saved['latent_scale']
            if device.type=='cuda' and saved.get('cuda_rng'):torch.cuda.set_rng_state_all(saved['cuda_rng'])
        record={'kind':'coordinate_stage_run','id':root.name,'schema_version':2,'architecture':ARCHITECTURE,'stage':stage,'training_physics':physics,'specialist':'heat+wave' if len(c.family_weights)>1 else next(iter(c.family_weights)),'config':c.to_dict(),'source':code_identity(),'scientific_source':scientific_source(),'config_identity':digest(c.to_dict()),'state':'preparing','progress':step,'budget':total,'pid':os.getpid(),'created_at':now(),'export':str(root/('autoencoder.pt' if stage=='autoencoder' else 'evolution.pt' if stage=='evolution' else 'coordinate.pt')),'dependency':dep}
        write_json(root/'run.json',record);write_json(root/'config.json',c.to_dict());cache=AnalysisCache();started=time.perf_counter();stop=[False];previous=None
        if threading.current_thread() is threading.main_thread():previous=signal.signal(signal.SIGINT,lambda *_:stop.__setitem__(0,True))
        components={};reference_seconds=0.;region_seconds=0.;encoder_seconds=0.;decoder_seconds=0.
        cache_identity={'data':data.metadata['sha256'],'dependency':dep,'sampling':c.encoder_points,'normalizations':norms,'latent_mean':mean.tolist() if mean is not None else None,'latent_scale':scale.tolist() if scale is not None else None,'schema':schema(c)};cache_root=root/'latent-targets'/digest(cache_identity)
        if stage=='flow' and jo['target_cache']:cache_root.mkdir(parents=True,exist_ok=True);write_json(cache_root/'identity.json',cache_identity)

        def encode(item,t=0.,complete=True):
            nonlocal encoder_seconds,reference_seconds
            tick=time.perf_counter();f=torch.tensor([family_index(item['family'])],device=device);xy=torch.tensor(item['context'],device=device)[None];state=state_samples(item,item['context'],t,c,norms[item['family']]).to(device)[None];reference_seconds+=time.perf_counter()-tick;tick=time.perf_counter();p=normalized_parameter(c,torch.tensor([item['parameter']],device=device),f)
            z=ae.encode(xy,state,p,f) if complete else ae.encode_state(xy,state,f)
            encoder_seconds+=time.perf_counter()-tick;return z,f,p

        def fields(item,z,f,p,times,evolved=False,start_time=0.):
            nonlocal region_seconds,decoder_seconds
            losses=[];kinds=2 if item['family']=='wave' else 1
            for j in range(kinds):
                tick=time.perf_counter();view=item_quantity(item,('u','v')[j]);n=norms[item['family']][j]
                q,target,weights,stats=grouped_queries(c,[view],g,n,cache,physical_times=times);region_seconds+=time.perf_counter()-tick;q=q[0].to(device);target=(target.to(device)-n['mean'])/n['scale'];weights=weights.to(device)
                tick=time.perf_counter()
                if evolved:
                    out=[];positions=[]
                    for t in torch.unique(q[:,-1],sorted=True):
                        ix=torch.where(q[:,-1].eq(t))[0];zt=ae.evolve(z,start_time,t,p,torch.ones(1,dtype=torch.bool,device=device),f);out.append(ae.decode_state(zt,q[ix,:2],j,f));positions.append(ix)
                    pred=torch.cat(out,1)[:,torch.argsort(torch.cat(positions))]
                else:pred=ae.decode_state(z,q[:,:2],j,f)
                decoder_seconds+=time.perf_counter()-tick;losses.append(differentiable_loss(pred,target,weights,loss_settings(c.field_loss).type));components[item['family']+'_'+('u','v')[j]]=float(losses[-1].detach())
            return torch.stack(losses).mean()

        def one(item,iteration):
            family=item['family']
            from .temporal import state_time
            phase=state_time(c,iteration,g,'query')
            if stage=='autoencoder':
                t=state_time(c,iteration,g,'encoder')
                z,f,p=encode(item,t);field=fields(item,z,f,p,[t]);scalar=(ae.decode_parameter(z,families=f)-p).square().mean();components['scalar']=float(scalar.detach())
                return field+.1*scalar+jo['latent_regularization']*z.square().mean()
            if stage=='evolution':
                tb=phase;ta=float(torch.rand((),generator=g))*tb if iteration%2 else 0.;tm=(ta+tb)/2
                with torch.no_grad():za,f,p=encode(item,ta);zt,_,_=encode(item,tb);z0,_,_=encode(item,0.)
                evolved=ae.evolve(za,ta,tb,p,torch.ones(1,dtype=torch.bool,device=device),f);field=fields(item,za,f,p,[tb],True,ta)
                mask=active_mask(c,torch.ones(1,dtype=torch.bool,device=device),f)[...,None];alignment=((evolved-zt).square()*mask).sum()/mask.expand_as(zt).sum()
                via=ae.evolve(ae.evolve(z0,0.,tm,p,torch.ones(1,dtype=torch.bool,device=device),f),tm,tb,p,torch.ones(1,dtype=torch.bool,device=device),f);direct=ae.evolve(z0,0.,tb,p,torch.ones(1,dtype=torch.bool,device=device),f);composition=((via-direct).square()*mask).sum()/mask.expand_as(via).sum()
                xy=torch.tensor(item['context'][:min(32,c.encoder_points)],device=device)
                physical_composition=torch.stack([(ae.decode_state(via,xy,j,f)-ae.decode_state(direct,xy,j,f)).square().mean() for j in range(2 if family=='wave' else 1)]).mean()
                components.update(latent_alignment=float(alignment.detach()),composition=float(composition.detach()),decoded_composition=float(physical_composition.detach()))
                return field+c.evolution.get('latent_weight',.1)*alignment+c.evolution.get('composition_weight',.01)*(composition+physical_composition)
            cached=cache_root/(item['id']+'.pt')
            with torch.no_grad():
                if jo['target_cache'] and cached.exists():target=torch.load(cached,map_location='cpu',weights_only=True).to(device);f=torch.tensor([family_index(family)],device=device);p=normalized_parameter(c,torch.tensor([item['parameter']],device=device),f)
                else:
                    z,f,p=encode(item);target=(z-mean.to(device))/scale.to(device)
                    if jo['target_cache']:atomic_write(cached,lambda out:torch.save(target.detach().cpu(),out))
                graph=ae.graph_encoder(f)
            known=torch.rand(1,generator=g).to(device)<jo['known_probability']
            if step<2 and jo['warmup_disclosure_coverage'] and 0<jo['known_probability']<1:known[:]=(iteration%2==0) if step==0 else (iteration%2!=0)
            observations=observation_batch(c,[item],g,norms,device);noise=torch.randn(target.shape,generator=g).to(device);s=torch.rand(1,generator=g).to(device);state,velocity,mask=flow_path(c,target,noise,s,known,f)
            prediction=model(state,s,p,observations,known,f,graph);loss=flow_loss(c,prediction,velocity,known,f);components.update(known_parameter=int(known[0]),observations=int(observations[2].sum()))
            auxiliary=jo['auxiliary']
            if any(auxiliary[k]>0 for k in ('field','observation','pde')):
                from .consistency import auxiliary_loss
                estimate=(state+(1-s[:,None,None])*prediction)*scale.to(device)+mean.to(device)
                loss=loss+auxiliary_loss(ae,c,estimate,s,p,known,f,observations,norms[family],auxiliary,item,g)
            return loss

        def calibrate():
            values=[];families=[]
            with torch.no_grad():
                for i in range(min(c.train_count,max(12,o['calibration_count']))):
                    z,f,_=encode(data.get('train',i));values.append(z.cpu());families.append(int(f[0]))
            all_z=torch.cat(values);m=torch.zeros(1,c.tokens,c.width);v=torch.ones_like(m)
            for group in layout(c):
                valid=torch.tensor([f==1 for f in families]) if group.name=='initial_v' else torch.ones(len(values),dtype=torch.bool)
                if valid.any():a=all_z[valid,group.start:group.stop].double();mu=a.mean((0,1));sigma=a.var((0,1),unbiased=False).sqrt().clamp_min(.01);m[:,group.start:group.stop]=mu.float();v[:,group.start:group.stop]=sigma.float()
            return m,v

        def save(completed=False):
            nonlocal mean,scale
            tick=time.perf_counter()
            if stage=='autoencoder' and completed:mean,scale=calibrate()
            cp={'schema':SCHEMA,'architecture':ARCHITECTURE,'engine_version':3,'joint_schema':schema(c),'stage':stage,'config':c.to_dict(),'config_identity':digest(c.to_dict()),'progress':step,'completed':completed,'autoencoder':ae.state_dict() if stage!='flow' else None,'velocity':model.state_dict() if stage=='flow' else None,'dependency':dep,'dataset_root':os.path.relpath(data.root,root),'dataset_identity':data.metadata['sha256'],'training_ids':[x for x in data.metadata['ids'] if x.startswith('train:')],'normalization':data.metadata['normalization'],'normalizations':norms,'normalization_fitted_ids':data.metadata['fitted_ids'],'latent_mean':mean,'latent_scale':scale,'optimizer':optimizer.state_dict(),'scheduler':{'settings':settings,'completed_updates':step},'scaler':scaler.state_dict(),'rng':g.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else [],'source':code_identity(),'scientific_source':scientific_source(),'environment':environment(RuntimeSpec(**c.runtime)),'history':history,'actual_precision':policy,'source_change_override':source_override,'effective_examples_processed':effective_examples,'field_loss':c.field_loss,'conditioning_policy':jo,'target_cache_identity':cache_identity,'training_seconds':time.perf_counter()-started,'training_physics':physics,'capabilities':{'state_schemas':physics['state_schemas'],'families':list(c.family_weights),'family_weights':c.family_weights,'dimension':2,'fields':['u','v (wave only)'],'state':'initial displacement + initial velocity(wave) + typed scalar; no trajectory latent','parameter_ranges':{'diffusivity':c.diffusivity_range,'wave_speed':c.wave_speed_range},'temporal':{'mode':'continuous','interval':list(c.interval)},'unconditional_trained':c.unconditional_probability>0 or c.observation_counts[0]==0,'scientific_accuracy':False}}
            atomic_write(checkpoint,lambda out:torch.save(cp,out))
            if completed:
                portable=portable_checkpoint({**cp,'autoencoder':ae.state_dict()}) if stage=='flow' else {k:v for k,v in cp.items() if k not in ('optimizer','scheduler','scaler','rng','torch_rng','cuda_rng')}
                if stage=='flow':portable['dataset_root']=None
                atomic_write(Path(record['export']),lambda out:torch.save(portable,out))
            record.update(state='completed' if completed else 'interrupted',progress=step,history=history[-200:],history_total=len(history),normalization=data.metadata['normalization'],normalizations=norms,training_seconds=time.perf_counter()-started,checkpoint_seconds=time.perf_counter()-tick,parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),dataset_identity=data.metadata['sha256'],updated_at=now(),capabilities=cp['capabilities'])
            record['checkpoint_bytes']=sum(f.stat().st_size for f in root.glob('*.pt'));record['memory']={'cuda_peak_allocated':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,'cuda_peak_reserved':torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None,'mps_current_allocated':torch.mps.current_allocated_memory() if device.type=='mps' else None};write_json(root/'history.json',history);write_json(root/'run.json',record)
            if event:event({'stage':stage,'kind':'checkpoint','path':str(root),'record':record,'continuing':not completed})

        try:
            since=0
            while step<total:
                if stop[0] or should_stop and should_stop():break
                tick=time.perf_counter();optimizer.zero_grad(set_to_none=True);lr=learning_rate(settings,step)
                for group in optimizer.param_groups:group['lr']=lr
                names=sorted(c.family_weights);weights=torch.tensor([c.family_weights[n] for n in names]);weights/=weights.sum();offset=float(torch.rand((),generator=g));chosen=torch.searchsorted(weights.cumsum(0),(torch.arange(c.batch_size)/c.batch_size+offset)%1).tolist();items=[]
                # Bounded rejection over deterministic identities; no cross-split variants.
                for chosen_family in chosen:
                    for trial in range(1000):
                        index=int(torch.randint(c.train_count,(),generator=g));item=data.get('train',index)
                        if item['family']==names[chosen_family]:items.append(item);break
                    else:raise ValueError('Unable to sample configured family from training split')
                components={};batch_loss=0.;per_family={};backward=0.
                model.train();ae.eval() if stage=='flow' else None;ae.encoder.eval() if stage!='autoencoder' else ae.encoder.train()
                for b,item in enumerate(items):
                    with autocast():loss=one(item,step*c.batch_size+b)
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite stage loss')
                    loss_value=float(loss.detach());batch_loss+=loss_value/c.batch_size;per_family.setdefault(item['family'],[]).append(loss_value)
                    tb=time.perf_counter();scaler.scale(loss/c.batch_size).backward();backward+=time.perf_counter()-tb
                scaler.unscale_(optimizer);grad=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],settings['gradient_clip'],error_if_nonfinite=True);scaler.step(optimizer);scaler.update();step+=1;since+=1;effective_examples+=c.batch_size;synchronize(device)
                row={'step':step,'training_loss':batch_loss,'family_example_counts':{k:sum(x['family']==k for x in items) for k in names},'per_family_loss':{k:float(np.mean(v)) for k,v in per_family.items()},'components':components.copy(),'gradient_norm':float(grad),'learning_rate':lr,'step_seconds':time.perf_counter()-tick,'backward_seconds':backward,'reference_seconds_cumulative':reference_seconds,'data_seconds':reference_seconds,'component_timings':{'reference_cumulative_host':reference_seconds,'encoder_cumulative_host':encoder_seconds,'region_reference_cumulative_host':region_seconds,'decoder_cumulative_host':decoder_seconds,'backward_step_host':backward},'timing_semantics':'Host inclusive cumulative component clocks; asynchronous GPU component timings are not kernel timings. Step wall time synchronized.','region_seconds_cumulative':region_seconds,'encoder_seconds_cumulative':encoder_seconds,'decoder_seconds_cumulative':decoder_seconds,'elapsed_seconds':time.perf_counter()-started,'effective_examples':effective_examples,'microbatch_semantics':'one realization at a time with exact batch accumulation; configured upper microbatch bound respected'}
                if step%o['validation_every']==0 or step==total:
                    ae.eval();model.eval();state=g.get_state();val={};vt=time.perf_counter()
                    with torch.enable_grad() if jo['auxiliary']['pde']>0 else torch.no_grad():
                        for i in range(min(c.validation_count,max(2,len(names)))):
                            item=data.get('validation',i);value=float(one(item,step*c.batch_size+i));val.setdefault(item['family'],[]).append(value)
                    g.set_state(state);row.update(validation_loss=float(np.mean([np.mean(v) for v in val.values()])),validation_per_family={k:float(np.mean(v)) for k,v in val.items()},validation_seconds=time.perf_counter()-vt)
                history.append(row);record.update(state='running',progress=step,history=history[-200:],history_total=len(history),elapsed_seconds=time.perf_counter()-started);write_json(root/'run.json',record)
                if step%c.checkpoint_every==0:save(False)
                if stop_after and since>=stop_after:break
            save(step>=total)
        except BaseException as error:
            record.update(state='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error));write_json(root/'run.json',record);raise
        finally:
            if previous is not None:signal.signal(signal.SIGINT,previous)
        return record
