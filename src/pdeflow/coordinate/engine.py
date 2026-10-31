"""Shared A1/A2/B implementation. Real references are targets, never inference context."""
from pathlib import Path
from contextlib import nullcontext
from collections import OrderedDict
from .profiling import Timings,host_peak_bytes
from .frozen import FrozenTargets
from . import batching
import os,time,signal,threading,math
import numpy as np
import torch
from pdeflow.runtime import RuntimeSpec,write_json,read_json,atomic_write,digest,file_digest,code_identity,environment,synchronize
from pdeflow.local import run_lock,now
from .contracts import CoordinateConfig
from .joint import ARCHITECTURE,schema,layout,options as joint_options,path as flow_path,flow_loss,active_mask
from .optimization import options,precision,learning_rate,scientific_source,model_contract,resume_contract,representation_contract
from .dataset import Dataset,state_samples,state_samples_batch,item_quantity,family_for
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
    count=max(1,max(ids));points=torch.zeros(len(items),count,3);features=torch.zeros_like(points);valid=torch.zeros(len(items),count,dtype=torch.bool);quantity=torch.zeros_like(valid,dtype=torch.long)
    for b,(q,f,k) in enumerate(rows):n=len(q);points[b,:n]=q;features[b,:n]=f;valid[b,:n]=True;quantity[b,:n]=k
    profiler=items[0]['solver'].profiler if items else None
    with profiler.measure('transfer',host=True) if profiler else nullcontext():
        values=(points,features,valid,quantity)
        if device.type=='cuda':values=tuple(x.pin_memory() for x in values)
        return tuple(x.to(device,non_blocking=device.type=='cuda') for x in values)


def train(config,output,stage,dependency=None,resume=False,stop_after=None,should_stop=None,event=None,extend=False):
    setup_started=time.perf_counter()
    c=CoordinateConfig(**config).validate();o=options(c,stage);jo=joint_options(c)
    if stage not in ('autoencoder','evolution','flow'):raise ValueError('Stages: autoencoder(A1), evolution(A2), flow(B)')
    from .physics import training_spec
    physics=training_spec(c);resolved_config=c.to_dict();config_identity=digest(resolved_config);source_identity=scientific_source();joint_schema=schema(c)
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
            if dependency_cp['scientific_source']['sha256']!=source_identity['sha256'] and not o['allow_source_change']:raise ValueError('Frozen dependency scientific source differs')
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
        optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=settings['learning_rate'],weight_decay=settings['weight_decay']);scaler=torch.amp.GradScaler('cuda',enabled=policy=='fp16',init_scale=o['initial_scale']);step=0;attempted=0;skipped_total=0;overflow_streak=0;history=[]
        mean=dependency_cp.get('latent_mean') if dependency_cp else None;scale=dependency_cp.get('latent_scale') if dependency_cp else None
        if saved:
            ae.load_state_dict(saved['autoencoder'])
            if stage=='flow':model.load_state_dict(saved['velocity'])
            optimizer.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler']);g.set_state(saved['rng']);torch.set_rng_state(saved['torch_rng']);step=saved['progress'];history=saved['history'];attempted=saved.get('attempted_updates',step);skipped_total=saved.get('skipped_updates',sum(r.get('optimizer_update_skipped',False) for r in history));overflow_streak=saved.get('overflow_streak',0);
            if skipped_total and saved.get('counter_semantics')!='successful_updates_v1':raise ValueError('Legacy skipped-step counters require an explicit migration; use recorded source, never silently treat skips as successes')
            effective_examples=saved.get('effective_examples_processed',step*c.batch_size);mean=saved['latent_mean'];scale=saved['latent_scale']
            if device.type=='cuda' and saved.get('cuda_rng'):torch.cuda.set_rng_state_all(saved['cuda_rng'])
        record={'kind':'coordinate_stage_run','id':root.name,'schema_version':2,'architecture':ARCHITECTURE,'stage':stage,'precision':policy,'batch_settings':__import__('pdeflow.training_settings',fromlist=['resolve_batches']).resolve_batches(c)[stage],'training_physics':physics,'specialist':'heat+wave' if len(c.family_weights)>1 else next(iter(c.family_weights)),'config':resolved_config,'source':code_identity(),'scientific_source':source_identity,'config_identity':config_identity,'state':'preparing','progress':step,'budget':total,'pid':os.getpid(),'created_at':now(),'export':str(root/('autoencoder.pt' if stage=='autoencoder' else 'evolution.pt' if stage=='evolution' else 'coordinate.pt')),'dependency':dep}
        write_json(root/'run.json',record);write_json(root/'config.json',resolved_config);cache=AnalysisCache();started=time.perf_counter();stop=[False];previous=None
        if threading.current_thread() is threading.main_thread():previous=signal.signal(signal.SIGINT,lambda *_:stop.__setitem__(0,True))
        components={};timings=Timings(device,o['profile_components']);cache.profiler=timings;frozen=OrderedDict();cache_hits=0;cache_misses=0;frozen_graphs={}
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        cache_identity={'data':data.metadata['sha256'],'dependency':dep,'sampling':c.encoder_points,'normalizations':norms,'latent_mean':mean.tolist() if mean is not None else None,'latent_scale':scale.tolist() if scale is not None else None,'schema':joint_schema,'scientific_source':source_identity['sha256'],'precision':policy,'sampling_identity':{'seed':c.seed,'context':'fixed deterministic example.context; physical t=0 for disk targets'},'version':3};cache_hash_started=time.perf_counter();cache_prefix=digest(cache_identity);record['cache_identity_hash_seconds_setup']=time.perf_counter()-cache_hash_started;cache_root=root/'latent-targets'/cache_prefix
        if stage!='autoencoder' and jo['target_cache']:cache_root.mkdir(parents=True,exist_ok=True);write_json(cache_root/'identity.json',cache_identity)

        targets=FrozenTargets(cache_prefix,o['frozen_cache_items'],timings,cache_root if stage!='autoencoder' and jo['target_cache'] else None,o['frozen_prefetch'])
        latent_mean_device=mean.to(device) if mean is not None else None;latent_scale_device=scale.to(device) if scale is not None else None

        def attach(item):
            item['solver'].profiler=timings
            return item

        def encode(item,t=0.,complete=True):
            nonlocal cache_hits,cache_misses
            attach(item)
            # Frozen only; include exact continuous time, representation/data/
            # normalization/coordinate identity, precision/autocast and mode.
            key=(cache_prefix,item['id'],float(t),complete,torch.is_autocast_enabled(device.type))
            use_cache=stage!='autoencoder' and o['performance_mode']=='optimized'
            if use_cache and key in frozen:
                cache_hits+=1;frozen.move_to_end(key)
                with timings.measure('transfer',host=True):return tuple(x.to(device,non_blocking=device.type=='cuda' and o['pin_memory']) for x in frozen[key])
            cache_misses+=1
            with timings.measure('encoder_reference',host=True):state=state_samples(item,item['context'],t,c,norms[item['family']])
            with timings.measure('transfer',host=True):
                f=torch.tensor([family_index(item['family'])],device=device);xy=torch.as_tensor(item['context'],device=device)[None]
                if device.type=='cuda' and o['pin_memory']:state=state.pin_memory()
                state=state.to(device,non_blocking=device.type=='cuda' and o['pin_memory'])[None];p=normalized_parameter(c,torch.tensor([item['parameter']],device=device),f)
            with timings.measure('encoder_compute'):
                z=ae.encode(xy,state,p,f) if complete else ae.encode_state(xy,state,f)
            if use_cache:
                with timings.measure('frozen_cache_store',host=True):frozen[key]=tuple(x.detach().cpu() for x in (z,f,p))
                if len(frozen)>o['frozen_cache_items']:frozen.popitem(last=False)
            return z,f,p

        def fields(item,z,f,p,times,evolved=False,start_time=0.):
            losses=[];kinds=2 if item['family']=='wave' else 1
            # One supervised physical time per solution in this state trainer.
            # Evolve once, shared by u/v decoders; never once per quantity/query.
            with timings.measure('evolution_compute'):
                state=ae.evolve(z,start_time,float(np.float32(times[0])),p,torch.ones(1,dtype=torch.bool,device=device),f) if evolved else z
            for j in range(kinds):
                view=item_quantity(item,('u','v')[j]);n=norms[item['family']][j]
                with timings.measure('region_supervision',host=True):q,target,weights,stats=grouped_queries(c,[view],g,n,cache,physical_times=times)
                if not torch.isfinite(weights).all() or (weights<0).any():raise ValueError('Invalid field supervision weights')
                with timings.measure('transfer',host=True):
                    q=q[0].to(device,non_blocking=device.type=='cuda');target=(target.to(device,non_blocking=device.type=='cuda')-n['mean'])/n['scale'];weights=weights.to(device,non_blocking=device.type=='cuda')
                with timings.measure('decoder_compute'):
                    pred=ae.decode_state(state,q[:,:2],j,f)
                    losses.append(differentiable_loss(pred,target,weights,loss_settings(c.field_loss).type,validate=False))
                components[item['family']+'_'+('u','v')[j]]=losses[-1].detach()
            return torch.stack(losses).mean()

        def one(item,iteration,defer=False):
            family=item['family']
            from .temporal import state_time
            phase=state_time(c,iteration,g,'query')
            if stage=='autoencoder':
                t=state_time(c,iteration,g,'encoder')
                z,f,p=encode(item,t);field=fields(item,z,f,p,[t]);scalar=(ae.decode_parameter(z,families=f)-p).square().mean();components['scalar']=scalar.detach()
                return field+.1*scalar+jo['latent_regularization']*z.square().mean()
            if stage=='evolution':
                tb=phase;ta=float(torch.rand((),generator=g))*tb if iteration%2 else 0.;tm=(ta+tb)/2
                with torch.no_grad():za,f,p=encode(item,ta);zt,_,_=encode(item,tb);z0,_,_=encode(item,0.)
                with timings.measure('evolution_compute'):evolved=ae.evolve(za,ta,tb,p,torch.ones(1,dtype=torch.bool,device=device),f)
                field=fields(item,za,f,p,[tb],True,ta)
                mask=active_mask(c,torch.ones(1,dtype=torch.bool,device=device),f)[...,None];alignment=((evolved-zt).square()*mask).sum()/mask.expand_as(zt).sum()
                with timings.measure('composition_compute'):via=ae.evolve(ae.evolve(z0,0.,tm,p,torch.ones(1,dtype=torch.bool,device=device),f),tm,tb,p,torch.ones(1,dtype=torch.bool,device=device),f);direct=ae.evolve(z0,0.,tb,p,torch.ones(1,dtype=torch.bool,device=device),f);composition=((via-direct).square()*mask).sum()/mask.expand_as(via).sum()
                xy=torch.tensor(item['context'][:min(32,c.encoder_points)],device=device)
                with timings.measure('composition_decode_compute'):physical_composition=torch.stack([(ae.decode_state(via,xy,j,f)-ae.decode_state(direct,xy,j,f)).square().mean() for j in range(2 if family=='wave' else 1)]).mean()
                components.update(latent_alignment=alignment.detach(),composition=composition.detach(),decoded_composition=physical_composition.detach())
                return field+c.evolution.get('latent_weight',.1)*alignment+c.evolution.get('composition_weight',.01)*(composition+physical_composition)
            cached=cache_root/(item['id']+('-amp' if torch.is_autocast_enabled(device.type) else '-fp32')+'.pt')
            with torch.no_grad(),timings.measure('frozen_target',host=True):
                if item['id'] in prepared_zero:
                    target=(prepared_zero[item['id']]-latent_mean_device)/latent_scale_device;f=torch.tensor([family_index(family)],device=device);p=normalized_parameter(c,torch.tensor([item['parameter']],device=device),f)
                elif jo['target_cache'] and cached.exists():
                    with timings.measure('target_cache_io',host=True):target=torch.load(cached,map_location='cpu',weights_only=True).to(device)
                    f=torch.tensor([family_index(family)],device=device);p=normalized_parameter(c,torch.tensor([item['parameter']],device=device),f)
                else:
                    z,f,p=encode(item);target=(z-latent_mean_device)/latent_scale_device
                    if jo['target_cache'] and (cached.exists() or len(targets.disk)<o['frozen_cache_items']):
                        with timings.measure('target_cache_io',host=True):atomic_write(cached,lambda out:torch.save(target.detach().cpu(),out));targets.disk.add(cached.stem)
                graph_key=(family,torch.is_autocast_enabled(device.type))
                if graph_key not in frozen_graphs or o['performance_mode']=='legacy':
                    with timings.measure('graph_compute'):frozen_graphs[graph_key]=ae.graph_encoder(f).detach()
                graph=frozen_graphs[graph_key]
            known=torch.rand(1,generator=g).to(device)<jo['known_probability']
            if step<2 and jo['warmup_disclosure_coverage'] and 0<jo['known_probability']<1:known[:]=(iteration%2==0) if step==0 else (iteration%2!=0)
            with timings.measure('observation_preparation',host=True):observations=observation_batch(c,[item],g,norms,device)
            noise=torch.randn(target.shape,generator=g).to(device);s=torch.rand(1,generator=g).to(device);state,velocity,mask=flow_path(c,target,noise,s,known,f)
            components.update(known_parameter=known[0].detach(),observations=observations[2].sum().detach(),known_parameter_examples=components.get('known_parameter_examples',0)+known.sum().detach(),observation_values=components.get('observation_values',0)+observations[2].sum().detach())
            if defer:return (state,s,p,observations,known,f,graph,velocity)
            with timings.measure('flow_compute'):prediction=model(state,s,p,observations,known,f,graph);loss=flow_loss(c,prediction,velocity,known,f)
            components.update(known_parameter=known[0].detach(),observations=observations[2].sum().detach())
            auxiliary=jo['auxiliary']
            if any(auxiliary[k]>0 for k in ('field','observation','pde')):
                from .consistency import auxiliary_loss
                estimate=(state+(1-s[:,None,None])*prediction)*latent_scale_device+latent_mean_device
                loss=loss+auxiliary_loss(ae,c,estimate,s,p,known,f,observations,norms[family],auxiliary,item,g)
            return loss

        prepared_zero={}
        def flow_microbatch(items,iteration):
            with torch.no_grad(),timings.measure('frozen_target',host=True):
                def prepare(requests):return state_samples_batch([x for x,_ in requests],[t for _,t in requests],c,norms)
                def compute(requests,state):
                    with timings.measure('transfer',host=True):
                        si=[x for x,_ in requests];sf=torch.tensor([family_index(x['family']) for x in si],device=device);sp=normalized_parameter(c,torch.tensor([x['parameter'] for x in si],device=device),sf);sx=torch.as_tensor(np.stack([x['context'] for x in si]),device=device)
                        if device.type=='cuda':state=state.pin_memory()
                        state=state.to(device,non_blocking=device.type=='cuda')
                    with timings.measure('encoder_compute'):return ae.encode(sx,state,sp,sf)
                values=targets.resolve([(item,0.) for item in items],torch.is_autocast_enabled(device.type),o['microbatch'],prepare,compute,device)
                with timings.measure('transfer',host=True):values=values.to(device,non_blocking=device.type=='cuda')
                prepared_zero.update({item['id']:values[i:i+1] for i,item in enumerate(items)})
            prepared=[one(item,iteration+i,True) for i,item in enumerate(items)]
            prepared_zero.clear()
            count=max(x[3][0].shape[1] for x in prepared);observations=[]
            for j in range(4):
                rows=[]
                for x in prepared:
                    value=x[3][j];padding=value.new_zeros((1,count-value.shape[1],*value.shape[2:]));rows.append(torch.cat([value,padding],1))
                observations.append(torch.cat(rows))
            state,s,p,known,f,graph,velocity=[torch.cat([x[j] for x in prepared]) for j in (0,1,2,4,5,6,7)]
            with timings.measure('flow_compute'):
                prediction=model(state,s,p,tuple(observations),known,f,graph)
                return [flow_loss(c,prediction[i:i+1],velocity[i:i+1],known[i:i+1],f[i:i+1]) for i in range(len(items))]

        def calibrate():
            values=[];families=[]
            with torch.no_grad():
                count=min(c.train_count,max(12,o['calibration_count']));size=o['microbatch'] if o['performance_mode']=='optimized' else 1
                for start in range(0,count,size):
                    items=data.items(range(start,min(count,start+size)),workers=o['workers'])
                    with timings.measure('reference_evaluation',host=True):state=state_samples_batch(items,[0.]*len(items),c,norms)
                    with timings.measure('transfer',host=True):
                        f=torch.tensor([family_index(x['family']) for x in items],device=device);xy=torch.as_tensor(np.stack([x['context'] for x in items]),device=device);state=state.to(device);p=normalized_parameter(c,torch.tensor([x['parameter'] for x in items],device=device),f)
                    with timings.measure('encoder_compute'):z=ae.encode(xy,state,p,f)
                    values.append(z.cpu());families.extend(f.cpu().tolist())
            all_z=torch.cat(values);m=torch.zeros(1,c.tokens,c.width);v=torch.ones_like(m)
            for group in layout(c):
                valid=torch.tensor([f==1 for f in families]) if group.name=='initial_v' else torch.ones(len(families),dtype=torch.bool)
                if valid.any():a=all_z[valid,group.start:group.stop].double();mu=a.mean((0,1));sigma=a.var((0,1),unbiased=False).sqrt().clamp_min(.01);m[:,group.start:group.stop]=mu.float();v[:,group.start:group.stop]=sigma.float()
            return m,v

        def save(completed=False):
            nonlocal mean,scale
            tick=time.perf_counter()
            if stage=='autoencoder' and completed:
                calibration_started=time.perf_counter();mean,scale=calibrate();record['latent_calibration_seconds']=time.perf_counter()-calibration_started
            cp={'schema':SCHEMA,'architecture':ARCHITECTURE,'engine_version':3,'joint_schema':joint_schema,'stage':stage,'config':resolved_config,'config_identity':config_identity,'progress':step,'attempted_updates':attempted,'successful_optimizer_updates':step,'skipped_updates':skipped_total,'overflow_streak':overflow_streak,'counter_semantics':'successful_updates_v1','initial_config':saved.get('initial_config',saved['config']) if saved else resolved_config,'continuation_history':(saved.get('continuation_history',[]) if saved else [])+([{'from_config':saved['config'],'to_config':resolved_config,'reason':'explicit batch/budget continuation','timestamp':now()}] if saved and saved['config']!=resolved_config else []),'batch_settings':__import__('pdeflow.training_settings',fromlist=['resolve_batches']).resolve_batches(c)[stage],'completed':completed,'autoencoder':ae.state_dict() if stage!='flow' else None,'velocity':model.state_dict() if stage=='flow' else None,'dependency':dep,'dataset_root':os.path.relpath(data.root,root),'dataset_identity':data.metadata['sha256'],'training_ids':[x for x in data.metadata['ids'] if x.startswith('train:')],'normalization':data.metadata['normalization'],'normalizations':norms,'normalization_fitted_ids':data.metadata['fitted_ids'],'latent_mean':mean,'latent_scale':scale,'optimizer':optimizer.state_dict(),'scheduler':{'settings':settings,'completed_updates':step,'semantics':'successful optimizer updates only'},'scaler':scaler.state_dict(),'rng':g.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else [],'source':code_identity(),'scientific_source':source_identity,'environment':environment(RuntimeSpec(**c.runtime)),'history':history,'actual_precision':policy,'source_change_override':source_override,'effective_examples_processed':effective_examples,'field_loss':c.field_loss,'conditioning_policy':jo,'target_cache_identity':cache_identity,'training_seconds':time.perf_counter()-started,'training_physics':physics,'capabilities':{'state_schemas':physics['state_schemas'],'families':list(c.family_weights),'family_weights':c.family_weights,'dimension':2,'fields':['u','v (wave only)'],'state':'initial displacement + initial velocity(wave) + typed scalar; no trajectory latent','parameter_ranges':{'diffusivity':c.diffusivity_range,'wave_speed':c.wave_speed_range},'temporal':{'mode':'continuous','interval':list(c.interval)},'unconditional_trained':c.unconditional_probability>0 or c.observation_counts[0]==0,'scientific_accuracy':False}}
            atomic_write(checkpoint,lambda out:torch.save(cp,out))
            if completed:
                portable=portable_checkpoint({**cp,'autoencoder':ae.state_dict()}) if stage=='flow' else {k:v for k,v in cp.items() if k not in ('optimizer','scheduler','scaler','rng','torch_rng','cuda_rng')}
                if stage=='flow':portable['dataset_root']=None
                atomic_write(Path(record['export']),lambda out:torch.save(portable,out))
            record.update(state='completed' if completed else 'interrupted',progress=step,attempted_updates=attempted,successful_optimizer_updates=step,skipped_updates=skipped_total,history=history[-200:],history_total=len(history),normalization=data.metadata['normalization'],normalizations=norms,training_seconds=time.perf_counter()-started,checkpoint_seconds=time.perf_counter()-tick,parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),dataset_identity=data.metadata['sha256'],updated_at=now(),capabilities=cp['capabilities'])
            record['checkpoint_bytes']=sum(f.stat().st_size for f in root.glob('*.pt'));record['memory']={'process_peak_host_rss_bytes':host_peak_bytes(),'host_peak_scope':'Process lifetime RSS high-water mark, not stage-local allocation','cuda_peak_allocated':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,'cuda_peak_reserved':torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None,'mps_current_allocated':torch.mps.current_allocated_memory() if device.type=='mps' else None};write_json(root/'history.json',history);write_json(root/'run.json',record)
            if event:event({'stage':stage,'kind':'checkpoint','path':str(root),'record':record,'continuing':not completed})

        record['setup_seconds']=time.perf_counter()-setup_started
        def scalarize(value):
            if isinstance(value,dict):return {k:scalarize(v) for k,v in value.items()}
            if torch.is_tensor(value):return value.detach().float().item()
            return value
        try:
            since=0
            while step<total:
                if stop[0] or should_stop and should_stop():break
                synchronize(device);timings.reset();prior_frozen=(cache_hits,cache_misses);prior_region=(cache.hits,cache.misses);prior_preparation=targets.stats.copy();prior_targets=(targets.stats['memory_hits']+targets.stats['disk_hits']+targets.stats['deduplicated_requests'],targets.stats['misses']);tick=time.perf_counter();optimizer.zero_grad(set_to_none=True);lr=learning_rate(settings,step)
                for group in optimizer.param_groups:group['lr']=lr
                names=sorted(c.family_weights);weights=torch.tensor([c.family_weights[n] for n in names]);weights/=weights.sum();offset=float(torch.rand((),generator=g));chosen=torch.searchsorted(weights.cumsum(0),(torch.arange(c.batch_size)/c.batch_size+offset)%1).tolist();items=[]
                # Reject by deterministic family identity, not by constructing
                # expensive complete wrong-family references. RNG draws unchanged.
                indices=[]
                for chosen_family in chosen:
                    for trial in range(1000):
                        index=int(torch.randint(c.train_count,(),generator=g))
                        if family_for(c,'train',index)==names[chosen_family]:indices.append(index);break
                    else:raise ValueError('Unable to sample configured family from training split')
                with timings.measure('dataset_initial_reference',host=True):items=[attach(x) for x in data.items(indices,workers=o['workers'])]
                components={};batch_loss=0.;per_family={};backward=0.
                model.train();ae.eval() if stage=='flow' else None;ae.encoder.eval() if stage!='autoencoder' else ae.encoder.train()
                use_batch=o['execution']=='batched' and o['performance_mode']=='optimized' and (stage!='flow' or not any(jo['auxiliary'][k]>0 for k in ('field','observation','pde')))
                size=o['microbatch'] if use_batch else 1;forward_sizes=[];quantity_sizes=[]
                for start in range(0,len(items),size):
                    group=items[start:start+size];forward_sizes.append(len(group))
                    if stage!='flow':quantity_sizes.append({'u':len(group),'v':sum(item['family']=='wave' for item in group)})
                    with autocast():
                        if use_batch and stage!='flow':
                            planned=batching.plans(c,group,g,effective_examples+start,stage,norms,cache,timings,item_quantity,grouped_queries)
                            losses=batching.forward(c,ae,planned,stage,norms,device,timings,targets,differentiable_loss,item_quantity,components)
                        elif use_batch:losses=flow_microbatch(group,effective_examples+start)
                        else:losses=[one(group[0],effective_examples+start)]
                    for item,loss in zip(group,losses):
                        loss_value=loss.detach().float();batch_loss+=loss_value/c.batch_size;per_family.setdefault(item['family'],[]).append(loss_value)
                    with timings.measure('backward_compute'):scaler.scale(torch.stack(list(losses)).sum()/c.batch_size).backward()
                batch_loss=float(batch_loss)
                if not math.isfinite(batch_loss):raise FloatingPointError('Nonfinite stage loss; optimizer not stepped')
                scale_before=scaler.get_scale()
                with timings.measure('optimizer_compute'):
                    from .optimization import optimizer_step
                    grad,finite,skipped=optimizer_step(optimizer,scaler,[p for p in model.parameters() if p.requires_grad],settings['gradient_clip'],policy)
                attempted+=1;since+=1;effective_examples+=c.batch_size
                if skipped:skipped_total+=1;overflow_streak+=1
                else:step+=1;overflow_streak=0
                synchronize(device);elapsed=time.perf_counter()-tick
                measured=timings.report()
                frozen_report=targets.report()
                cache_report=({'hits_cumulative':frozen_report['memory_hits']+frozen_report['disk_hits']+frozen_report['deduplicated_requests'],'misses_cumulative':frozen_report['misses'],'hits_this_step':frozen_report['memory_hits']+frozen_report['disk_hits']+frozen_report['deduplicated_requests']-prior_targets[0],'misses_this_step':frozen_report['misses']-prior_targets[1],'items':frozen_report['items'],'bytes':frozen_report['bytes']} if use_batch else {'hits_cumulative':cache_hits,'misses_cumulative':cache_misses,'hits_this_step':cache_hits-prior_frozen[0],'misses_this_step':cache_misses-prior_frozen[1],'items':len(frozen),'bytes':sum(x.numel()*x.element_size() for values in frozen.values() for x in values)})
                row={'step':step,'attempted_update':attempted,'successful_optimizer_updates':step,'skipped_updates_total':skipped_total,'consecutive_overflows':overflow_streak,'configured_batch':__import__('pdeflow.training_settings',fromlist=['resolve_batches']).resolve_batches(c)[stage],'training_loss':batch_loss,'family_example_counts':{k:sum(x['family']==k for x in items) for k in names},'per_family_loss':{k:float(torch.stack(v).mean()) for k,v in per_family.items()},'components':scalarize(components),'gradient_norm':float(grad) if finite else None,'gradients_finite':finite,'optimizer_update_skipped':skipped,'scaler_scale':scaler.get_scale(),'learning_rate':lr,'step_seconds':elapsed,'data_seconds':sum((measured or {}).get('host_exclusive_seconds',{}).get(k,0.) for k in ('dataset_initial_reference','reference_evaluation','region_analysis')),'component_timings':measured,'data_seconds_definition':'Sum of exclusive host reference evaluation, region analysis and dataset construction; see nested component scopes for packing/other preprocessing. Prefetched CPU reference work is separately recorded in frozen_preparation_step; it can overlap compute and is not added here.', 'timing_semantics':'Per-update synchronized wall time, excluding validation/persistence. Nested component scopes explicitly labelled. No cumulative counter is a step cost.','elapsed_seconds':time.perf_counter()-started,'effective_examples':effective_examples,'frozen_cache':cache_report,'region_cache':{'hits_cumulative':cache.hits,'misses_cumulative':cache.misses,'hits_this_step':cache.hits-prior_region[0],'misses_this_step':cache.misses-prior_region[1]},'frozen_preparation':frozen_report,'frozen_preparation_step':{k:v-prior_preparation.get(k,0) for k,v in targets.stats.items() if k!='maximum_encoding_batch'},'microbatch_semantics':'A1/A2/default B use actual mixed-family neural microbatches; losses sum per example / effective batch. Explicit accumulated control and optional B auxiliaries remain serial.','actual_forward_microbatch':max(forward_sizes),'forward_microbatch_sizes':forward_sizes,'quantity_forward_microbatch_sizes':quantity_sizes}
                if skipped:
                    from .optimization import gradient_diagnostics
                    row['overflow_diagnostics']={**gradient_diagnostics(model.named_parameters()),'scale_before':scale_before,'scale_after':scaler.get_scale(),'loss_finite':True}
                if not skipped and (step%o['validation_every']==0 or step==total):
                    ae.eval();model.eval();state=g.get_state();val={};vt=time.perf_counter()
                    with torch.enable_grad() if jo['auxiliary']['pde']>0 else torch.no_grad():
                        for i in range(min(c.validation_count,max(len(names),o['validation_subset']))):
                            item=attach(data.get('validation',i));value=float(one(item,effective_examples+i));val.setdefault(item['family'],[]).append(value)
                    synchronize(device);g.set_state(state);row.update(validation_loss=float(np.mean([np.mean(v) for v in val.values()])),validation_per_family={k:float(np.mean(v)) for k,v in val.items()},validation_seconds=time.perf_counter()-vt)
                persistence_started=time.perf_counter()
                history.append(row);record.update(state='running',progress=step,attempted_updates=attempted,successful_optimizer_updates=step,skipped_updates=skipped_total,history=history[-200:],history_total=len(history),elapsed_seconds=time.perf_counter()-started);write_json(root/'run.json',record)
                if not skipped and step%c.checkpoint_every==0:save(False)
                row['persistence_seconds']=time.perf_counter()-persistence_started
                row['end_to_end_seconds']=time.perf_counter()-tick
                if overflow_streak>=o['max_consecutive_overflows']:
                    save(False);raise FloatingPointError('Repeated FP16 overflow: '+str(overflow_streak)+' consecutive skipped attempts; see overflow_diagnostics in history.json/resume.pt for unscaled invalid parameter names and scale history. Startup recovery was bounded, not established. Scheduler did not advance.')
                if stop_after and since>=stop_after:break
            final_save_started=time.perf_counter();save(step>=total);record['final_persistence_and_calibration_seconds']=time.perf_counter()-final_save_started
            record['total_wall_seconds']=time.perf_counter()-setup_started;write_json(root/'run.json',record)
        except BaseException as error:
            record.update(state='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error),error_type=type(error).__name__,traceback=__import__('traceback').format_exc());write_json(root/'run.json',record);raise
        finally:
            targets.close()
            if previous is not None:signal.signal(signal.SIGINT,previous)
        return record
