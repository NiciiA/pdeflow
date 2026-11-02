"""Batched A1/A2 forwards, preserving exact per-realization quantity reductions."""
from dataclasses import dataclass
import numpy as np
import torch
from .dataset import state_samples_batch
from .temporal import state_time,evolution_time
from .physics import family_index
from .joint import active_mask,options as joint_options
from .training import normalized_parameter
from pdeflow.field_losses import settings as loss_settings


@dataclass
class StatePlan:
    item:dict
    ta:float
    tb:float
    tm:float
    queries:list
    seed:int = 0


def plans(c,items,g,iteration,stage,norms,cache,timings,item_quantity,grouped_queries):
    rows=[]
    for i,item in enumerate(items):
        phase=evolution_time(c,iteration+i,g) if stage=='evolution' else state_time(c,iteration+i,g,'query')
        if stage=='autoencoder':ta=state_time(c,iteration+i,g,'encoder');tb=ta;tm=ta
        else:
            tb=phase;ta=float(torch.rand((),generator=g))*tb if (iteration+i)%2 else 0.;tm=(ta+tb)/2
        fields=[]
        for j,name in enumerate(('u','v') if item['family']=='wave' else ('u',)):
            view=item_quantity(item,name);normal=norms[item['family']][j]
            with timings.measure('region_supervision',host=True):q,y,w,_=grouped_queries(c,[view],g,normal,cache,[tb])
            if not torch.isfinite(w).all() or (w<0).any():raise ValueError('Invalid reference-derived field weights')
            fields.append((q[0,:,:2],(y[0]-normal['mean'])/normal['scale'],w[0]))
        rows.append(StatePlan(item,ta,tb,tm,fields,c.seed+iteration+i))
    return rows


def forward(c,ae,rows,stage,norms,device,timings,frozen,differentiable_loss,item_quantity,components,return_parts=False):
    from .evolution_telemetry import record,time_rows
    if stage=='evolution':time_rows(components,rows,c.interval[1])
    items=[r.item for r in rows];n=len(items)
    with timings.measure('transfer',host=True):
        f=torch.tensor([family_index(x['family']) for x in items],device=device)
        p=normalized_parameter(c,torch.tensor([x['parameter'] for x in items],device=device),f)
        xy=torch.as_tensor(np.stack([x['context'] for x in items]),device=device)
        ta=torch.tensor([r.ta for r in rows],device=device);tb=torch.tensor([r.tb for r in rows],device=device);tm=torch.tensor([r.tm for r in rows],device=device)
    def encode(requests,state):
        with timings.measure('transfer',host=True):
            si=[x for x,_ in requests];sf=torch.tensor([family_index(x['family']) for x in si],device=device)
            sp=normalized_parameter(c,torch.tensor([x['parameter'] for x in si],device=device),sf)
            sx=torch.as_tensor(np.stack([x['context'] for x in si]),device=device)
            if device.type=='cuda':state=state.pin_memory()
            state=state.to(device,non_blocking=device.type=='cuda')
        with timings.measure('encoder_compute'):return ae.encode(sx,state,sp,sf)
    def prepare(requests):return state_samples_batch([x for x,_ in requests],[t for _,t in requests],c,norms)
    if stage=='autoencoder':
        with timings.measure('reference_evaluation',host=True):state=state_samples_batch(items,[r.ta for r in rows],c,norms)
        z=encode([(r.item,r.ta) for r in rows],state);zt=None;z0=None
    else:
        requests=[(r.item,t) for r in rows for t in (r.ta,r.tb,0.)]
        with torch.no_grad(),timings.measure('frozen_target',host=True):
            states=frozen.resolve(requests,torch.is_autocast_enabled(device.type),len(rows),prepare,encode,device)
            with timings.measure('transfer',host=True):states=states.to(device,non_blocking=device.type=='cuda')
            states=states.reshape(n,3,c.tokens,c.width);z=states[:,0];zt=states[:,1];z0=states[:,2]
    known=torch.ones(n,dtype=torch.bool,device=device)
    if stage=='evolution':
        with timings.measure('graph_compute'):graph=ae.graph_encoder(f)
        with timings.measure('evolution_compute'):evolved=ae.evolve(z,ta,tb,p,known,f,graph)
        field_state=evolved
    else:field_state=z
    field_losses=[[] for _ in rows]
    for j,name in enumerate(('u','v')):
        active=[i for i,r in enumerate(rows) if j<len(r.queries)]
        if not active:continue
        with timings.measure('transfer',host=True):
            ix=torch.tensor(active,device=device);q=torch.stack([rows[i].queries[j][0] for i in active]).to(device,non_blocking=device.type=='cuda')
            targets=torch.stack([rows[i].queries[j][1] for i in active]).to(device,non_blocking=device.type=='cuda');weights=torch.stack([rows[i].queries[j][2] for i in active]).to(device,non_blocking=device.type=='cuda')
        with timings.measure('decoder_compute'):
            pred=ae.decode_state(field_state[ix],q,j,f[ix])
            for k,i in enumerate(active):
                # Individual reduction retains the public loss/gradient checks
                # and equal quantity weighting; forwards are fully batched.
                item_quantity(rows[i].item,name)
                loss=differentiable_loss(pred[k:k+1],targets[k:k+1],weights[k:k+1],loss_settings(c.field_loss).type,validate=False)
                field_losses[i].append(loss);record(components,rows[i].item['family']+'_'+name,loss)
    from .quantity_weights import combine,settings as quantity_settings
    field=torch.stack([combine(c,x) if stage=='autoencoder' else torch.stack(x).mean() for x in field_losses])
    from .evolution_objective import structured_field_loss,physics_loss
    with timings.measure('spectral_field_compute'):
        extra,info=structured_field_loss(c,ae,field_state,rows,f,norms,device,stage=='autoencoder')
    domain_field=field;spectral_term=extra
    if stage=='evolution':record(components,'objective_domain_field',field);record(components,'objective_spectral',extra)
    field=field+extra;components.update(info)
    if stage=='autoencoder':
        scalar=(ae.decode_parameter(z,families=f)-p).float().square();components['scalar']=scalar[-1].detach()
        return field+quantity_settings(c.representation_weights)['parameter']*scalar+joint_options(c)['latent_regularization']*z.float().square().mean((1,2))
    mask=active_mask(c,known,f)[...,None]
    def latent_error(a,b):return ((a.float()-b.float()).square()*mask).sum((1,2))/mask.expand_as(a).sum((1,2))
    alignment=latent_error(evolved,zt)
    with timings.measure('composition_compute'):
        via=ae.evolve(ae.evolve(z0,0.,tm,p,known,f,graph),tm,tb,p,known,f,graph)
        direct=ae.evolve(z0,0.,tb,p,known,f,graph)
        composition=latent_error(via,direct)
    physical=[[] for _ in rows]
    with timings.measure('composition_decode_compute'):
        for j in range(2):
            active=[i for i,r in enumerate(rows) if j<len(r.queries)]
            if not active:continue
            ix=torch.tensor(active,device=device);points=xy[ix,:min(32,c.encoder_points)]
            a=ae.decode_state(via[ix],points,j,f[ix]);b=ae.decode_state(direct[ix],points,j,f[ix]);values=(a.float()-b.float()).square().mean(1)
            for k,i in enumerate(active):physical[i].append(values[k])
    physical=torch.stack([torch.stack(x).mean() for x in physical])
    components.update(latent_alignment=alignment[-1].detach(),composition=composition[-1].detach(),decoded_composition=physical[-1].detach())
    # Independent seed per realization keeps the base supervision RNG unchanged
    # and makes this optional term invariant to microbatch partitioning.
    extras=[]
    with timings.measure('physics_residual_compute'):
        for i,r in enumerate(rows):
            extra,info=physics_loss(c,ae,z0[i:i+1],p[i:i+1],known[i:i+1],f[i:i+1],norms,r.seed)
            extras.append(extra[0]);components.update(info)
    kinematics=[]
    from .evolution_objective import kinematic_loss
    with timings.measure('kinematic_compute'):
        for i,r in enumerate(rows):
            value,info=kinematic_loss(c,ae,z0[i:i+1],p[i:i+1],known[i:i+1],f[i:i+1],norms,r.seed+911)
            kinematics.append(value[0])
            for key,value in info.items():
                if key in ('kinematic_examples','kinematic_queries'):components[key]=components.get(key,0)+value
                elif key=='kinematic_raw':record(components,key,value.reshape(1))
                else:components[key]=value
    kinematic_term=torch.stack(kinematics);record(components,'objective_kinematic',kinematic_term)
    alignment_term=c.evolution.get('latent_weight',.1)*alignment;composition_term=c.evolution.get('composition_weight',.01)*(composition+physical);physics_term=torch.stack(extras)
    record(components,'objective_alignment',alignment_term);record(components,'objective_composition',composition_term);record(components,'objective_physics',physics_term)
    loss=field+alignment_term+composition_term+physics_term+kinematic_term
    if return_parts:return loss,{'domain':domain_field,'spectral':spectral_term,'alignment':alignment_term,'composition':composition_term,'physics':physics_term,'kinematic':kinematic_term}
    return loss
