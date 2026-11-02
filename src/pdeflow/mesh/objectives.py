"""Decoded physical objectives, candidate scoring, and optional soft consistency."""
import torch
from pdeflow.field_losses import differentiable_loss,settings as field_settings
from .state import active_mask,group_loss,enforce,parameter_physical
from .sampling import fgn_noise,flow_path,flow_endpoint,diffusion_path,diffusion_endpoint,diffusion_schedule


def fair_crps(a,b,y):return .5*((a-y).abs()+(b-y).abs()-(a-b).abs())


def fair_energy(a,b,y,weights=None):
    if weights is None:weights=torch.full_like(a,1/a.shape[-1])
    def norm(x):return torch.linalg.vector_norm(x.float()*weights.sqrt(),dim=-1)
    return .5*(norm(a-y)+norm(b-y)-norm(a-b))


def predictions(model,state,batch,initial):
    coords=batch.q0 if initial else batch.qt;time=0. if initial else batch.times;evolved=model.evolve(state,time,batch.families)
    return torch.stack([model.decoder(evolved,coords[:,j],batch.families)[...,j] for j in range(2)],1)


def quantity_mean(values,families):
    mask=torch.stack([torch.ones_like(families),families.eq(1)],1).to(values.dtype)
    return (values*mask).sum(1)/mask.sum(1)


def field_loss(c,pred,target,weights,families):
    rows=[]
    for i in range(len(pred)):
        kinds=2 if families[i]==1 else 1;losses=[differentiable_loss(pred[i:i+1,j],target[i:i+1,j],weights[i:i+1,j],field_settings(c.field_loss).type,validate=False) for j in range(kinds)];rows.append(torch.stack(losses).mean())
    return torch.stack(rows)


def supervised(model,state,batch):
    c=model.c;initial=predictions(model,state,batch,True);l0=field_loss(c,initial,batch.y0,batch.w0,batch.families);lt=torch.zeros_like(l0)
    wi=c.objective.get('initial',1. if not c.kind['trajectory'] else .5);wt=c.objective.get('trajectory',.5) if c.kind['trajectory'] else 0.
    if c.kind['trajectory']:lt=field_loss(c,predictions(model,state,batch,False),batch.yt,batch.wt,batch.families)
    scalar=(state[:,-1,0]-batch.target_state[:,-1,0]).square()*(~batch.known)
    nodal=(state[:,:c.nodes]-batch.target_state[:,:c.nodes]).square().mean(1);anchor=quantity_mean(nodal,batch.families)
    spectral=torch.zeros_like(l0);spec=field_settings(c.field_loss)
    if spec.spectral_weight>0:
        from .geometry import regular
        from pdeflow.coordinate.evolution_objective import phase_spectral_mse
        xy=regular((spec.spectral_resolution,)*2).to(state.device)*state.new_tensor(c.domain['lengths'])+state.new_tensor(c.domain['origins']);parts=[]
        for time,target,weight in [(0.,batch.spectral0,wi),*(([(batch.times,batch.spectralt,wt)]) if c.kind['trajectory'] else [])]:
            pred=model.decode(state,xy,batch.families,time).transpose(1,2).reshape(len(state),2,spec.spectral_resolution,spec.spectral_resolution)
            losses=torch.stack([phase_spectral_mse(pred[:,j],target[:,j],spec.spectral_frequency_power) for j in range(2)],1);parts.append(weight*quantity_mean(losses,batch.families))
        spectral=spec.spectral_weight*sum(parts)
    result=wi*l0+wt*lt+c.objective.get('parameter',.1)*scalar+c.objective.get('anchor',.1)*anchor+spectral
    return result,{'initial_field':l0,'trajectory_field':lt,'parameter':scalar,'physical_anchor':anchor,'spectral':spectral}


def observation_loss(model,state,batch,norms,selection=None):
    c=model.c;budget=int(c.objective.get('observation_sensor_budget',8));values=[]
    for i in range(len(state)):
        indices=torch.where(batch.observations[2][i])[0][:budget]
        if selection is not None and not bool(selection[i]):indices=indices[:0]
        if not len(indices):values.append(state[i].sum()*0);continue
        coords=batch.observations[0][i:i+1,indices];qty=batch.observations[3][i:i+1,indices];truth=batch.observations[1][i,indices,0];sigma=batch.observations[1][i,indices,1];pred=model.decode_points(state[i:i+1],coords,qty,batch.families[i:i+1])[0]
        scale=pred.new_tensor([x['scale'] for x in norms[('heat','wave')[int(batch.families[i])]]])[qty[0]];residual=(pred-truth)*scale;physical_sigma=sigma*scale;tol=c.objective.get('observation_tolerance_relative',.05)*scale;discrepancy=c.objective.get('observation_discrepancy_relative',.05)*scale
        denominator=torch.where(sigma>0,physical_sigma.square()+discrepancy.square(),tol.square())
        if (denominator<=0).any():raise ValueError('Observation penalty needs explicit positive exact tolerance or Gaussian discrepancy')
        values.append((.5*residual.square()/denominator).mean())
    return torch.stack(values)


def physical_loss(model,state,batch,norms,q,force_diagnostics=False):
    c=model.c;weights={k:(1. if force_diagnostics else c.objective.get(k,0.)) for k in ('heat','kinematic','acceleration','heat_mean','heat_dissipation')}
    if not any(weights.values()):return state.float().sum((1,2))*0,{}
    if not c.kind['trajectory']:raise ValueError('Physical evolution consistency is unavailable for initial-only M1')
    from pdeflow.coordinate.model import derivative_attention
    with torch.enable_grad(),torch.autocast(state.device.type,enabled=False),derivative_attention():
        points=q.detach().float().clone().requires_grad_(True);kind=torch.zeros(points.shape[:2],dtype=torch.long,device=state.device);u=model.decode_points(state.float(),points,kind,batch.families);v=model.decode_points(state.float(),points,torch.ones_like(kind),batch.families)
        table=u.new_tensor([[[n['mean'],n['scale']] for n in norms.get(name,[{'mean':0.,'scale':1.}]*2)] for name in ('heat','wave')]);ns=table[batch.families];u=u*ns[:,0,1,None]+ns[:,0,0,None];v=v*ns[:,1,1,None]+ns[:,1,0,None];du=torch.autograd.grad(u.sum(),points,create_graph=True)[0];duration=c.interval[1]-c.interval[0];kin=((du[...,-1]-v)/(ns[:,0,1,None]/duration)).square().mean(1)*batch.families.eq(1);heat=acc=torch.zeros_like(kin)
        if weights['heat'] or weights['acceleration']:
            lap=sum(torch.autograd.grad(du[...,j].sum(),points,create_graph=True)[0][...,j] for j in range(2));p=parameter_physical(c,state[:,-1,0],batch.families)
            heat=((du[...,-1]-p[:,None]*lap)/(ns[:,0,1,None]/duration)).square().mean(1)*batch.families.eq(0)
            if weights['acceleration']:
                dv=torch.autograd.grad(v.sum(),points,create_graph=True)[0];acc=((dv[...,-1]-p[:,None].square()*lap)/(ns[:,1,1,None]/duration)).square().mean(1)*batch.families.eq(1)
        mean=energy=torch.zeros_like(kin)
        if weights['heat_mean'] or weights['heat_dissipation']:
            from .geometry import regular
            xy=regular((4,4)).to(state.device)*state.new_tensor(c.domain['lengths'])+state.new_tensor(c.domain['origins']);initial=model.decode(state.float(),xy,batch.families,0.)[...,0];later=model.decode(state.float(),xy,batch.families,q[:,:,-1].mean(1))[...,0]
            mean=(later.mean(1)-initial.mean(1)).square()*batch.families.eq(0)
            # Soft coarse-grid diagnostics, not exact continuous conservation.
            energy=torch.relu(later.var(1,unbiased=False)-initial.var(1,unbiased=False)).square()*batch.families.eq(0)
        return weights['heat']*heat+weights['kinematic']*kin+weights['acceleration']*acc+weights['heat_mean']*mean+weights['heat_dissipation']*energy,{'heat_residual':heat,'wave_kinematic':kin,'wave_acceleration':acc,'heat_mean_drift':mean,'heat_variance_increase':energy}


def loss(model,batch,stage,norms,noise,s,k,physics_q):
    c=model.c;components={};context=None
    if stage!='dynamics':context=model.condition(batch.observations,batch.parameter,batch.known,batch.families)
    if stage=='dynamics':state=batch.target_state;objective,components=supervised(model,state,batch)
    elif stage in ('reconstruction','central'):
        zero=fgn_noise(c,len(batch.parameter),torch.Generator(),batch.parameter.device,True) if c.architecture=='mesh_m3_fgn_v1' else None;state=model.initial(context,batch.parameter,batch.known,batch.families,zero);objective,components=supervised(model,state,batch)
    elif stage=='stochastic':
        zero=fgn_noise(c,len(batch.parameter),torch.Generator(),batch.parameter.device,True);central=model.initial(context,batch.parameter,batch.known,batch.families,zero);central_loss,parts=supervised(model,central,batch);a=model.initial(context,batch.parameter,batch.known,batch.families,noise[0]);b=model.initial(context,batch.parameter,batch.known,batch.families,noise[1]);scores=[];energies=[]
        for initial,weight in ((True,c.objective.get('initial',.5)),(False,c.objective.get('trajectory',.5))):
            pa=predictions(model,a,batch,initial)[...,:batch.domain_queries];pb=predictions(model,b,batch,initial)[...,:batch.domain_queries];y=(batch.y0 if initial else batch.yt)[...,:batch.domain_queries];scores.append(weight*quantity_mean(fair_crps(pa,pb,y).mean(-1),batch.families));energies.append(weight*quantity_mean(torch.stack([fair_energy(pa[:,j],pb[:,j],y[:,j]) for j in range(2)],1),batch.families))
        crps=sum(scores)+c.objective.get('parameter',.1)*fair_crps(a[:,-1,0],b[:,-1,0],batch.target_state[:,-1,0])*(~batch.known);energy=sum(energies);objective=c.objective.get('crps',1.)*crps+c.objective.get('central',.1)*central_loss+c.objective.get('energy',0.)*energy;state=central;components={**parts,'fair_crps':crps,'zero_noise_central':central_loss,'energy_score':energy}
    else:
        if stage=='flow':current,target=flow_path(c,batch.target_state,noise,s,batch.parameter,batch.known,batch.families);prediction=model.generative(current,s,context,batch.known,batch.families);state=flow_endpoint(current,prediction,s)
        elif stage=='diffusion':
            current,target=diffusion_path(c,batch.target_state,noise,k,batch.parameter,batch.known,batch.families);prediction=model.generative(current,k.float()/c.sampler['diffusion_steps'],context,batch.known,batch.families);alpha,sigma=diffusion_schedule(c.sampler['diffusion_steps'],current.device);state,_=diffusion_endpoint(current,prediction,alpha[k,None,None],sigma[k,None,None])
        else:raise ValueError('Unsupported candidate stage')
        state=enforce(c,state,batch.parameter,batch.known,batch.families);objective=group_loss(c,prediction,target,batch.known,batch.families);components[stage+'_physical_state_objective']=objective
    if c.objective.get('observation',0)>0:
        if not c.kind['trajectory'] and bool(batch.observations[0][...,2][batch.observations[2]].ne(0).any()):raise ValueError('M1 cannot evaluate later-time observation likelihood without evolution')
        selection=s>=c.objective.get('observation_minimum_s',.5) if stage=='flow' else None;obs=observation_loss(model,state,batch,norms,selection);objective=objective+c.objective['observation']*obs;components['observation_penalty']=obs
    if any(c.objective.get(x,0)>0 for x in ('heat','kinematic','acceleration','heat_mean','heat_dissipation')):
        states=(a,b) if stage=='stochastic' else (state,)
        penalties=[physical_loss(model,draw,batch,norms,physics_q) for draw in states]
        objective=objective+sum(x[0] for x in penalties)/len(penalties);components.update({key:sum(x[1][key] for x in penalties)/len(penalties) for key in penalties[0][1]})
    components['total_objective']=objective
    return objective,components
