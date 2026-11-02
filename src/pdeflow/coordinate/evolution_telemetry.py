"""Detached, bounded stage measurements; never used to weight the objective."""
import torch


def record(components,key,values):
    components.setdefault('_measurements',{}).setdefault(key,[]).append(values.detach().float().reshape(-1))


def summary(components):
    return {**{k:v for k,v in components.items() if k!='_measurements'},**{k:torch.cat(v).mean() for k,v in components.get('_measurements',{}).items()}}


def flow_rows(components,flow,extra,info):
    """Reduce actual per-example objectives over ALL forward microbatches.

    Counts sum, raw endpoint penalties average by example, beta is a step setting.
    Recording a last-microbatch mean would disagree with the differentiated loss.
    """
    record(components,'flow_matching',flow)
    record(components,'objective_observation_auxiliary',extra)
    counts={'observation_selected','observation_population','observation_gaussian_values','observation_tolerance_values','observation_eligible_examples'}
    for key,value in info.items():
        if key in counts:components[key]=components.get(key,0)+value
        elif key in ('observation_endpoint_loss','observation_effective_beta_case_mean'):record(components,key,value.expand(flow.numel()))
        elif key=='observation_flow_s_min':components[key]=torch.minimum(components.get(key,value),value)
        elif key=='observation_flow_s_max':components[key]=torch.maximum(components.get(key,value),value)
        else:components[key]=value


def time_rows(components,rows,horizon=1.):
    for r in rows:
        dt=r.tb-r.ta
        for key,value in {'a2_ta':r.ta,'a2_tb':r.tb,'a2_dt':dt,'a2_zero_dt_fraction':float(dt==0),'a2_early_dt_fraction':float(0<dt<=.01*horizon),'a2_tb_initial_fraction':float(r.tb==0),'a2_tb_early_fraction':float(0<r.tb<=.01*horizon),'a2_tb_intermediate_fraction':float(.01*horizon<r.tb<=.5*horizon),'a2_tb_late_fraction':float(r.tb>.5*horizon)}.items():
            record(components,key,torch.tensor([value]))


class UpdateProbe:
    """Deterministic <=32 values/tensor, only at first/cadence/validation updates.

    This is a sampled actual parameter delta, not a full-network update norm.
    No RNG, gradients, optimizer buffers or model tensors are modified.
    """
    def __init__(self,named):
        self.rows=[]
        for name,p in named:
            if not p.requires_grad:continue
            ix=torch.linspace(0,p.numel()-1,min(32,p.numel()),device=p.device).long()
            self.rows.append((name,p,ix,p.detach().reshape(-1)[ix].float().clone()))
    def result(self):
        result={}
        for group in ('evolution','graph_encoder'):
            rows=[(p,ix,before) for name,p,ix,before in self.rows if name.startswith(group+'.')]
            if not rows:continue
            before=torch.cat([a for p,ix,a in rows]);after=torch.cat([p.detach().reshape(-1)[ix].float() for p,ix,a in rows]);delta=after-before
            gradients=[p.grad.detach().reshape(-1)[ix].float() for p,ix,a in rows if p.grad is not None]
            gradnorm=torch.cat(gradients).norm() if gradients else torch.tensor(0.)
            result[group]={'sampled_elements':len(before),'parameter_delta_l2':float(delta.norm()),'parameter_delta_max_abs':float(delta.abs().max()),'relative_parameter_delta_l2':float(delta.norm()/before.norm().clamp_min(1e-20)), 'sampled_clipped_gradient_l2':float(gradnorm) if torch.isfinite(gradnorm) else None}
        return {'modules':result,'scope':'Deterministic bounded parameter samples, not full-network update norm; gradients are unscaled and clipped at this boundary.'}


def ownership(ae,optimizer):
    trainable={name:p for name,p in ae.named_parameters() if p.requires_grad};ids={id(p) for g in optimizer.param_groups for p in g['params']}
    if ids!={id(p) for p in trainable.values()} or any(not name.startswith(('evolution.','graph_encoder.')) for name in trainable):raise ValueError('A2 optimizer ownership differs from trainable physical evolution/graph modules')
    return {'trainable_parameters':sum(p.numel() for p in trainable.values()),'trainable_modules':['evolution','graph_encoder'],'frozen_modules':['encoder','decoder','coefficient'],'optimizer_ownership_verified':True,'latent_transform':'Raw frozen complete-state encoding; no flow latent calibration in A2','known_parameter':'Normalized typed D/c supplied to T; wave physics uses c squared'}
