"""Explicit experimental latent correction: observations only, frozen learned T/D."""
from pathlib import Path
import hashlib,time
import numpy as np
import torch
from pdeflow.runtime import write_json,atomic_write,file_digest,code_identity
from .diagnostic_cases import load_case,sensor_truth
from .diagnostic_metrics import field_metrics,sensor_metrics
from .diagnostics import exact_sensors,_plot
from .contracts import grid_queries
from .inference import CoordinateBundle
from .joint import active_mask
from .physics import get,invariant_drift


def state_hash(modules):
    h=hashlib.sha256()
    for i,m in enumerate(modules):
        for name,t in m.state_dict().items():h.update(f'{i}:{name}'.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def correct(checkpoint,case_path,output,device='cpu',steps=20,learning_rate=.02,prior=.01):
    if type(steps)!=int or not 1<=steps<=40 or not np.isfinite([learning_rate,prior]).all() or learning_rate<=0 or prior<=0:raise ValueError('Correction requires1..40 steps, positive learning rate and positive prior strength')
    root=Path(output)
    if root.exists():raise FileExistsError('Choose a fresh correction experiment directory')
    case,p,solver,_,noise=load_case(case_path)
    if file_digest(checkpoint)!=case['checkpoint_sha256']:raise ValueError('Correction case belongs to another checkpoint')
    rows=case['observations']
    if not rows:raise ValueError('Latent correction needs measurements')
    bundle=CoordinateBundle(checkpoint,device);c=bundle.config;dev=bundle.device;start=time.perf_counter();q=grid_queries(p.grid).to(dev);shape=(len(p.grid.times),*p.grid.shape)
    _,generated=bundle.sample(p,rows,q[:1].cpu(),initial_noise=noise,**case['inference']);pn=bundle.parameter.to(dev);known=bundle.known.to(dev);f=bundle.families.to(dev)
    initial=((bundle.decoded_latent.to(dev)-bundle.mean)/bundle.scale).detach();theta=torch.nn.Parameter(initial.clone());mask=active_mask(c,known,f)[...,None]
    before_hash=state_hash((bundle.ae,bundle.velocity));optimizer=torch.optim.Adam([theta],lr=learning_rate);history=[];names=get(p.family).state
    truth={name:solver.evaluate(q.cpu().numpy(),case['true_parameter_evaluation_only'],c.interval,name).reshape(shape) for name in names};clean=sensor_truth(solver,rows,case['true_parameter_evaluation_only'],c.interval);heldout=case['heldout_observations_evaluation_only'];raw={};fields={}
    @torch.no_grad()
    def evaluate(label):
        z=theta.detach()*bundle.scale+bundle.mean;arrays={name:bundle.decode(z,q,pn,known,f,name).cpu().numpy().reshape(len(noise),*shape) for name in names};fields[label]=arrays;raw.update({label+'_'+name:a for name,a in arrays.items()})
        sensors=exact_sensors(bundle,z,pn,known,f,rows);held=exact_sensors(bundle,z,pn,known,f,heldout);parameters=bundle.ae.decode_parameter(z,pn,known,True,f).cpu().tolist()
        return {'quantities':{name:field_metrics(arrays[name],truth[name],p.grid.times,c,name,bundle.cp['normalizations'][p.family][j]) for j,name in enumerate(names)},'conditioning_sensor_metrics':sensor_metrics(sensors,rows,clean),'heldout_sensor_metrics':sensor_metrics(held,heldout,[r['value'] for r in heldout]),'parameters':parameters,'physics_invariants':invariant_drift(arrays['u'],arrays.get('v'),p.grid.times,c.lengths,parameters)}
    before=evaluate('uncorrected')
    # No reference arrays, clean values, held-out sensors or true unknown parameter
    # occur in this objective. Only supplied measurements and frozen transforms.
    for k in range(steps):
        optimizer.zero_grad(set_to_none=True);z=theta*bundle.scale+bundle.mean;per_sample=z.new_zeros(len(z))
        for j,name in enumerate(names):
            indices=[i for i,r in enumerate(rows) if r['field']==name]
            if not indices:continue
            coordinates=torch.tensor([[rows[i]['x'],rows[i]['y'],rows[i]['time']] for i in indices],device=dev);norm=bundle.cp['normalizations'][p.family][j]
            pred=bundle.ae.decode(z,coordinates,pn,known,name,f);target=torch.tensor([(rows[i]['value']-norm['mean'])/norm['scale'] for i in indices],device=dev)
            sigma=torch.tensor([rows[i].get('sigma',0.)/norm['scale'] for i in indices],device=dev)
            denominator=torch.where(sigma>0,sigma,torch.ones_like(sigma))
            per_sample+=((pred-target)/denominator).square().sum(1)/len(rows)
        trust=((theta-initial)*mask).square().sum((1,2))/mask.expand_as(theta).sum((1,2))
        loss=(per_sample+prior*trust).mean()
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite correction objective; no sigma substitution or numerical fallback')
        loss.backward();gradient=torch.nn.utils.clip_grad_norm_([theta],10.,error_if_nonfinite=True);optimizer.step()
        with torch.no_grad():theta.copy_(initial+(theta-initial)*mask)
        history.append({'correction_step':k+1,'observation_penalty':float(per_sample.mean().detach()),'prior_penalty':float(trust.mean().detach()),'total':float(loss.detach()),'gradient_norm':float(gradient)})
    after=evaluate('corrected');after_hash=state_hash((bundle.ae,bundle.velocity))
    if after_hash!=before_hash:raise RuntimeError('Frozen physical/generative model changed during correction')
    if not torch.equal(theta.detach()[~mask.expand_as(theta)],initial[~mask.expand_as(initial)]):raise RuntimeError('Inactive known/absent latent group changed')
    root.mkdir(parents=True);raw.update({'reference_'+name:a for name,a in truth.items()});raw['queries']=q.cpu().numpy();raw['initial_standardized_latent']=initial.cpu().numpy();raw['corrected_standardized_latent']=theta.detach().cpu().numpy()
    atomic_write(root/'fields.npz',lambda out:np.savez_compressed(out,**raw))
    report={'kind':'experimental_observation_only_latent_correction_v1','checkpoint_sha256':file_digest(checkpoint),'case_identity':case['case_identity'],'source':code_identity(),'device':str(dev),'steps':steps,'learning_rate':learning_rate,'prior_strength':prior,'optimizer':'Adam on standardized active initial-state/unknown-parameter latents only','history':history,'before':before,'after':after,'seconds':time.perf_counter()-start,'timing_definition':'Generation, before/after scoring, frozen hashing and correction after model loading, not optimizer-only time','initial_generation':generated,
       'frozen_state_sha256_before':before_hash,'frozen_state_sha256_after':after_hash,'objective':'Mean per-sample mean sensor squared standardized residual + prior * mean squared active standardized-latent departure from the original generated sample',
       'noise_policy':'sigma>0: exact supplied physical sigma after quantity normalization; sigma=0: squared error in recorded training-field units, a deterministic penalty, NOT assumed uncertainty. Diagnostic noiseless tolerance remains separately1e-5+1e-4*abs(value). Tiny positive sigma can fail explicitly if arithmetic becomes nonfinite.',
       'constraints':'Known coefficient and heat-absent velocity groups fixed; inferred coefficient stays in trained positive domain via existing decoder. Frozen T/D queried at exact sensor x,y,t,quantity.',
       'reference_used_in_optimization':False,'selection':'Final predeclared correction step, every sample and ensemble mean; no reference-based stopping or best-of-N',
       'limits':'Inference-time optimization changes the sample distribution. Not an unmodified generative sample, exact posterior correction, calibrated uncertainty or retrained model; held-out/field accuracy can worsen.'}
    write_json(root/'correction.json',report);_plot(root,fields,truth,p,list(p.grid.times));return report
