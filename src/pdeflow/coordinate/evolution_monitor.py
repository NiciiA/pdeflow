"""Fixed physical-time A2 diagnostics and owner-thread previews, never training targets."""
from contextlib import contextmanager
from pathlib import Path
import time
import numpy as np
import torch
from pdeflow.runtime import atomic_write,write_json,read_json,file_digest
from .dataset import example,state_samples
from .physics import get,family_index,invariant_drift
from .training import normalized_parameter


@contextmanager
def evaluation_guard(model,generator=None,seed=None):
    """Restore every module mode and all applicable RNGs, even after failure."""
    from .random_state import capture,restore
    modes=[(m,m.training) for m in model.modules()];cpu=torch.get_rng_state();ambient=capture()
    cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    local=generator.get_state() if generator is not None else None
    if generator is not None and seed is not None:generator.manual_seed(seed)
    model.eval()
    try:yield
    finally:
        for module,mode in modes:module.training=mode
        torch.set_rng_state(cpu);restore(ambient)
        if cuda:torch.cuda.set_rng_state_all(cuda)
        if generator is not None:generator.set_state(local)


def fixed_times(c):return tuple(float(x*c.interval[1]) for x in (0.,.01,.05,.237,.5,1.))


def indices(c,per_family=1):
    # The first six examples of each enabled family are explicitly stratified.
    return list(range(min(c.validation_count,per_family*len(c.family_weights))))


def errors(pred,ref,scale):
    mse=float(np.mean((np.asarray(pred,dtype=np.float64)-ref)**2));den=float(np.mean(np.asarray(ref,dtype=np.float64)**2));floor=(float(scale)*1e-6)**2
    return {'mse':mse,'rmse':float(np.sqrt(mse)),'reference_mse':den,'relative_l2':float(np.sqrt(mse/max(den,floor))),'relative_l2_denominator_floor':floor}


def evaluate(c,ae,norms,items,times=None,resolution=16,include_oracle=False,include_physics=False):
    """Same encoded initial state for learned T and held-initial baseline.

    Physical units, periodic endpoint-excluded grid; no latent calibration in A2.
    Re-encoding the true state at t is a representation diagnostic, not a lower
    bound on the decoder's possible error and never inference conditioning.
    """
    if not 4<=resolution<=64:raise ValueError('Bounded A2 evaluation grid must be4..64')
    times=tuple(times or fixed_times(c));dev=next(ae.parameters()).device
    axes=[torch.arange(resolution,dtype=torch.float32)/resolution*L+o for L,o in zip(c.lengths,c.origins)]
    xy=torch.stack(torch.meshgrid(*axes,indexing='ij'),-1).reshape(-1,2).to(dev)
    records=[];images=[];start=time.perf_counter()
    with evaluation_guard(ae),torch.no_grad():
        for item in items:
            family=item['family'];f=torch.tensor([family_index(family)],device=dev);p=normalized_parameter(c,torch.tensor([item['parameter']],device=dev),f);known=torch.ones(1,dtype=torch.bool,device=dev);ns=norms[family]
            q=torch.as_tensor(item['context'],device=dev)[None]
            encode=lambda t:ae.encode(q,state_samples(item,item['context'],t,c,ns).to(dev)[None],p,f)
            z0=encode(0.);held=[ae.decode_state(z0,xy,j,f)[0]*n['scale']+n['mean'] for j,n in enumerate(ns)]
            predicted=[[] for _ in ns];truths=[[] for _ in ns];rows=[];identity=None
            for t in times:
                z=ae.evolve(z0,0.,t,p,known,f)
                if t==0:identity=float((z-z0).abs().max())
                zref=encode(t) if include_oracle else None
                for j,name in enumerate(get(family).state):
                    n=ns[j];pred=(ae.decode_state(z,xy,j,f)[0]*n['scale']+n['mean']).float().cpu().numpy().reshape(resolution,resolution)
                    base=held[j].float().cpu().numpy().reshape(resolution,resolution)
                    ref=item['solver'].frame(t,item['parameter'],c.interval,(resolution,resolution),name)
                    row={'id':item['id'],'family':family,'quantity':name,'physical_time':float(t),'learned':errors(pred,ref,n['scale']),'held_initial':errors(base,ref,n['scale']),'reference_change_from_initial_mse':float(np.mean((ref-item['solver'].frame(0.,item['parameter'],c.interval,(resolution,resolution),name))**2)),'normalization_scale':n['scale']}
                    if zref is not None:
                        oracle=(ae.decode_state(zref,xy,j,f)[0]*n['scale']+n['mean']).float().cpu().numpy().reshape(resolution,resolution)
                        row['reencoded_reference']=errors(oracle,ref,n['scale'])
                    rows.append(row);predicted[j].append(pred);truths[j].append(ref)
                    images.append({'id':item['id'],'family':family,'quantity':name,'physical_time':float(t),'mse':row['learned']['mse'],'rmse':row['learned']['rmse'],'reference':np.asarray(ref),'prediction':pred,'absolute_error':np.abs(pred-ref)})
            quantities={}
            t=np.asarray(times);w=np.r_[np.diff(t)[0]/2,(t[2:]-t[:-2])/2,np.diff(t)[-1]/2]/(t[-1]-t[0]) if len(t)>1 else np.ones(1)
            for j,name in enumerate(get(family).state):
                selected=[r for r in rows if r['quantity']==name];a=float(np.dot(w,[r['learned']['mse'] for r in selected]));b=float(np.dot(w,[r['held_initial']['mse'] for r in selected]))
                quantities[name]={'physical_time_quadrature_mse':a,'held_initial_mse':b,'ratio_to_held_initial':a/b if b>1e-20 else None,'beats_held_initial':a<b,'normalized_mse':a/ns[j]['scale']**2,'held_initial_normalized_mse':b/ns[j]['scale']**2}
            residual_report={}
            if include_physics:
                from .evolution_objective import decoded_residuals
                probes=torch.tensor([[[c.origins[0]+.31*c.lengths[0],c.origins[1]+.43*c.lengths[1],t*c.interval[1]] for t in (.01,.237,.5,1.)]],device=dev)
                residuals=decoded_residuals(ae,z0,probes,p,known,f,norms)
                residual_report={k:float(v.detach().square().mean()) for k,v in residuals.items() if (k=='heat')==(family=='heat')}
            records.append({'normalized_strong_residual_mse':residual_report,'residual_measure':'Four declared coordinate/time probes through actual T/D, not a uniform domain integral','id':item['id'],'family':family,'case':item['case'],'parameter':item['parameter'],'parameter_status':'known supervised A2 scalar','initial_identity_max_error':identity,'quantities':quantities,'per_time':rows,'invariants':invariant_drift(np.asarray(predicted[0]),np.asarray(predicted[1]) if family=='wave' else None,times,c.lengths,item['parameter'])})
    learned=np.mean([np.mean([v['normalized_mse'] for v in r['quantities'].values()]) for r in records]);held=np.mean([np.mean([v['held_initial_normalized_mse'] for v in r['quantities'].values()]) for r in records])
    report={'records':records,'aggregate_normalized_mse':float(learned),'held_initial_normalized_mse':float(held),'ratio_to_held_initial':float(learned/held) if held>1e-20 else None,'beats_held_initial':bool(learned<held),'physical_times':list(times),'resolution':resolution,'seconds':time.perf_counter()-start,'aggregation':'Equal fixed realizations, equal active quantities; periodic spatial node mean and normalized trapezoidal physical-time quadrature. RMSE=sqrt(MSE), relative L2 divides by max(reference mean square,(global training quantity scale*1e-6)^2). Per-quantity physical metrics are separate.','meaning':'Fixed reference-encoded initial state, then CURRENT learned evolution. Held baseline decodes the SAME initial latent unchanged; not observation-only inference or scientific validation.'}
    return report,images


def publish_preview(root,c,ae,norms,items,update):
    # Only the training owner calls this after a complete optimizer boundary.
    report,images=evaluate(c,ae,norms,items,(.237*c.interval[1],),resolution=12)
    root=Path(root);path=root/'evolution-preview.npz';frames=np.stack([np.stack([x['reference'],x['prediction'],x['absolute_error']]) for x in images])
    atomic_write(path,lambda out:np.savez_compressed(out,frames=frames))
    rows=[{k:v for k,v in x.items() if k not in ('reference','prediction','absolute_error')} for x in images]
    record={'stage':'evolution','checkpoint_update':update,'preview_generation_update':update,'created_unix':time.time(),'array':path.name,'array_sha256':file_digest(path),'rows':rows,'meaning':report['meaning']}
    write_json(root/'evolution-preview.json',record);return record


def read_preview(root):
    root=Path(root);path=root/'evolution-preview.json'
    if not path.exists():return None
    meta=read_json(path);array=root/'evolution-preview.npz'
    from io import BytesIO
    import hashlib
    payload=array.read_bytes()
    if meta['array']!=array.name or hashlib.sha256(payload).hexdigest()!=meta['array_sha256']:return None  # publication race: never show mismatched weights/labels
    with np.load(BytesIO(payload),allow_pickle=False) as values:frames=values['frames'].copy()
    return meta,frames
