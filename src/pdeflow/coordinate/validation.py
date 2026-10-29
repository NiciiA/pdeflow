"""Bounded representative validation; full reference remains evaluation-only."""
from pathlib import Path
import time
import numpy as np
import torch
from pdeflow.runtime import write_json,atomic_write,file_digest,RuntimeSpec
from pdeflow.core import GridSpec,DomainSpec
from .dataset import Dataset
from .model import Autoencoder,decode_chunks
from .training import read_checkpoint,select_context,normalized_parameter,data_config
from .evaluation import error_summary,common_grid_diagnostics,interpolate_grid
from .contracts import grid_queries


def representation(checkpoint,output,device='cpu',split='validation',subset=8,allow_partial=False):
    cp,c=read_checkpoint(checkpoint,'autoencoder')
    if c.temporal_mode=='continuous':
        from .temporal_validation import representation as evaluate
        return evaluate(checkpoint,output,device,split,subset,allow_partial)
    if not (cp['completed'] or allow_partial) or split not in ('validation','test'):raise ValueError('Representation validation requires a completed stage and explicit held-out split')
    if Path(output).exists():raise FileExistsError('Validation output exists; select a fresh report')
    data=Dataset(c,Path(checkpoint).parent,False);n=cp['normalization'];dev,_=RuntimeSpec(device=device).resolve()
    ae=Autoencoder(c).to(dev).eval().requires_grad_(False);ae.load_state_dict(cp['autoencoder']);grid=data_config(c).grid;records=[];previews=[];started=time.perf_counter()
    with torch.no_grad():
        for i in range(min(subset,c.validation_count if split=='validation' else c.test_count)):
            y,d,items=data.batch([i],split);cq,cv=select_context(data.q,(y-n['mean'])/n['scale'],c);p=normalized_parameter(c,d).to(dev)
            z=ae.encode(cq[None].to(dev),cv.to(dev),p);a=(decode_chunks(ae,z,data.q.to(dev),p)*n['scale']+n['mean']).cpu().numpy()
            truth=y.numpy();shape=(len(c.times),*c.reference_shape);pred=a.reshape(shape);ref=truth.reshape(shape);axes=tuple(range(1,len(shape)))
            sa=np.abs(np.fft.rfftn(pred,axes=axes))**2;sb=np.abs(np.fft.rfftn(ref,axes=axes))**2
            field_report=None
            if c.dimension==2:
                from pdeflow.field_losses import trajectory_metrics
                field_report=trajectory_metrics(pred,ref,c.times,c.lengths,c.field_loss,n['scale'])
            records.append({'field_metrics':field_report,'id':items[0]['id'],'family':items[0]['family'],**error_summary(a,truth[0]),
               'initial_rmse':float(np.sqrt(np.mean((pred[0]-ref[0])**2))),'rmse_per_physical_time':np.sqrt(np.mean((pred-ref)**2,axis=axes)).tolist(),
               'spectral_power_relative_l2':float(np.linalg.norm(sa-sb)/max(np.linalg.norm(sb),1e-12)),
               'constant_baseline_mse':float(np.mean((truth-n['mean'])**2)),**common_grid_diagnostics(a,grid,float(d[0]))})
            if len(previews)<6:previews.append(np.stack([ref,pred,np.abs(ref-pred)]))
    baseline=float(np.mean([r['constant_baseline_mse'] for r in records]));mse=float(np.mean([r['mse_per_sample'][0] for r in records]))
    report={'stage':'autoencoder','checkpoint_sha256':file_digest(checkpoint),'split':split,'records':records,'seconds':time.perf_counter()-started,
       'constant_baseline_mse':baseline,'validation_mse':mse,'mse_ratio':mse/baseline if baseline>1e-15 else None,'dataset_identity':data.metadata['sha256'],
       'meaning':'Complete field encoder reconstruction on fixed held-out solutions. Not observation-only generation. Baseline fitted to training calibration only.'}
    write_json(output,report);atomic_write(Path(output).with_suffix('.npz'),lambda f:np.savez_compressed(f,preview=np.asarray(previews)))
    return report


def flow(checkpoint,output,device='cpu',subset=6,split='validation',counts=None,noises=None,particles=2,steps=4,resolution=16,allow_partial=False):
    from .inference import CoordinateBundle
    from pdeflow.equations import heat_problem
    if split not in ('validation','test'):raise ValueError('Choose an explicit held-out split')
    bundle=CoordinateBundle(checkpoint,device,allow_partial=allow_partial);c=bundle.config
    if c.temporal_mode=='continuous':
        from .temporal_validation import flow as evaluate
        return evaluate(checkpoint,output,device,subset,split,counts,noises,particles,steps,resolution,allow_partial)
    # Portable checkpoint carries weights; data is required ONLY for held-out evaluation.
    dependency=(Path(checkpoint).parent/bundle.cp['dependency']['path']).resolve();data=Dataset(c,dependency.parent,False)
    shape=tuple(min(resolution,n) for n in c.reference_shape);grid=GridSpec(shape,DomainSpec(c.lengths,c.origins),c.times);queries=grid_queries(grid)
    counts=counts or sorted(set([max(1,c.observation_counts[0]),min(16,c.observation_counts[1]),c.observation_counts[1]]));noises=noises or sorted(set([0.,c.noise_range[1]]))
    records=[];preview=None;started=time.perf_counter()
    for i in range(min(subset,c.validation_count if split=='validation' else c.test_count)):
        item=data.get(split,i);y=item['values'];p=heat_problem(grid,item['D']);truth=interpolate_grid(y.reshape(len(c.times),*c.reference_shape),data_config(c).grid,queries.numpy())
        for count in counts:
            for sigma in noises:
                g=torch.Generator().manual_seed(c.seed+10000+i);ids=torch.randperm(len(data.q),generator=g)[:min(count,len(data.q))];observed=torch.tensor(y)[ids]+sigma*torch.randn(len(ids),generator=g)
                rows=[{'field':'u',**{a:float(q[j]) for j,a in enumerate(grid.axis_names)},'time':float(q[-1]),'value':float(v),'sigma':sigma} for q,v in zip(data.q[ids],observed)]
                samples,diagnostic=bundle.sample(p,rows,queries,steps=steps,particles=particles,seed=91,chunk_size=512)
                row={'id':item['id'],'family':item['family'],'count':len(rows),'sigma':sigma,**error_summary(samples,truth),
                     'rmse_per_physical_time':np.sqrt(np.mean((samples.mean(0)-truth).reshape(len(c.times),-1)**2,axis=1)).tolist(),
                     'sensor_consistency':diagnostic['observation_consistency'],'physics':diagnostic['physics'],'seconds':diagnostic['seconds']}
                records.append(row)
                if preview is None:preview=np.stack([truth.reshape(len(c.times),*shape),samples.mean(0).reshape(len(c.times),*shape),np.abs(samples.mean(0)-truth).reshape(len(c.times),*shape)])
    report={'checkpoint_sha256':file_digest(checkpoint),'split':split,'records':records,'settings':{'subset':subset,'counts':counts,'noises':noises,'particles':particles,'heun_steps':steps,'query_shape':shape,'time':list(c.times),'seed':91},'seconds':time.perf_counter()-started,'meaning':'Fixed held-out solutions, individual samples and ensemble mean separate; spread is not calibrated confidence. Test split only when explicitly requested.'}
    write_json(output,report);atomic_write(Path(output).with_suffix('.npz'),lambda f:np.savez_compressed(f,preview=preview))
    return report
