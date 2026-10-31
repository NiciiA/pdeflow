"""Held-out, physical-unit diagnostics for all three shared model stages."""
from pathlib import Path
import time
import numpy as np
import torch
from pdeflow.runtime import write_json,atomic_write,file_digest,RuntimeSpec
from pdeflow.core import GridSpec,DomainSpec
from .dataset import Dataset,state_samples
from .model import Autoencoder
from .training import read_checkpoint,normalized_parameter
from .physics import problem,family_index,get,invariant_drift
from .contracts import grid_queries
from .evaluation import error_summary,spectral_diagnostics
from pdeflow.field_losses import trajectory_metrics


def evaluation_grid(c,resolution=16):
    return GridSpec((resolution,resolution),DomainSpec(c.lengths,c.origins),(0.,.01,.05,.1,.237,.5,.75,1.))


def temporal_errors(prediction,reference,times,interval=(0.,1.)):
    a=np.asarray(prediction).reshape(len(times),-1);b=np.asarray(reference).reshape(a.shape);mse=np.mean((a-b)**2,1);t=np.asarray(times)
    return {'physical_times':list(times),'rmse_per_physical_time':np.sqrt(mse).tolist(),'initial_rmse':float(np.sqrt(mse[0])),
       'early_rmse':float(np.sqrt(mse[t<=.1].mean())),'uniform_space_time_mse':float(np.trapezoid(mse,t)/(t[-1]-t[0])),
       'uniform_metric_definition':'Spatial uniform node mean, trapezoidal quadrature in physical time on the explicitly recorded nonuniform evaluation times; not training-weighted loss'}


def _data(cp,c,checkpoint):
    root=Path(checkpoint).parent
    if cp.get('dataset_root'):root=(root/cp['dataset_root']).resolve()
    elif cp.get('dependency'):
        dep=(root/cp['dependency']['path']).resolve();other,_=read_checkpoint(dep);return _data(other,c,dep)
    return Dataset(c,root,False)


def representation(checkpoint,output,device='cpu',split='validation',subset=8,allow_partial=False):
    return physical(checkpoint,output,device,split,subset,allow_partial,'autoencoder')


def evolution(checkpoint,output,device='cpu',split='validation',subset=8,allow_partial=False):
    return physical(checkpoint,output,device,split,subset,allow_partial,'evolution')


def physical(checkpoint,output,device,split,subset,allow_partial,stage):
    cp,c=read_checkpoint(checkpoint,stage)
    if not (cp['completed'] or allow_partial) or split not in ('validation','test'):raise ValueError('Requires a completed stage and explicit held-out split')
    if Path(output).exists():raise FileExistsError('Choose a fresh validation report')
    data=_data(cp,c,checkpoint);dev,_=RuntimeSpec(device=device,threads=2).resolve();ae=Autoencoder(c).to(dev).eval().requires_grad_(False);ae.load_state_dict(cp['autoencoder']);grid=evaluation_grid(c);xy=grid_queries(GridSpec(grid.shape,grid.domain,(0.,)))[:,:2].to(dev);records=[];previews=[];started=time.perf_counter()
    with torch.no_grad():
        for i in range(min(subset,c.validation_count if split=='validation' else c.test_count)):
            item=data.get(split,i);family=item['family'];f=torch.tensor([family_index(family)],device=dev);p=normalized_parameter(c,torch.tensor([item['parameter']],device=dev),f);known=torch.ones(1,dtype=torch.bool,device=dev);norms=cp['normalizations'][family];cq=torch.tensor(item['context'],device=dev)[None]
            encode=lambda t:ae.encode(cq,state_samples(item,item['context'],t,c,norms).to(dev)[None],p,f)
            zinit=encode(0.);preds=[[],[]];truths=[[],[]];composition=[]
            for t in grid.times:
                z=encode(t) if stage=='autoencoder' else ae.evolve(zinit,0.,t,p,known,f)
                for j,name in enumerate(get(family).state):
                    preds[j].append((ae.decode_state(z,xy,j,f)[0]*norms[j]['scale']+norms[j]['mean']).cpu().numpy().reshape(grid.shape));truths[j].append(item['solver'].frame(t,item['parameter'],c.interval,grid.shape,name))
                via=ae.evolve(ae.evolve(zinit,0.,t/2,p,known,f),t/2,t,p,known,f)
                if stage=='evolution':composition.append({'time':t,'latent_mse':float((via-z).square().mean()),'decoded_normalized_mse':float(torch.stack([(ae.decode_state(via,xy,j,f)-ae.decode_state(z,xy,j,f)).square().mean() for j in range(len(get(family).state))]).mean())})
            quantities={}
            for j,name in enumerate(get(family).state):
                a=np.asarray(preds[j]);b=np.asarray(truths[j]);base=np.full_like(b,norms[j]['mean']);metrics=temporal_errors(a,b,grid.times,c.interval);baseline=temporal_errors(base,b,grid.times,c.interval)['uniform_space_time_mse'];sa=np.abs(np.fft.rfft2(a))**2;sb=np.abs(np.fft.rfft2(b))**2
                quantities[name]={**error_summary(a.reshape(1,-1),b.ravel()),**metrics,'phase_amplitude':spectral_diagnostics(a,b),'constant_baseline_mse':baseline,'normalized_validation_mse':metrics['uniform_space_time_mse']/norms[j]['scale']**2,'normalized_constant_baseline_mse':baseline/norms[j]['scale']**2,'spectral_power_relative_l2':float(np.linalg.norm(sa-sb)/max(np.linalg.norm(sb),1e-12)), 'field_metrics':trajectory_metrics(a,b,grid.times,c.lengths,c.field_loss,norms[j]['scale'])}
                if len(previews)<8:previews.append(np.stack([b,a,np.abs(a-b)]))
            scalar=float(ae.decode_parameter(zinit,p,torch.zeros_like(known),True,f)[0]);records.append({'id':item['id'],'family':family,'case':item['case'],'quantities':quantities,'composition':composition,'parameter_error':abs(scalar-item['parameter']),'initial_identity_max_error':float((ae.evolve(zinit,0.,0.,p,known,f)-zinit).abs().max()),'invariants':invariant_drift(np.asarray(preds[0]),np.asarray(preds[1]) if family=='wave' else None,grid.times,c.lengths,item['parameter'])})
    # Equal examples within family, equal quantities within example, equal enabled families.
    def aggregate(key):return float(np.mean([np.mean([np.mean([q[key] for q in r['quantities'].values()]) for r in records if r['family']==f]) for f in c.family_weights if any(r['family']==f for r in records)]))
    mse=aggregate('normalized_validation_mse');baseline=aggregate('normalized_constant_baseline_mse')
    report={'stage':stage,'checkpoint_sha256':file_digest(checkpoint),'split':split,'records':records,'validation_mse':mse,'constant_baseline_mse':baseline,'mse_ratio':mse/baseline if baseline>1e-15 else None,'aggregation':'Equal family, example and active quantity; aggregate gate uses training-normalized MSE, constant baseline uses identical normalization/objective. Physical per-quantity metrics retained.','seconds':time.perf_counter()-started,'dataset_identity':data.metadata['sha256'],'meaning':'Complete reference state encoding for A1; encoded INITIAL state then learned evolution for A2. Not observation-only reconstruction; tiny training is not accuracy validation.'}
    write_json(output,report);atomic_write(Path(output).with_suffix('.npz'),lambda f:np.savez_compressed(f,preview=np.asarray(previews)));return report


def flow(checkpoint,output,device='cpu',subset=6,split='validation',counts=None,noises=None,particles=2,steps=4,resolution=16,allow_partial=False,placements=None,seeds=None):
    from .inference import CoordinateBundle
    from pdeflow.heat_placement import place
    if split not in ('validation','test'):raise ValueError('Choose an explicit held-out split')
    if Path(output).exists():raise FileExistsError('Choose a fresh report')
    bundle=CoordinateBundle(checkpoint,device,allow_partial);c=bundle.config;data=_data(bundle.cp,c,checkpoint);grid=evaluation_grid(c,resolution);q=grid_queries(grid);records=[];previews=[];started=time.perf_counter();counts=counts or [min(12,c.observation_counts[1])];noises=noises if noises is not None else [0.];seeds=seeds or [91];placements=placements or ['random']
    for i in range(min(subset,c.validation_count if split=='validation' else c.test_count)):
        item=data.get(split,i);family=item['family'];parameter=get(family).parameter
        for known in (True,False):
            for count in counts:
                for sigma in noises:
                    for seed in seeds:
                        for placement in placements:
                            xy,provenance=place(count,placement,seed,c.lengths,c.origins,field=item['solver'].frame(.237,item['parameter'],c.interval,(32,32)) if placement=='heat' else None);rng=np.random.default_rng(seed);at=rng.uniform(0,1,count);ids=rng.integers(2,size=count) if family=='wave' else np.zeros(count,dtype=int);rows=[]
                            for j,name in enumerate(get(family).state):
                                ix=ids==j;coord=np.c_[xy[ix],at[ix]];values=item['solver'].evaluate(coord,item['parameter'],c.interval,name)+sigma*rng.normal(size=ix.sum())
                                rows.extend({'field':name,'x':float(x),'y':float(y),'time':float(t),'value':float(v),'sigma':sigma,'sampling':provenance} for (x,y,t),v in zip(coord,values))
                            samples,diag=bundle.sample(problem(grid,family,item['parameter'] if known else None),rows,q,steps,particles,seed);quantities={}
                            for j,name in enumerate(get(family).state):
                                a=samples if j==0 else bundle.velocity_samples;truth=item['solver'].evaluate(q.numpy(),item['parameter'],c.interval,name);shape=(len(grid.times),*grid.shape);pred=a.mean(0).reshape(shape);ref=truth.reshape(shape)
                                quantities[name]={**error_summary(a,truth),**temporal_errors(pred,ref,grid.times,c.interval),'phase_amplitude':spectral_diagnostics(pred,ref),'field_metrics':trajectory_metrics(pred,ref,grid.times,c.lengths,c.field_loss,bundle.cp['normalizations'][family][j]['scale'])}
                                if len(previews)<8:previews.append(np.stack([ref,pred,np.abs(pred-ref)]))
                            records.append({'id':item['id'],'family':family,'case':item['case'],'parameter_status':'known' if known else 'unknown','quantities':quantities,'parameter_absolute_errors':[abs(x-item['parameter']) for x in diag['inferred_parameters'][parameter]],'sensor_consistency':diag['observation_consistency'],'physics':diag['physics'],'seconds':diag['seconds'],'placement':provenance})
    report={'checkpoint_sha256':file_digest(checkpoint),'split':split,'records':records,'seconds':time.perf_counter()-started,'settings':{'query_shape':grid.shape,'times':grid.times,'counts':counts,'noises':noises,'seeds':seeds,'steps':steps},'meaning':'Known AND unknown typed scalar modes; only explicit u/v rows condition generation. Reference initial fields/velocity/unknown scalar are evaluation-only. Same shared checkpoint. No accuracy or identifiability guarantee.'}
    write_json(output,report);atomic_write(Path(output).with_suffix('.npz'),lambda f:np.savez_compressed(f,preview=np.asarray(previews)));return report
