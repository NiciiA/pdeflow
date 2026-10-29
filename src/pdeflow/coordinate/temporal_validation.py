"""Bounded continuous-time diagnostics, including uniform and time-stratum errors."""
from pathlib import Path
import time
import numpy as np
import torch
from pdeflow.runtime import write_json,atomic_write,file_digest,RuntimeSpec
from pdeflow.core import GridSpec,DomainSpec
from pdeflow.equations import heat_problem
from .training import read_checkpoint,normalized_parameter
from .dataset import Dataset
from .contracts import grid_queries
from .model import Autoencoder,decode_chunks
from .evaluation import error_summary
from pdeflow.field_losses import trajectory_metrics,aggregate_metrics,grouped_queries,AnalysisCache,settings,differentiable_loss


def evaluation_grid(c,resolution=16):
    fractions=sorted(set([*np.linspace(0,1,11),.001,.01,.037,.237]))
    times=tuple(float(c.interval[0]+(c.interval[1]-c.interval[0])*t) for t in fractions)
    loss=settings(c.field_loss)
    shape=tuple(max(min(resolution,n),loss.analysis_resolution) if loss.estimator=='grouped_stratified_v1' else min(resolution,n) for n in c.reference_shape)
    return GridSpec(shape,DomainSpec(c.lengths,c.origins),times)


def temporal_errors(prediction,reference,times,interval):
    a=np.asarray(prediction).reshape(len(times),-1);b=np.asarray(reference).reshape(a.shape);mse=np.mean((a-b)**2,axis=1)
    f=(np.asarray(times)-interval[0])/(interval[1]-interval[0]);uniform=np.isclose(f[:,None],np.linspace(0,1,11)[None,:],atol=1e-10).any(1)
    mean_error=(a-a.mean(1,keepdims=True))-(b-b.mean(1,keepdims=True))
    groups={'initial':f==0,'early':(f>0)&(f<=.1),'intermediate':(f>.1)&(f<=.6),'late':(f>.6)&(f<1),'final':f==1}
    return {'rmse_per_physical_time':np.sqrt(mse).tolist(),'physical_times':list(times),
      'uniform_space_time_mse':float(np.trapezoid(mse[uniform],f[uniform])),
      'uniform_metric_definition':'Trapezoidal quadrature on11 equally spaced physical times, spatial node-mean squared error; includes endpoint half weights',
      'time_strata_rmse':{k:float(np.sqrt(mse[v].mean())) for k,v in groups.items()},
      'mean_subtracted_rmse':float(np.sqrt(np.square(mean_error).mean())),
      'absolute_rmse':float(np.sqrt(mse.mean()))}


def representation(checkpoint,output,device='cpu',split='validation',subset=8,allow_partial=False):
    cp,c=read_checkpoint(checkpoint,'autoencoder')
    if not (cp['completed'] or allow_partial) or split not in ('validation','test'):raise ValueError('Choose completed representation and a held-out split')
    if Path(output).exists():raise FileExistsError('Choose a fresh validation report')
    data=Dataset(c,Path(checkpoint).parent,False);norm=cp['normalization'];dev,_=RuntimeSpec(device=device).resolve()
    ae=Autoencoder(c).to(dev).eval().requires_grad_(False);ae.load_state_dict(cp['autoencoder']);grid=evaluation_grid(c);q=grid_queries(grid);rows=[];previews=[];started=time.perf_counter()
    from .temporal import points
    weighted=points(c,c.query_points,torch.Generator().manual_seed(c.seed+831),'query')
    with torch.no_grad():
        for i in range(min(subset,c.validation_count if split=='validation' else c.test_count)):
            item=data.get(split,i);parameter=normalized_parameter(c,torch.tensor([item['D']],device=dev));cq=torch.tensor(item['context'],device=dev)[None];cv=(torch.tensor(item['context_values'],device=dev)[None]-norm['mean'])/norm['scale']
            z=ae.encode(cq,cv,parameter);pred=(decode_chunks(ae,z,q.to(dev),parameter)*norm['scale']+norm['mean']).cpu().numpy()[0];truth=item['solver'].evaluate(q.numpy(),item['D'],c.interval)
            a=pred.reshape(len(grid.times),*grid.shape);b=truth.reshape(a.shape);metrics=temporal_errors(a,b,grid.times,c.interval)
            wa=(decode_chunks(ae,z,weighted.to(dev),parameter)*norm['scale']+norm['mean']).cpu().numpy()[0];wb=item['solver'].evaluate(weighted.numpy(),item['D'],c.interval)
            sa=np.abs(np.fft.rfftn(a,axes=(1,2)))**2;sb=np.abs(np.fft.rfftn(b,axes=(1,2)))**2
            field_report=trajectory_metrics(a,b,grid.times,c.lengths,c.field_loss,norm['scale'])
            if settings(c.field_loss).estimator=='grouped_stratified_v1':
                gq,gt,gw,_=grouped_queries(c,[item],torch.Generator().manual_seed(c.seed+832+i),norm,AnalysisCache())
                gp=decode_chunks(ae,z,gq[0].to(dev),parameter)
                training_objective=float(differentiable_loss(gp,(gt.to(dev)-norm['mean'])/norm['scale'],gw.to(dev),settings(c.field_loss).type))
            else:training_objective=float(np.mean((wa-wb)**2)/norm['scale']**2)
            baseline=temporal_errors(np.full_like(b,norm['mean']),b,grid.times,c.interval)['uniform_space_time_mse']
            rows.append({'id':item['id'],'family':item['family'],**error_summary(pred[None],truth),**metrics,'initial_rmse':metrics['time_strata_rmse']['initial'],
                         'field_metrics':field_report,'training_weighted_objective_normalized':training_objective,'training_objective_definition':'Recorded loss on fixed held-out queries from the training physical-time distribution, global training normalization; distinct from uniform physical validation','training_weighted_mse':float(np.mean((wa-wb)**2)),'spectral_power_relative_l2':float(np.linalg.norm(sa-sb)/max(np.linalg.norm(sb),1e-12)),'constant_baseline_mse':baseline})
            if len(previews)<6:previews.append(np.stack([b,a,np.abs(a-b)]))
    mse=float(np.mean([r['uniform_space_time_mse'] for r in rows]));baseline=float(np.mean([r['constant_baseline_mse'] for r in rows]))
    result={'stage':'autoencoder','checkpoint_sha256':file_digest(checkpoint),'split':split,'records':rows,'constant_baseline_mse':baseline,'validation_mse':mse,'mse_ratio':mse/baseline if baseline>1e-15 else None,
            'field_metrics':aggregate_metrics([r['field_metrics'] for r in rows]),'field_loss':__import__('dataclasses').asdict(settings(c.field_loss)),'dataset_identity':data.metadata['sha256'],'seconds':time.perf_counter()-started,'settings':{'query_shape':grid.shape,'physical_times':grid.times,'subset':len(rows),'training_weighted_query_count':c.query_points},'meaning':'Uniform space-time gate; time strata and deliberately training-weighted objective separate; no test tuning'}
    write_json(output,result);atomic_write(Path(output).with_suffix('.npz'),lambda f:np.savez_compressed(f,preview=previews));return result


def flow(checkpoint,output,device='cpu',subset=6,split='validation',counts=None,noises=None,particles=2,steps=4,resolution=16,allow_partial=False,placements=None,seeds=None):
    from .inference import CoordinateBundle
    from pdeflow.heat_placement import place
    if split not in ('validation','test'):raise ValueError('Choose a held-out split')
    bundle=CoordinateBundle(checkpoint,device,allow_partial);c=bundle.config;data=Dataset(c,(Path(checkpoint).parent/bundle.cp['dependency']['path']).resolve().parent,False)
    grid=evaluation_grid(c,resolution);q=grid_queries(grid);records=[];preview=None;started=time.perf_counter()
    counts=counts or sorted(set([max(1,c.observation_counts[0]),min(16,c.observation_counts[1]),c.observation_counts[1]]));noises=noises if noises is not None else sorted(set([0.,c.noise_range[1]]));placements=placements or ['random'];seeds=seeds or [91]
    design=.237*c.interval[1]
    for i in range(min(subset,c.validation_count if split=='validation' else c.test_count)):
        item=data.get(split,i);truth=item['solver'].evaluate(q.numpy(),item['D'],c.interval);problem=heat_problem(grid,item['D'])
        for count in counts:
            for sigma in noises:
                for policy in placements:
                    for seed in seeds:
                        xy,provenance=place(count,policy,seed,c.lengths,c.origins,field=item['solver'].frame(design,item['D'],c.interval,shape=(32,32)) if policy=='heat' else None)
                        coordinates=np.c_[xy,np.full(count,design)];clean=item['solver'].evaluate(coordinates,item['D'],c.interval);noisy=clean+sigma*np.random.default_rng(seed+1000).normal(size=count)
                        rows=[{'field':'u','x':float(x),'y':float(y),'time':design,'value':float(v),'sigma':sigma} for (x,y),v in zip(xy,noisy)]
                        samples,diag=bundle.sample(problem,rows,q,steps=steps,particles=particles,seed=seed,chunk_size=512)
                        records.append({'id':item['id'],'family':item['family'],'count':count,'sigma':sigma,'placement':provenance,'seed':seed,**error_summary(samples,truth),**temporal_errors(samples.mean(0),truth,grid.times,c.interval),
                                        'field_metrics':trajectory_metrics(samples.mean(0).reshape(len(grid.times),*grid.shape),truth.reshape(len(grid.times),*grid.shape),grid.times,c.lengths,c.field_loss,bundle.cp['normalization']['scale']),'sensor_consistency':diag['observation_consistency'],'physics':diag['physics']})
                        if preview is None:
                            shape=(len(grid.times),*grid.shape);preview=np.stack([truth.reshape(shape),samples.mean(0).reshape(shape),np.abs(samples.mean(0)-truth).reshape(shape)])
    result={'checkpoint_sha256':file_digest(checkpoint),'split':split,'records':records,'settings':{'subset':subset,'counts':counts,'noises':noises,'particles':particles,'heun_steps':steps,'query_shape':grid.shape,'physical_times':grid.times,'sensor_design_time':design,'placements':placements,'seeds':seeds},'seconds':time.perf_counter()-started,
            'meaning':'Whole-domain matched validation; reference-assisted positions convey information. Individual and ensemble errors separate; no scientific winner or calibrated uncertainty claimed'}
    write_json(output,result);atomic_write(Path(output).with_suffix('.npz'),lambda f:np.savez_compressed(f,preview=preview));return result


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cpu');p.add_argument('--subset',type=int,default=2);p.add_argument('--count',type=int,default=25);p.add_argument('--noise',type=float,default=.01);p.add_argument('--seeds',type=int,nargs='+',default=[91,92,93]);a=p.parse_args()
    flow(a.checkpoint,a.output,a.device,subset=a.subset,counts=[a.count],noises=[a.noise],placements=['random','even','heat'],seeds=a.seeds)
if __name__=='__main__':main()
