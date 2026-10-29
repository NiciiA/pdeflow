"""Analytic Gaussian and mixture flows; no neural-training approximation."""
from dataclasses import replace
from pathlib import Path
import math,time
import numpy as np
import torch
from pdeflow.fk import FKConfig,simulate
from pdeflow.runtime import write_json,RuntimeSpec


def gaussian_velocity(mean,variance):
    def velocity(x,s):
        m=x.new_tensor(mean);v=x.new_tensor(variance);q=(1-s)**2+s*s*v
        return m+(s*v-(1-s))/q*(x-s*m)
    return velocity


def gaussian_energy(y,variance):
    return lambda x:{'observation':.5*((x-x.new_tensor(y)).square()/x.new_tensor(variance)).sum(-1)}


def gaussian_target(mean,variance,y,noise_variance):
    m=np.asarray(mean);v=np.asarray(variance);r=np.asarray(noise_variance);cov=1/(1/v+1/r)
    return cov*(m/v+np.asarray(y)/r),np.diag(cov)


def mixture_velocity(x,s):
    # Component-wise exact independent-Gaussian-source linear CFM, equal mixture.
    means=x.new_tensor([[-2.,0.],[2.,0.]])
    q=(1-s)**2+s*s*.36
    responsibilities=(-.5*(x[...,None,:]-s*means).square().sum(-1)/q).softmax(-1)
    m=(responsibilities[...,None]*means).sum(-2)
    return m+(s*.36-(1-s))/q*(x-s*m)


def mixture_target(y=(.5,.2),r=(.9,.8)):
    means=np.array([[-2.,0.],[2.,0.]]);y=np.array(y);r=np.array(r);c=1/(1/.36+1/r)
    posterior=c*(means/.36+y/r)
    logs=-.5*((y-means)**2/(r+.36)).sum(1);w=np.exp(logs-logs.max());w/=w.sum()
    m=(w[:,None]*posterior).sum(0);diff=posterior-m
    cov=np.diag(c)+np.einsum('k,ki,kj->ij',w,diff,diff)
    occupancy=sum(w[k]*.5*(1+math.erf(posterior[k,0]/math.sqrt(2*c[0]))) for k in range(2))
    return m,cov,float(occupancy),w,posterior,c


def moments(run):
    x=run.state.components['model'].cpu().double();w=run.state.log_weights.exp();m=(x*w[...,None]).sum(1);d=x-m[:,None]
    cov=torch.einsum('bp,bpi,bpj->bij',w,d,d)
    return m,cov


def plot_summary(root,kind,summaries):
    """Keep step count separate from population size in convergence figures."""
    import os
    os.environ.setdefault('MPLCONFIGDIR',str(root/'.matplotlib'))
    os.environ.setdefault('XDG_CACHE_HOME',str(root/'.cache'))
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    counts=sorted({row['steps'] for row in summaries})
    fig,axes=plt.subplots(1,len(counts),figsize=(6*len(counts),4),squeeze=False)
    for ax,n in zip(axes[0],counts):
        for method in ['base','importance','fk','no_resampling']:
            rows=sorted([r for r in summaries if r['method']==method and r['steps']==n],key=lambda r:r['particles'])
            ax.errorbar([r['particles'] for r in rows],[r['mean_error'] for r in rows],
                        yerr=[r['mean_error_std'] for r in rows],marker='o',capsize=3,label=method)
        ax.set(xlabel='Particles',ylabel='Mean error (error bars: seed standard deviation)',title=f'{kind}: {n} EM steps')
        ax.legend()
    fig.suptitle('Base compares to base law; tilted methods compare to analytic target')
    fig.tight_layout();fig.savefig(root/'mean-errors.png');plt.close(fig)


def benchmark(output,kind='gaussian',device='cpu',quick=False):
    root=Path(output)
    if root.exists():raise FileExistsError('Choose a fresh toy output')
    root.mkdir(parents=True);runtime=RuntimeSpec(device=device);dev,dtype=runtime.resolve()
    dtype=torch.float32 if dev.type=='mps' else torch.float64
    velocity=gaussian_velocity([.7,-.4],[1.44,.64]) if kind=='gaussian' else mixture_velocity
    target=gaussian_target([.7,-.4],[1.44,.64],[1.4,.3],[.25,.49]) if kind=='gaussian' else mixture_target()[:2]
    energy=gaussian_energy([1.4,.3],[.25,.49]) if kind=='gaussian' else gaussian_energy([.5,.2],[.9,.8])
    budgets=[(512,16),(4096,16),(512,128),(4096,128)] if not quick else [(512,32)]
    records=[];cfg=FKConfig(kappa=.4,chunk_size=8192,ess_fraction=.7,resample_every=4)
    # Low-dimensional analytic callbacks can process a larger chunk than local CNNs.
    cfg=replace(cfg,chunk_size=256,max_particle_evaluations=1048576,record_particle_history=False)
    for particles,steps in budgets:
        for seed in [11,23,37,53]:
            for method in ['base','importance','fk','no_resampling']:
                start=time.perf_counter();run=simulate(velocity,energy,(1,particles,2),steps,seed,cfg,device=dev,dtype=dtype,method=method)
                m,c= moments(run);mu=np.asarray(target[0]);cov=np.asarray(target[1])
                if method=='base':mu=np.array([.7,-.4]) if kind=='gaussian' else np.zeros(2);cov=np.diag([1.44,.64]) if kind=='gaussian' else np.diag([4.36,.36])
                x=run.state.components['model'][0].cpu().double();w=run.state.log_weights[0].exp()
                record={'method':method,'seed':seed,'particles':particles,'steps':steps,'mean':m[0].tolist(),'covariance':c[0].tolist(),'mean_error':float(np.linalg.norm(m[0].numpy()-mu)),'covariance_error':float(np.linalg.norm(c[0].numpy()-cov)),'positive_mode_occupancy':float(w[x[:,0]>0].sum()),'target_positive_occupancy':mixture_target()[2] if kind=='mixture' and method!='base' else None,'ess':float(1/w.square().sum()),'unique_ancestors':len(torch.unique(run.state.ancestors[0])),'seconds':time.perf_counter()-start,'cost':run.diagnostics}
                records.append(record)
    summaries=[]
    for p,n in budgets:
        for method in ['base','importance','fk','no_resampling']:
            rows=[r for r in records if r['particles']==p and r['steps']==n and r['method']==method]
            summaries.append({'particles':p,'steps':n,'method':method,'positive_mode_occupancy':float(np.mean([r['positive_mode_occupancy'] for r in rows])),'resampling_events':float(np.mean([r['cost']['resampling_events'] for r in rows])),**{name:float(np.mean([r[name] for r in rows])) for name in ['mean_error','covariance_error','ess','unique_ancestors','seconds']},'mean_error_std':float(np.std([r['mean_error'] for r in rows]))})
    result={'kind':kind,'device':str(dev),'dtype':str(dtype),'analytic_target_mean':np.asarray(target[0]).tolist(),'analytic_target_covariance':np.asarray(target[1]).tolist(),'records':records,'summary':summaries,'learned_model_error':False,'seeds':[11,23,37,53]}
    write_json(root/'results.json',result)
    plot_summary(root,kind,summaries)
    return result


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--kind',choices=['gaussian','mixture'],default='gaussian');parser.add_argument('--device',default='cpu');parser.add_argument('--quick',action='store_true');a=parser.parse_args()
    r=benchmark(a.output,a.kind,a.device,a.quick);print(r['summary'])
if __name__=='__main__':main()
