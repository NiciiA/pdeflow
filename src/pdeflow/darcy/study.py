"""Bounded online learning ladder with validation-only tuning and fixed sensors."""
import argparse
from dataclasses import replace
from pathlib import Path
import shutil,time
import numpy as np
import torch
from pdeflow.core import ProblemSpec
from pdeflow.data.providers import prepare,load_sample
from pdeflow.data.physical import seed_for
from pdeflow.darcy.inverse import inverse_baseline
from pdeflow.evaluation import metrics,export_result
from pdeflow.observations import sensor_observations,ObservationSet,PointOperator
from pdeflow.runtime import Config,RuntimeSpec,read_json,write_json
from pdeflow.training import train,load_inference
from pdeflow.sampling import sample


def cases(c,root):
    all_cases={};manifest={}
    for split in ('validation','test'):
        all_cases[split]=[];manifest[split]=[]
        for i in range(c.provider.validation_count if split=='validation' else c.provider.test_count):
            physical=load_sample(root/'evaluation'/split/str(i),c)
            if np.sqrt(np.sum(np.asarray(physical.problem.known_fields['f'])**2))<1e-8:raise ValueError('Principal inverse benchmark cannot use zero forcing')
            for fields in ('pressure','both'):
                for density in (4,12):
                    key=f'{split}-{i}-{fields}-{density}';seed=seed_for(c,split,i,stream=f'benchmark-observations:{fields}:{density}')%(2**63)
                    obs=sensor_observations(physical.values[None],c.grid,density,seed,.005,.02)
                    if fields=='pressure':
                        select=obs.operator.indices[:,0]==1
                        obs=ObservationSet(PointOperator(c.grid,obs.operator.indices[select],2),obs.values[:,select],obs.mask[:,select],.005,.02)
                    record={'id':key,'problem_id':physical.specification['id'],'fields':fields,'spatial_sensors':density,'observations':obs.to_dict(),'observation_seed':seed}
                    manifest[split].append(record);all_cases[split].append((key,physical.problem,obs,physical.values[None]))
    write_json(root/'benchmark-manifest.json',manifest);return all_cases


def score(report):return float(np.mean([report['fields'][field]['relative_l2_unobserved'] for field in ('a','log_a','u')]))

def summarized(reports):
    result={'count':len(reports),'fields':{},'failures':0}
    for field in ('a','log_a','u'):result['fields'][field]=float(np.mean([r['fields'][field]['relative_l2_unobserved'] for r in reports]))
    for metric in ('observation_rmse','pde_fd_rmse','pressure_gauge_max_abs_mean','ensemble_rms_spread'):
        result[metric]=float(np.mean([r[metric] for r in reports]))
    result['seconds']=float(sum(r['cost']['seconds'] for r in reports));result['model_calls']=sum(r['cost'].get('model_evaluations',0) for r in reports)
    result['forward_solves']=sum(r['cost'].get('forward_solves',0) for r in reports);result['adjoint_solves']=sum(r['cost'].get('adjoint_solves',0) for r in reports)
    result['positivity_violations']=sum(r['positivity_violations'] for r in reports)
    return result


def run(config,output):
    root=Path(output)
    if root.exists():raise FileExistsError('Choose a fresh Darcy study directory')
    root.mkdir(parents=True);start=time.perf_counter();budgets=(0,20,80)
    c=replace(config,runtime=RuntimeSpec(device='cpu'),training=replace(config.training,steps=80,checkpoint_every=20,batch_size=2),inference=replace(config.inference,steps=6,particles=2))
    write_json(root/'config.json',c.to_dict());prepare(c,root/'evaluation');benchmark=cases(c,root)
    training=[]
    for budget in budgets:
        result=train(c,None,root/'training',resume=root/'training/checkpoint.pt' if budget else None,stop_after=budget,evaluation_path=root/'evaluation');result.pop('history')
        target=root/f'budget-{budget}';target.mkdir();shutil.copy2(root/'training/inference.pt',target/'inference.pt')
        training.append({'budget':budget,'cost':result['cost'],'provider':read_json(root/'training/data-accounting.json')})
    final=load_inference(root/'budget-80/inference.pt',c.runtime);guidance=[]
    for observation,physics in ((.1,.0001),(.5,.0001),(.1,.01),(.5,.01)):
        inf=replace(c.inference,observation_strength=observation,physics_strength=physics);reports=[]
        for key,p,obs,truth in benchmark['validation']:
            reports.append(metrics(sample(final,p,obs,inf,'observation_pde'),p,obs,truth))
        guidance.append({'observation':observation,'physics':physics,'validation_score':float(np.mean([score(r) for r in reports]))})
    chosen=min(guidance,key=lambda x:x['validation_score']);regularization=[]
    for ridge in (.001,.1):
        reports=[metrics(inverse_baseline(p,obs,ridge=ridge),p,obs,truth) for key,p,obs,truth in benchmark['validation']]
        regularization.append({'ridge':ridge,'validation_score':float(np.mean([score(r) for r in reports]))})
    selected_ridge=min(regularization,key=lambda x:x['validation_score'])['ridge']
    write_json(root/'validation-selection.json',{'guidance_trials':guidance,'baseline_trials':regularization,'selected_guidance':chosen,'selected_ridge':selected_ridge,'selection_metric':'mean unobserved relative L2 over a, log(a), pressure and fixed validation scenarios; no test selection'})
    baseline_reports=[]
    for key,p,obs,truth in benchmark['test']:
        result=inverse_baseline(p,obs,ridge=selected_ridge);report=metrics(result,p,obs,truth);baseline_reports.append(report);write_json(root/'test-baseline'/f'{key}.json',report)
        if key==benchmark['test'][0][0]:export_result(root/'figures/conventional',result,p,obs,truth)
    summary={}
    inf=replace(c.inference,observation_strength=chosen['observation'],physics_strength=chosen['physics'])
    for budget in budgets:
        bundle=load_inference(root/f'budget-{budget}/inference.pt',c.runtime);summary[str(budget)]={}
        for mode in ('unguided','observation','observation_pde'):
            reports=[]
            for key,p,obs,truth in benchmark['test']:
                result=sample(bundle,p,obs,inf,mode);report=metrics(result,p,obs,truth);reports.append(report);write_json(root/f'budget-{budget}'/mode/f'{key}.json',report)
                if key==benchmark['test'][0][0]:export_result(root/f'figures/budget-{budget}-{mode}',result,p,obs,truth)
            summary[str(budget)][mode]=summarized(reports)
        summary[str(budget)]['darcy_inverse']=summarized(baseline_reports)
    report={'budgets':list(budgets),'training':training,'test':summary,'selection':read_json(root/'validation-selection.json'),'seconds':time.perf_counter()-start,'independent_validation_problems':c.provider.validation_count,'independent_test_problems':c.provider.test_count,'scenarios_per_problem':4,'observation_noise_std':.005,'training_path':'one online training trajectory with fixed prior, source streams and calibration','budget_selection':'No winning budget selected; report all test points','scientific_claim':'bounded one-seed pilot; single-forcing restricted-prior identifiability only; scenarios share physical problems'}
    write_json(root/'study.json',report)
    import os
    os.environ.setdefault('MPLCONFIGDIR',str(root/'.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,3.5),constrained_layout=True)
    for ax,field in zip(axes,('a','log_a','u')):
        for method in ('unguided','observation','observation_pde','darcy_inverse'):ax.plot(budgets,[summary[str(b)][method]['fields'][field] for b in budgets],marker='o',label=method)
        ax.set_title(field);ax.set_xlabel('Optimizer updates');ax.set_ylabel('Unobserved relative L2');ax.legend(fontsize=7)
    fig.savefig(root/'budget-comparison.png',dpi=130);plt.close(fig);return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(Config.load(a.config,check_device=False),a.output)
