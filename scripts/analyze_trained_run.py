"""Bounded real-weight audit. No optimizer/training, no artifact mutation.

Run with the checkpoint's verified source on PYTHONPATH when science differs.
Outputs are append-only phases; an existing phase is checked/skipped, not replaced.
"""
from pathlib import Path
import argparse,json,time,hashlib,shutil
import numpy as np
import torch
from pdeflow.runtime import write_json,read_json,atomic_write,file_digest,synchronize
from pdeflow.coordinate.training import read_checkpoint,normalized_parameter
from pdeflow.coordinate.optimization import scientific_source
from pdeflow.coordinate.model import Autoencoder
from pdeflow.coordinate.dataset import example,state_samples,identity
from pdeflow.coordinate.physics import get,family_index,invariant_drift
from pdeflow.coordinate.evolution_monitor import evaluate as evolution_evaluate
from pdeflow.coordinate.diagnostic_metrics import field_metrics
from pdeflow.coordinate.evaluation import spectral_diagnostics


def oracles(run,root,device):
    destination=root/'oracles'
    if (destination/'metrics.json').exists():return
    destination.mkdir(exist_ok=True)
    cp,c=read_checkpoint(run/'evolution/evolution.pt','evolution')
    if cp['scientific_source']['sha256']!=scientific_source()['sha256']:raise ValueError('Use exact saved scientific source for oracle evaluation')
    assert cp['dataset_identity']==identity(c)
    model=Autoencoder(c).to(device).eval().requires_grad_(False);model.load_state_dict(cp['autoencoder'],strict=True)
    # Fixed coverage prefix, plus seeded independent validation examples beyond gate prefix.
    indices=[*range(12),*sorted(np.random.default_rng(20261101).choice(np.arange(24,c.validation_count),8,replace=False).tolist())]
    times=sorted(set([0.,.001,.01,.05,.1,.237,.5,.75,1.,*np.random.default_rng(3281).uniform(0,1,6)]))
    times=[float(t*c.interval[1]) for t in times];records=[]
    for index in indices:
        target=destination/f'case-{index}.json'
        if target.exists():records.append(read_json(target));continue
        item=example(c,'validation',index);norms=cp['normalizations'][item['family']]
        report,images=evolution_evaluate(c,model,cp['normalizations'],[item],times,32,True,index in (0,3,8,9))
        row=report['records'][0];frames={};quantities={}
        for j,name in enumerate(get(item['family']).state):
            subset=[x for x in images if x['quantity']==name]
            truth=np.stack([x['reference'] for x in subset]);pred=np.stack([x['prediction'] for x in subset]);held=np.repeat(pred[:1],len(times),0)
            quantities[name]={'a1_initial':field_metrics(pred[None,:1],truth[:1],[0.],c,name,norms[j]),'a2':field_metrics(pred[None],truth,times,c,name,norms[j]),'held_initial':field_metrics(held[None],truth,times,c,name,norms[j]),'phase_amplitude':spectral_diagnostics(pred,truth)}
            frames.update({name+'_reference':truth,name+'_a2':pred,name+'_held_initial':held})
            # Independent mean/nonconstant mode diagnostics and periodic seam at exact endpoints.
            f=torch.tensor([family_index(item['family'])],device=device);pn=normalized_parameter(c,torch.tensor([item['parameter']],device=device),f);known=torch.ones(1,dtype=torch.bool,device=device)
            with torch.no_grad():
                z=model.encode(torch.tensor(item['context'],device=device)[None],state_samples(item,item['context'],0.,c,norms).to(device)[None],pn,f)
                q=torch.tensor([[c.origins[0],c.origins[1]+.31*c.lengths[1],.237*c.interval[1]],[c.origins[0]+c.lengths[0],c.origins[1]+.31*c.lengths[1],.237*c.interval[1]],[c.origins[0]+.31*c.lengths[0],c.origins[1],.237*c.interval[1]],[c.origins[0]+.31*c.lengths[0],c.origins[1]+c.lengths[1],.237*c.interval[1]]],device=device)
                values=model.decode(z,q,pn,known,name,f)[0]
            quantities[name]['periodic_endpoint_difference_normalized']=[float(abs(values[0]-values[1])),float(abs(values[2]-values[3]))]
            quantities[name]['nonconstant_l2_per_time']={'reference':np.sqrt(((truth-truth.mean((-2,-1),keepdims=True))**2).mean((-2,-1))).tolist(),'a2':np.sqrt(((pred-pred.mean((-2,-1),keepdims=True))**2).mean((-2,-1))).tolist()}
        record={'index':index,'id':item['id'],'family':item['family'],'case':item['case'],'parameter':item['parameter'],'stage_updates':cp['progress'],'checkpoint_sha256':file_digest(run/'evolution/evolution.pt'),'oracle':row,'quantities':quantities,'times':times,'grid':[32,32]}
        atomic_write(destination/f'case-{index}.npz',lambda stream:np.savez_compressed(stream,**frames));write_json(target,record);records.append(record)
        print('ORACLE',index,item['family'],item['case'],'ratio',report['ratio_to_held_initial'],flush=True)
    write_json(destination/'metrics.json',{'records':records,'indices':indices,'times':times,'split':'validation; fixed coverage prefix overlaps recorded gate; eight independently seeded extra validation IDs; untouched test split unused','device':str(device),'source':scientific_source(),'aggregation':'Per quantity physical grid means/trapezoidal time quadrature; no initial AE error floor asserted'})


def curves(run,root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=root/'curves';output.mkdir(exist_ok=True);summaries={}
    for stage in ('autoencoder','evolution','flow'):
        checkpoint_name={'autoencoder':'autoencoder.pt','evolution':'evolution.pt','flow':'coordinate.pt'}[stage];checkpoint_id=file_digest(run/stage/checkpoint_name)[:12]
        history=read_json(run/stage/'history.json');steps=np.array([x['step'] for x in history]);loss=np.array([x['training_loss'] for x in history]);v=[x for x in history if x.get('validation_loss') is not None]
        fig,axes=plt.subplots(3,1,figsize=(11,10),layout='constrained');axes[0].plot(steps,loss,alpha=.18,label='raw weighted training objective')
        if len(loss)>=100:axes[0].plot(steps[99:],np.convolve(loss,np.ones(100)/100,'valid'),label='100 attempted-update moving average')
        axes[0].plot([x['step'] for x in v],[x['validation_loss'] for x in v],'o-',label='fixed weighted validation');axes[0].legend();axes[0].set_yscale('log');axes[0].set_ylabel('Stage-specific objective')
        axes[1].plot(steps,[x['learning_rate'] for x in history]);axes[1].set_ylabel('Learning rate')
        keys=sorted({k for x in history for k in x.get('components',{}) if k.startswith('objective_') or k in ('flow_matching','observation_endpoint_loss')});keys=[k for k in keys if any(abs(y.get('components',{}).get(k,0))>0 for y in history)]
        for key in keys:
            values=np.array([x.get('components',{}).get(key,np.nan) for x in history]);axes[2].plot(steps[99:],np.convolve(values,np.ones(100)/100,'valid'),label=(key+' (recorded LAST microbatch)' if stage=='flow' and key=='flow_matching' else key))
        if stage=='flow':axes[2].plot(steps[99:],np.convolve(loss,np.ones(100)/100,'valid'),label='Actual full-batch FM (all decoded auxiliaries disabled)')
        if keys:axes[2].legend(fontsize=8)
        axes[2].set_ylabel('Active objective component (100-row mean)');axes[2].set_xlabel('Successful optimizer update (repeats mark skipped attempts)')
        fig.suptitle(f'Run {read_json(run/"session.json")["name"]} · {stage} · checkpoint {checkpoint_id} · latest {history[-1]["step"]}; mixed families/times, stage objectives differ');fig.savefig(output/f'{stage}.png',dpi=140);plt.close(fig)
        families=sorted({k for x in history for k in x.get('per_family_loss',{})});fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
        for f in families:
            values=np.array([x.get('per_family_loss',{}).get(f,np.nan) for x in history]);ax.plot(steps[99:],np.convolve(values,np.ones(100)/100,'valid'),label=f)
        ax.legend();ax.set_title(f'{stage}: recorded family training objectives, run {read_json(run/"session.json")["name"]} · checkpoint {checkpoint_id}');fig.savefig(output/f'{stage}-families.png',dpi=140);plt.close(fig)
        summaries[stage]={'rows':len(history),'first100_mean':float(loss[:100].mean()),'last100_mean':float(loss[-100:].mean()),'first_validation':v[0]['validation_loss'],'last_validation':v[-1]['validation_loss'],'warmup_fraction':(read_json(root/'weight-inventory.json')['stages'][stage]['scheduler']['settings']['warmup']/2500),'last100_family_means':{f:float(np.mean([x['per_family_loss'][f] for x in history[-100:]])) for f in families},'component_keys':keys,'quantity_breakdown':'Only quantities/time fractions recorded in components; no invented per-quantity training loss history'}
    write_json(output/'summary.json',summaries)


def cases_and_b(run,root,device):
    from pdeflow.coordinate.diagnostic_cases import capture,load_case
    from pdeflow.coordinate.diagnostics import diagnose
    checkpoint=run/'flow/coordinate.pt';cases=root/'cases';cases.mkdir(exist_ok=True);reports=root/'reconstruction';reports.mkdir(exist_ok=True)
    # New run: do not splice old latent noise of a different layout into this model.
    for label,index in [('validation-cosine',0),('validation-signed',8),('validation-travelling',3),('validation-velocity',7)]:
        folder=cases/label
        if not folder.exists():capture(checkpoint,folder,validation_index=index,training_observations=True,seed=91,steps=8,particles=4,resolution=32)
    origins={}
    for label in ('periodic','hotspot','wave'):
        old=Path('outputs/reconstruction-diagnosis-20261101/cases')/label
        old_case=read_json(old/'case.json');ref=Path('outputs/local/workbench/data')/old_case['reference_label']['reference_id'];folder=cases/('saved-'+label)
        if not folder.exists():capture(checkpoint,folder,reference=ref,observations=old_case['observations'],problem=__import__('pdeflow.core',fromlist=['ProblemSpec']).ProblemSpec.from_dict(old_case['problem']),seed=old_case['inference']['seed'],steps=8,particles=4,resolution=32)
        with np.load(old/'noise.npz',allow_pickle=False) as a:old_shape=list(a['initial_flow_noise'].shape)
        with np.load(folder/'noise.npz',allow_pickle=False) as a:new_shape=list(a['initial_flow_noise'].shape)
        origins[label]={'original_case_identity':old_case['case_identity'],'original_checkpoint_sha256':old_case['checkpoint_sha256'],'original_noise_shape':old_shape,'new_noise_shape':new_shape,'observations_exactly_reused':read_json(folder/'case.json')['observations']==old_case['observations'],'bitwise_old_replay_compatible':old_shape==new_shape,'comparison':'Same physical problem/reference/measurements. Old latent shape differs, so new explicitly recorded seed91 Gaussian noise; no old source/checkpoint/case hash rewritten.'}
    write_json(root/'saved-case-provenance.json',origins)
    for folder in sorted(cases.iterdir()):
        if not (folder/'case.json').exists() or (reports/folder.name/'diagnosis.json').exists():continue
        print('B START',folder.name,flush=True)
        result=diagnose(checkpoint,folder,reports/folder.name,str(device),('a1','a2','b'),run/'evolution/evolution.pt',run/'autoencoder/autoencoder.pt',False,())
        print('B DONE',folder.name,'seconds',result['seconds'],flush=True)


def ablations(run,root,device):
    from pdeflow.coordinate.inference import CoordinateBundle
    from pdeflow.coordinate.diagnostic_cases import load_case,sensor_truth
    from pdeflow.coordinate.diagnostics import observation_ablations,integration_probe,exact_sensors,canonical_known
    from pdeflow.coordinate.diagnostic_metrics import sensor_metrics
    from pdeflow.core import GridSpec
    from pdeflow.coordinate.contracts import grid_queries
    from dataclasses import replace
    destination=root/'ablations';destination.mkdir(exist_ok=True)
    bundle=CoordinateBundle(run/'flow/coordinate.pt',str(device));c=bundle.config
    for label in ('validation-cosine','validation-travelling','validation-velocity'):
        path=destination/(label+'.json')
        if path.exists():continue
        case,problem,solver,context,noise=load_case(root/'cases'/label);original=dict(case['inference']);case['inference']={**original,'steps':4}
        print('ABLATION START',label,flush=True)
        report=observation_ablations(bundle,case,problem,noise,solver)
        report['integration']=integration_probe(bundle,case,problem,noise,solver,(4,8,16))
        q=grid_queries(GridSpec((12,12),problem.grid.domain,(0.,.1,.5,1.)));true=case['true_parameter_evaluation_only'];reference=solver.evaluate(q.numpy(),true,c.interval,'u');rows=case['observations'];clean=sensor_truth(solver,rows,true,c.interval);variants={}
        # Known/unknown disclosure, actual new sensor-time collections and nested counts.
        variants['known']=(canonical_known(problem,true),rows)
        variants['unknown']=(replace(problem,known_parameters={},unknown_parameters=(get(problem.family).parameter,)),rows)
        for name,at in [('early_collection',.02*c.interval[1]),('late_collection',.83*c.interval[1])]:
            new=[{**r,'time':at} for r in rows];truth=sensor_truth(solver,new,true,c.interval)
            new=[{**r,'value':float(value+old['value']-oldclean)} for r,value,old,oldclean in zip(new,truth,rows,clean)];variants[name]=(problem,new)
        for n in (8,16,32):
            if n<len(rows):variants['nested_'+str(n)]=(problem,rows[:n])
        values={};base=None
        for name,(p,observations) in variants.items():
            prediction,meta=bundle.sample(p,observations,q,initial_noise=noise,**case['inference']);z=bundle.decoded_latent;v={'latent':z,'field':torch.tensor(prediction)}
            if base is None:base=v
            sensors=exact_sensors(bundle,z.to(device),bundle.parameter.to(device),bundle.known.to(device),bundle.families.to(device),observations);ownclean=sensor_truth(solver,observations,true,c.interval)
            held=exact_sensors(bundle,z.to(device),bundle.parameter.to(device),bundle.known.to(device),bundle.families.to(device),case['heldout_observations_evaluation_only'])
            values[name]={'count':len(observations),'changes_from_known':{k:float((x-base[k]).square().mean().sqrt()) for k,x in v.items()},'conditioning':sensor_metrics(sensors,observations,ownclean),'heldout':sensor_metrics(held,case['heldout_observations_evaluation_only'],[r['value'] for r in case['heldout_observations_evaluation_only']]),'u_field_mse_per_sample':np.mean((prediction-reference)**2,1).tolist(),'u_ensemble_mean_mse':float(np.mean((prediction.mean(0)-reference)**2)),'inferred_parameters':meta['inferred_parameters'],'parameter_truth_evaluation_only':true}
        report.update(physical_controls=values,settings=case['inference'],note='All controls use identical explicitly saved new-model Gaussian noise. New sensor-time collections recompute reference values at exact times and retain the same noise realization. No observation rows are edited in saved cases.')
        write_json(path,report);print('ABLATION DONE',label,flush=True)


def corrections(run,root,device):
    from pdeflow.coordinate.diagnostic_correction import correct
    target=root/'corrections';target.mkdir(exist_ok=True)
    for label in ('saved-periodic','validation-velocity'):
        if (target/label/'correction.json').exists():continue
        print('CORRECTION START',label,flush=True)
        result=correct(run/'flow/coordinate.pt',root/'cases'/label,target/label,str(device),20,.02,.01)
        print('CORRECTION DONE',label,'seconds',result['seconds'],flush=True)


def physics_audit(run,root,device):
    from pdeflow.coordinate.evolution_objective import decoded_residuals
    target=root/'physics';target.mkdir(exist_ok=True)
    cp,c=read_checkpoint(run/'evolution/evolution.pt');ae=Autoencoder(c).to(device).eval().requires_grad_(False);ae.load_state_dict(cp['autoencoder'],strict=True)
    times=np.linspace(*c.interval,129);xy=torch.stack(torch.meshgrid(torch.arange(16,device=device)/16,torch.arange(16,device=device)/16,indexing='ij'),-1).reshape(-1,2);records=[]
    for index in (0,3,7):
        file=target/f'case-{index}.json'
        if file.exists():records.append(read_json(file));continue
        item=example(c,'validation',index);ns=cp['normalizations'][item['family']];f=torch.tensor([family_index(item['family'])],device=device);p=normalized_parameter(c,torch.tensor([item['parameter']],device=device),f);known=torch.ones(1,dtype=torch.bool,device=device)
        with torch.no_grad():z=ae.encode(torch.tensor(item['context'],device=device)[None],state_samples(item,item['context'],0.,c,ns).to(device)[None],p,f)
        fields={name:[] for name in get(item['family']).state};truth={name:[] for name in fields}
        with torch.no_grad():
            for t in times:
                evolved=ae.evolve(z,0.,float(t),p,known,f)
                for j,name in enumerate(fields):fields[name].append((ae.decode_state(evolved,xy,j,f)[0]*ns[j]['scale']+ns[j]['mean']).cpu().numpy().reshape(16,16));truth[name].append(item['solver'].frame(float(t),item['parameter'],c.interval,(16,16),name))
        fields={k:np.asarray(v) for k,v in fields.items()};truth={k:np.asarray(v) for k,v in truth.items()};q=np.array([[.31,.43,t] for t in (.02,.237,.5,.83)]);coords=torch.tensor(q,dtype=torch.float32,device=device)[None]
        auto=decoded_residuals(ae,z,coords,p,known,f,cp['normalizations']);names=['heat'] if item['family']=='heat' else ['wave_u','wave_v'];auto={k:auto[k].detach().cpu().numpy()[0] for k in names};finite=[]
        @torch.no_grad()
        def pred(points,name):
            j=0 if name=='u' else 1
            return ae.decode(z,torch.tensor(points,dtype=torch.float32,device=device),p,known,name,f)[0].cpu().numpy().astype(float)*ns[j]['scale']+ns[j]['mean']
        for h in (.02,.01,.005):
            results={}
            for label,fun in [('learned',pred),('reference',lambda points,name:item['solver'].evaluate(points,item['parameter'],c.interval,name))]:
                u=fun(q,'u');lap=0
                for axis in (0,1):
                    a=q.copy();b=q.copy();a[:,axis]+=h;b[:,axis]-=h;lap+=(fun(a,'u')-2*u+fun(b,'u'))/h**2
                a=q.copy();b=q.copy();a[:,2]+=h;b[:,2]-=h;ut=(fun(a,'u')-fun(b,'u'))/(2*h)
                if item['family']=='heat':res={'heat':(ut-item['parameter']*lap)/(ns[0]['scale']/c.interval[1])}
                else:
                    v=fun(q,'v');vt=(fun(a,'v')-fun(b,'v'))/(2*h);res={'wave_u':(ut-v)/(ns[0]['scale']/c.interval[1]),'wave_v':(vt-item['parameter']**2*lap)/(ns[1]['scale']/c.interval[1])}
                results[label]={k:values.tolist() for k,values in res.items()}
            finite.append({'h_space_and_time':h,**results,'learned_autodiff_vs_fd_rms':{k:float(np.sqrt(np.mean((auto[k]-np.array(results['learned'][k]))**2))) for k in auto}})
        short=[]
        for tb in (.05,.237,.5,.83,1.):
            for ta in (0.,max(0.,tb-.05)):
                with torch.no_grad():
                    za=ae.encode(torch.tensor(item['context'],device=device)[None],state_samples(item,item['context'],ta,c,ns).to(device)[None],p,f);zt=ae.evolve(za,ta,tb,p,known,f)
                    errors={name:float(np.mean(((ae.decode_state(zt,xy,j,f)[0]*ns[j]['scale']+ns[j]['mean']).cpu().numpy().reshape(16,16)-item['solver'].frame(tb,item['parameter'],c.interval,(16,16),name))**2)) for j,name in enumerate(fields)}
                short.append({'ta':ta,'tb':tb,'dt':tb-ta,'physical_mse':errors,'meaning':'Oracle reference state encoded at ta, not inference; compares short with long evolution using true ta state'})
        record={'index':index,'family':item['family'],'case':item['case'],'times':times.tolist(),'coordinates_for_residual':q.tolist(),'autodiff_normalized_residuals':{k:v.tolist() for k,v in auto.items()},'finite_difference_checks':finite,'invariants':invariant_drift(fields['u'],fields.get('v'),times,c.lengths,item['parameter']),'reference_invariants':invariant_drift(truth['u'],truth.get('v'),times,c.lengths,item['parameter']),'paired_evolution':short,'checkpoint_sha256':file_digest(run/'evolution/evolution.pt'),'device':str(device),'finite_difference_limit':'Central physical differences, h decreases; learned FP32 cancellation and O(h²) truncation both present. Reference float64 analytic Fourier values are independent.'}
        atomic_write(target/f'case-{index}.npz',lambda stream:np.savez_compressed(stream,**{k+'_learned':v for k,v in fields.items()},**{k+'_reference':v for k,v in truth.items()},times=times));write_json(file,record);records.append(record);print('PHYSICS DONE',index,flush=True)
    write_json(target/'metrics.json',records)


def interfaces(run,root,device):
    from pdeflow.coordinate.inference import CoordinateBundle
    from pdeflow.coordinate.diagnostic_cases import load_case
    from pdeflow.coordinate.diagnostic_inventory import oracle_model
    from pdeflow.coordinate.joint import path,active_mask
    from pdeflow.coordinate.contracts import grid_queries
    from pdeflow.core import GridSpec
    from unittest.mock import patch
    from pdeflow.numerics.dynamic import SpectralState
    destination=root/'interfaces.json'
    if destination.exists():return
    bundle=CoordinateBundle(run/'flow/coordinate.pt',str(device));bundle.checkpoint_path=str(run/'flow/coordinate.pt');ae,lineage=oracle_model(bundle,run/'evolution/evolution.pt',run/'autoencoder/autoencoder.pt');c=bundle.config;records=[]
    for index in (0,3,7):
        item=example(c,'validation',index);ns=bundle.cp['normalizations'][item['family']];f=torch.tensor([family_index(item['family'])],device=device);p=normalized_parameter(c,torch.tensor([item['parameter']],device=device),f);known=torch.ones(1,dtype=torch.bool,device=device);xy=torch.tensor(item['context'],device=device)[None]
        axes=[torch.arange(32,device=device)/32]*2;space=torch.stack(torch.meshgrid(*axes,indexing='ij'),-1).reshape(-1,2)
        with torch.no_grad():
            z=ae.encode(xy,state_samples(item,item['context'],0.,c,ns).to(device)[None],p,f);standard=(z-bundle.mean)/bundle.scale;roundtrip=standard*bundle.scale+bundle.mean
            noise=torch.randn(standard.shape,generator=torch.Generator().manual_seed(941)).to(device);s=torch.tensor([.37],device=device);state,velocity,mask=path(c,standard,noise,s,known,f)
            row={'id':item['id'],'family':item['family'],'latent_transform_roundtrip_max_abs':float((roundtrip-z).abs().max()),'straight_path_endpoint_max_abs_error':float((state+(1-s[:,None,None])*velocity-standard*mask).abs().max()),'latent_noise_sensitivity':[]}
            perturb=torch.randn(z.shape,generator=torch.Generator().manual_seed(147)).to(device)*active_mask(c,known,f)[...,None]
            for epsilon in (0.,.01,.05,.1):
                fields={};perturbed=(standard+epsilon*perturb)*bundle.scale+bundle.mean
                for j,name in enumerate(get(item['family']).state):
                    prediction=(ae.decode_state(perturbed,space,j,f)[0]*ns[j]['scale']+ns[j]['mean']).cpu().numpy().reshape(32,32);truth=item['solver'].frame(0.,item['parameter'],c.interval,(32,32),name)
                    fields[name]={'physical_mse':float(np.mean((prediction-truth)**2))}
                row['latent_noise_sensitivity'].append({'standardized_independent_noise_rms':epsilon,'quantities':fields})
        records.append(row)
    case,problem,solver,_,noise=load_case(root/'cases/validation-cosine');q=grid_queries(GridSpec((8,8),problem.grid.domain,(0.,.237,1.)));settings={**case['inference'],'steps':4}
    a,first=bundle.sample(problem,case['observations'],q,initial_noise=noise,**settings)
    def forbidden(*args,**kwargs):raise AssertionError('Reference solver called inside ordinary conditional inference')
    with patch.object(SpectralState,'evaluate',forbidden),patch.object(SpectralState,'frame',forbidden):b,second=bundle.sample(problem,case['observations'],q,initial_noise=noise,**settings)
    write_json(destination,{'lineage':lineage,'records':records,'reference_isolation':{'maximum_output_difference':float(np.max(abs(a-b))),'all_reference_calls_forbidden_during_second_inference':True},'scope':'Actual trained weights, no optimizer. Latent perturbations are a decoder sensitivity diagnostic, not a trained noise model or improved reconstruction.'})


def derivative_check(run,root,device):
    cp,c=read_checkpoint(run/'evolution/evolution.pt');ae=Autoencoder(c).to(device).eval().requires_grad_(False);ae.load_state_dict(cp['autoencoder'],strict=True);records=[]
    for index in (0,3,7):
        item=example(c,'validation',index);ns=cp['normalizations'][item['family']];f=torch.tensor([family_index(item['family'])],device=device);p=normalized_parameter(c,torch.tensor([item['parameter']],device=device),f);known=torch.ones(1,dtype=torch.bool,device=device)
        with torch.no_grad():z=ae.encode(torch.tensor(item['context'],device=device)[None],state_samples(item,item['context'],0.,c,ns).to(device)[None],p,f)
        q=torch.tensor([[.31,.43,t] for t in (.02,.237,.5,.83)],device=device)
        for j,name in enumerate(get(item['family']).state):
            differentiable=q.detach().clone().requires_grad_(True);prediction=ae.decode(z,differentiable,p,known,name,f)[0]*ns[j]['scale']+ns[j]['mean'];dt=torch.autograd.grad(prediction.sum(),differentiable)[0][:,-1].detach().cpu().numpy();checks=[]
            with torch.no_grad():
                base=ae.decode(z,q,p,known,name,f)[0]*ns[j]['scale']+ns[j]['mean']
                for h in (.002,.001,.0005,.0002):
                    a=q.clone();b=q.clone();a[:,-1]+=h;b[:,-1]-=h;fd=((ae.decode(z,a,p,known,name,f)[0]-ae.decode(z,b,p,known,name,f)[0])*ns[j]['scale']/(2*h)).cpu().numpy();checks.append({'h':h,'physical_dt':fd.tolist(),'rms_difference':float(np.sqrt(np.mean((fd-dt)**2)))})
            records.append({'index':index,'family':item['family'],'quantity':name,'autodiff_physical_dt':dt.tolist(),'differentiable_vs_grouped_inference_max_abs':float((prediction.detach()-base).abs().max()),'finite_difference':checks})
    write_json(root/'physics/derivative-refinement.json',{'records':records,'precision':'Actual MPS FP32; small-h finite differences have cancellation. Check convergence over h rather than exact equality. No physical reference or model change.'})


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-directory',required=True);p.add_argument('--output',required=True);p.add_argument('--device',choices=['cpu','mps','cuda'],default='cpu');p.add_argument('--phase',choices=['oracles','curves','reconstruction','ablations','correction','physics','interfaces','derivatives'],required=True);args=p.parse_args()
    from pdeflow.runtime import RuntimeSpec
    device,_=RuntimeSpec(device=args.device,threads=2).resolve();run=Path(args.run_directory).resolve();root=Path(args.output).resolve();root.mkdir(exist_ok=True,parents=True)
    config=read_json(run/'resolved-config.json')
    if config['lengths']!=[1.,1.] or config['origins']!=[0.,0.] or config['time_interval']!=[0.,1.] or set(config['family_weights'])!={'heat','wave'} or config['validation_count']<32:
        raise ValueError('This fixed audit protocol requires joint unit-square heat/wave, interval[0,1] and at least32 validation cases; rescale and declare a new protocol for other supports')
    if args.phase!='curves' and scientific_source()['sha256']!=read_json(run/'source/identity.json')['scientific_source']['sha256']:
        raise ValueError('Run numerical phases with the verified preserved scientific source on PYTHONPATH; no silent source change')
    if args.phase=='oracles':oracles(run,root,device)
    elif args.phase=='curves':curves(run,root)
    elif args.phase=='reconstruction':cases_and_b(run,root,device)
    elif args.phase=='ablations':ablations(run,root,device)
    elif args.phase=='correction':corrections(run,root,device)
    elif args.phase=='physics':physics_audit(run,root,device)
    elif args.phase=='interfaces':interfaces(run,root,device)
    else:derivative_check(run,root,device)
if __name__=='__main__':main()
