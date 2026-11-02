"""Immutable shared cases and physical/scoring comparisons; truth is evaluation-only."""
from pathlib import Path
from dataclasses import replace
import time
import numpy as np
import torch
from pdeflow.runtime import write_json,read_json,atomic_write,file_digest,digest
from pdeflow.core import GridSpec,DomainSpec,ProblemSpec
from pdeflow.coordinate.physics import problem as physical_problem,get,invariant_drift
from pdeflow.coordinate.contracts import grid_queries
from pdeflow.coordinate.diagnostic_metrics import sensor_metrics
from pdeflow.coordinate.evaluation import spectral_diagnostics
from pdeflow.coordinate.conditioning import physical_rows
from pdeflow.coordinate.engine import observation_batch
from pdeflow.coordinate.dataset import example
from .contracts import MeshConfig,SCHEMA
from .data import Data
from .inference import MeshBundle
from .sampling import fgn_noise


def create_cases(config,output,count=8,split='validation',seed=917):
    c=(config if isinstance(config,MeshConfig) else MeshConfig(**config)).validate();root=Path(output)
    if root.exists():raise FileExistsError('Cases are immutable; choose a NEW output directory')
    if split not in ('validation','test') or not 1<=count<=64:raise ValueError('Fixed validation/test cases1..64')
    root.mkdir(parents=True);data=Data(c,root/'training-scale-recipe');r=c.reference_config();rng=np.random.default_rng(seed);g=torch.Generator().manual_seed(seed);records=[]
    for index in range(count):
        item=example(r,split,index);family=item['family'];f=torch.tensor([int(family=='wave')]);grid=GridSpec((16,16),DomainSpec(tuple(c.domain['lengths']),tuple(c.domain['origins'])),(0.,.01*c.interval[1],.237*c.interval[1],.83*c.interval[1],c.interval[1]));true=item['parameter'];known=index%2==0;p=physical_problem(grid,family,true if known else None);packed=observation_batch(r,[item],g,data.norms,torch.device('cpu'));rows=physical_rows(packed,data.norms,family);held=[]
        for j in range(32):
            xy=rng.random(2)*np.asarray(c.domain['lengths'])+np.asarray(c.domain['origins']);t=0. if j<16 else rng.uniform(*c.interval);q='v' if family=='wave' and j%2 else 'u';value=float(item['solver'].evaluate(np.array([[*xy,t]]),true,c.interval,q)[0]);held.append({'x':float(xy[0]),'y':float(xy[1]),'time':float(t),'field':q,'value':value,'sigma':0.})
        queries=grid_queries(grid).numpy();truth=np.stack([item['solver'].evaluate(queries,true,c.interval,q) for q in get(family).state],-1);directory=root/f'case-{index:03d}';directory.mkdir();atomic_write(directory/'truth.npz',lambda file:np.savez_compressed(file,reference=truth,initial=item['solver'].initial,velocity=item['solver'].velocity if family=='wave' else np.zeros_like(item['solver'].initial)))
        from pdeflow.coordinate.diagnostic_cases import sensor_truth
        record={'clean_observation_values_evaluation_only':sensor_truth(item['solver'],rows,true,c.interval).tolist(),'schema':'mesh_comparison_case_v1','id':item['id'],'split':split,'problem':p.to_dict(),'observations':rows,'heldout':held,'true_parameter_evaluation_only':true,'normalizations':data.norms[family],'dataset_identity':data.dataset.metadata['sha256'],'noise_seed':seed+index,'truth_sha256':file_digest(directory/'truth.npz'),'reference_contract':'Continuous finite Fourier interpolant, complete u0/v0; clean truth never passed to production inference'};record['identity']=digest(record);write_json(directory/'case.json',record);records.append({'directory':directory.name,'identity':record['identity']})
    write_json(root/'cases.json',{'cases':records,'seed':seed,'split':split,'config':c.to_dict(),'policy':'Generated ONCE for all candidates; case order does not change data/noise. Test cases are final evaluation only.'});return records


def load_case(path):
    root=Path(path);case=read_json(root/'case.json');identity=case.get('identity')
    if case['schema']!='mesh_comparison_case_v1' or digest({k:v for k,v in case.items() if k!='identity'})!=identity or file_digest(root/'truth.npz')!=case['truth_sha256']:raise ValueError('Comparison case identity/checksum mismatch')
    p=ProblemSpec.from_dict(case['problem'])
    with np.load(root/'truth.npz',allow_pickle=False) as a:truth=a['reference'].copy();initial=a['initial'].copy();velocity=a['velocity'].copy()
    return case,p,truth,initial,velocity


def _predict_sensors(bundle,p,rows,mesh):
    if not rows:return np.empty((len(bundle.state if mesh else bundle.decoded_latent),0))
    n=len(bundle.state if mesh else bundle.decoded_latent);result=np.empty((n,len(rows)))
    with torch.no_grad():
        for name in get(p.family).state:
            ix=[i for i,r in enumerate(rows) if r['field']==name]
            if not ix:continue
            q=torch.tensor([[rows[i]['x'],rows[i]['y'],rows[i]['time']] for i in ix])
            if mesh:a=bundle.decode(bundle.state.to(bundle.device),q,bundle.families.to(bundle.device),name)
            else:a=bundle.decode(bundle.decoded_latent.to(bundle.device),q.to(bundle.device),bundle.parameter.to(bundle.device),bundle.known.to(bundle.device),bundle.families.to(bundle.device),name)
            result[:,ix]=a.cpu().numpy()
    return result


def _field_metrics(samples,truth,scale,times,shape):
    error=samples-truth;per=np.mean(error**2,1);mean=samples.mean(0);mse=float(np.mean((mean-truth)**2));floor=(scale*1e-6)**2;den=max(float(np.mean(truth**2)),floor)
    return {'individual_mse':per.tolist(),'individual_rmse':np.sqrt(per).tolist(),'ensemble_mean_mse':mse,'ensemble_mean_rmse':mse**.5,'ensemble_mean_mae':float(abs(mean-truth).mean()),'relative_l2':float(np.sqrt(mse/den)),'relative_l2_floor':floor,'per_time_rmse':np.sqrt(np.mean((mean-truth).reshape(len(times),*shape)**2,(1,2))).tolist(),'times':list(times),'spread_rms':float(np.sqrt(np.mean(samples.var(0)))),'spatial_ensemble_mean_correlation':float(np.corrcoef(mean,truth)[0,1]) if np.std(mean)>1e-12 and np.std(truth)>1e-12 else None,'temporal_increment_mse':float(np.mean((np.diff(samples.reshape(len(samples),len(times),*shape),axis=1)-np.diff(truth.reshape(len(times),*shape),axis=0))**2)) if len(times)>1 else None,'spectral_amplitude_phase':spectral_diagnostics(mean.reshape(len(times),*shape),truth.reshape(len(times),*shape)),'aggregation':'Uniform common periodic nodes and equal declared evaluation times; not uniform continuous-time quadrature'}


def _scores(samples,truth,scale):
    x=np.asarray(samples)/scale;y=np.asarray(truth)/scale;m=len(x)
    if m<2:return {'available':False,'reason':'Deterministic or single-sample output; no ensemble calibration claim'}
    first=np.mean(abs(x-y));pairs=sum(np.mean(abs(x[i]-x[j])) for i in range(m) for j in range(i));crps=first-pairs/(m*(m-1));norm=lambda a:float(np.sqrt(np.mean(a*a)));energy=np.mean([norm(a-y) for a in x])-sum(norm(x[i]-x[j]) for i in range(m) for j in range(i))/(m*(m-1));lo,hi=np.quantile(x,[.05,.95],axis=0)
    return {'available':True,'sample_count':m,'fair_crps_normalized':float(crps),'fair_energy_normalized_query_vector':float(energy),'empirical_90_coverage':float(np.mean((y>=lo)&(y<=hi))),'mean_interval_width_normalized':float(np.mean(hi-lo)),'limits':'Tiny empirical quantiles do not establish calibration. Pointwise CRPS does not determine joint dependence; energy loses sensitivity in high dimension.'}


def compare(checkpoints,cases_root,output,device='cpu',particles=4,steps=8):
    root=Path(output)
    if root.exists():raise FileExistsError('Preserve comparison outputs; use a NEW directory')
    root.mkdir(parents=True);case_files=sorted(Path(cases_root).glob('*/case.json'));all_models={}
    if not case_files:raise ValueError('No immutable comparison cases')
    for label,path in checkpoints.items():
        raw=torch.load(path,map_location='cpu',weights_only=True);mesh=raw.get('schema')==SCHEMA
        if mesh:bundle=MeshBundle(path,device);c=bundle.config;ref=c.reference_config();generative=c.kind['generative'];initial_only=not c.kind['trajectory'];parameter_count=sum(x.numel() for x in bundle.model.parameters())
        else:
            from pdeflow.coordinate.inference import CoordinateBundle
            bundle=CoordinateBundle(path,device);c=bundle.config;ref=c;generative=True;initial_only=False;parameter_count=sum(x.numel() for x in bundle.ae.parameters())+sum(x.numel() for x in bundle.velocity.parameters())
        records=[]
        for file in case_files:
            case,p,truth,initial,velocity=load_case(file.parent);rows=case['observations'];q=grid_queries(p.grid);times=p.grid.times
            if initial_only:q=q[:np.prod(p.grid.shape)];truth=truth[:len(q)];times=(0.,)
            count=particles if generative else 1;g=torch.Generator().manual_seed(case['noise_seed'])
            noise=fgn_noise(c,count,g,bundle.device) if mesh and c.architecture=='mesh_m3_fgn_v1' else torch.randn(count,c.nodes+1,2,generator=g) if mesh and generative else torch.randn(count,c.tokens,c.width,generator=g) if not mesh else None
            from pdeflow.coordinate.diagnostics import shuffle_within_quantity
            variants={'correct':rows,'tuple_permuted':list(reversed(rows)),'wrong_value_pairing':shuffle_within_quantity(rows,np.random.default_rng(case['noise_seed'])),'wrong_time_pairing':[{**r,'time':ref.interval[0]+ref.interval[1]-r['time']} for r in rows]};results={};correct_array=None
            for name,observations in variants.items():
                started=time.perf_counter();a,meta=bundle.sample(p,observations,q,steps=steps,particles=count,seed=case['noise_seed'],initial_noise=noise);wall=time.perf_counter()-started;arrays=[a]
                if p.family=='wave':arrays.append(bundle.velocity_samples)
                eligible=lambda r:not initial_only or r['time']==0
                conditioning=[r for r in rows if eligible(r)];held=[r for r in case['heldout'] if eligible(r)];sensor=_predict_sensors(bundle,p,conditioning,mesh);unseen=_predict_sensors(bundle,p,held,mesh);fields={};scores={}
                for j,quantity in enumerate(get(p.family).state):
                    scale=case['normalizations'][j]['scale'];fields[quantity]=_field_metrics(arrays[j],truth[:,j],scale,times,p.grid.shape);scores[quantity]=_scores(arrays[j],truth[:,j],scale)
                physical=meta.get('physics',{})
                if mesh and not initial_only:
                    from types import SimpleNamespace
                    from .objectives import physical_loss
                    probe=torch.tensor([[[.31*c.domain['lengths'][0]+c.domain['origins'][0],.43*c.domain['lengths'][1]+c.domain['origins'][1],t*c.interval[1]] for t in (.02,.237,.5,.83)]],device=bundle.device).expand(count,-1,-1)
                    _,residuals=physical_loss(bundle.model,bundle.state.to(bundle.device),SimpleNamespace(families=bundle.families.to(bundle.device)),bundle.cp['normalizations'],probe,True);physical={**physical,'normalized_decoded_residual_mse':{k:v.detach().cpu().tolist() for k,v in residuals.items()},'residual_scope':'Four declared x/y/t probes, independent per generated sample; not a whole-domain guarantee'}
                joint=np.concatenate([arr/case['normalizations'][j]['scale'] for j,arr in enumerate(arrays)],1);joint_truth=np.concatenate([truth[:,j]/case['normalizations'][j]['scale'] for j in range(truth.shape[1])]);joint_score=_scores(joint,joint_truth,1.)
                invariants=invariant_drift(np.asarray(arrays[0]).reshape(count,len(times),*p.grid.shape),np.asarray(arrays[1]).reshape(count,len(times),*p.grid.shape) if len(arrays)>1 else None,times,p.grid.domain.lengths,meta['inferred_parameters'][get(p.family).parameter])
                if correct_array is None:correct_array=a
                results[name]={'fields':fields,'scores':scores,'joint_quantity_space_time_score':joint_score,'conditioning':sensor_metrics(sensor,conditioning,[v for r,v in zip(rows,case['clean_observation_values_evaluation_only']) if eligible(r)]),'heldout':sensor_metrics(unseen,held,[r['value'] for r in held]),'parameter_values':meta['inferred_parameters'],'parameter_absolute_errors':[abs(v-case['true_parameter_evaluation_only']) for v in meta['inferred_parameters'][get(p.family).parameter]],'known_parameter':bool(p.known_parameters),'seconds_ensemble':wall,'seconds_per_sample':wall/count,'generation_seconds':meta.get('generation_seconds'),'physical_evolution_and_query_decode_seconds':meta.get('physical_evolution_and_query_decode_seconds'),'rms_change_from_correct':float(np.sqrt(np.mean((a-correct_array)**2))),'maximum_change_from_correct':float(np.max(abs(a-correct_array))),'physics':physical,'invariants':invariants,'memory':meta.get('memory'),'query_workload':meta.get('query_workload'),'observation_scope':'Only t=0 sensors evaluable by M1; later known timestamps still condition its encoder' if initial_only else 'Every stored sensor time evaluated exactly'}
                if name=='correct':atomic_write(root/(label+'-'+file.parent.name+'.npz'),lambda stream:np.savez_compressed(stream,samples=np.stack(arrays,-1),queries=q.numpy(),truth=truth))
            records.append({'case_identity':case['identity'],'id':case['id'],'family':p.family,'split':case['split'],'task':'initial-only' if initial_only else 'time-dependent reconstruction','dataset_identity_matches':bundle.cp.get('dataset_identity')==case['dataset_identity'],'variants':results})
        provenance=bundle.cp.get('training_provenance',{});all_models[label]={'checkpoint':str(Path(path).resolve()),'checkpoint_sha256':file_digest(path),'architecture':bundle.cp['architecture'],'parameter_count':parameter_count,'training_provenance':provenance,'observed_total_examples':sum(x.get('effective_examples',x.get('examples_processed',0)) for x in provenance.values() if isinstance(x,dict)) if any(isinstance(x,dict) and ('effective_examples' in x or 'examples_processed' in x) for x in provenance.values()) else None,'observed_training_seconds':sum(x.get('training_seconds',0) for x in provenance.values() if isinstance(x,dict)) if any(isinstance(x,dict) and 'training_seconds' in x for x in provenance.values()) else None,'cost_missing_policy':'Unrecorded costs are null, never zero; attach original run metadata for full legacy accounting','records':records,'legacy':not mesh,'generative':generative,'accuracy_claim':False};write_json(root/(label+'.json'),all_models[label])
    summary={'models':all_models,'comparison_policy':'All samples and ensemble means; no hidden best-of-N. Same stored observations/truth/query coordinates and declared evaluation times. Fixed noise within each model for pairing controls; noise dimension differs across model families.','equal_budget_summary':{'examples':{k:v['observed_total_examples'] for k,v in all_models.items()},'wall_seconds':{k:v['observed_training_seconds'] for k,v in all_models.items()},'equal_wall_accuracy':'Unavailable unless matching wall-time checkpoint exports are supplied; no interpolation of model accuracy from final losses.','confounds':'Warm starts, pretraining, differing datasets/capacities and objectives must be disclosed; parameter matching alone is not fairness.'},'calibration':'Not established by these bounded ensembles'};write_json(root/'comparison.json',summary);return summary
