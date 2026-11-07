"""Fixed observation-only validation; dynamics stage explicitly uses true physical nodes."""
import torch
import copy,hashlib
from dataclasses import replace
from contextlib import contextmanager
from pdeflow.runtime import digest,write_json,read_json
from pdeflow.coordinate.evolution_monitor import evaluation_guard
from .sampling import generate,fgn_noise
from .objectives import predictions,quantity_mean


@contextmanager
def _guard(model):
    device=next(model.parameters()).device
    mps=torch.mps.get_rng_state() if device.type=='mps' else None
    try:
        with evaluation_guard(model),torch.no_grad():yield
    finally:
        if mps is not None:torch.mps.set_rng_state(mps)


def _tensor_id(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def fixed_batches(model,data,cases=None):
    """Cache only this stage's bounded immutable reference/conditioning cohort on CPU."""
    c=model.c;number=cases or c.validation_cases
    key=digest({'protocol':c.validation_protocol,'cases':number,'data':data.reference.to_dict(),
                'batch':data.c.batch_size if c.validation_protocol=='legacy_v1' else len(c.family_weights),
                'sampler':c.sampler,'architecture':c.architecture,'norms':data.norms})
    cache=getattr(data,'_mesh_validation_cache',{})
    if key in cache:return cache[key]
    view=copy.copy(data)
    from pdeflow.field_losses import AnalysisCache
    view.analysis=AnalysisCache()
    if c.validation_protocol=='balanced_v2':
        view.c=replace(data.c,batch_size=len(c.family_weights))
        number=min(number,min(len(data.indices['validation'][f]) for f in c.family_weights))
        if not number:raise ValueError('Fixed validation needs held-out cases for every enabled family')
    batches=tuple(view.prepare(i,'validation') for i in range(number))
    spec={'protocol':c.validation_protocol,'architecture':c.architecture,'dataset_identity':data.dataset.metadata['sha256'],
          'reference_recipe':data.dataset.metadata['recipe'],'normalizations':data.norms,'sampler':c.sampler,
          'evaluation_generator':'CPU torch.Generator, fixed per-batch seed; coherent state reused across physical query times',
          'normalization':'Squared error / fixed TRAIN family/quantity scale^2; scale=max(TRAIN std,1e-3), not per-case or prediction variance',
          'time_weighting':'Initial only for M1; otherwise equal initial and fixed sampled target time per case, not uniform space-time error',
          'batches':[]}
    for i,b in enumerate(batches):
        initial=[]
        for identifier in b.ids:
            item=data.dataset.get('validation',int(identifier.split(':')[1]))
            initial.append({'id':identifier,'family':item['family'],'parameter':item['parameter'],'case':item['case'],
                            'seed':item['seed'],'reference_initial_sha256':hashlib.sha256(item['initial'].tobytes()).hexdigest()})
        spec['batches'].append({'cases':initial,'case_ids':b.ids,'known_parameters':b.known.tolist(),
            'parameter_normalized':b.parameter.tolist(),'families':b.families.tolist(),
            'observations':{'coordinates_and_measurement_times':b.observations[0].tolist(),
                'normalized_value_sigma_and_features':b.observations[1].tolist(),'valid':b.observations[2].tolist(),'quantities':b.observations[3].tolist()},
            'initial_queries':b.q0[...,:b.domain_queries,:].tolist(),'target_queries':b.qt[...,:b.domain_queries,:].tolist(),
            'physical_target_times':b.times.tolist(),'initial_time':0.,'domain_queries_per_quantity_time':b.domain_queries,
            'reference_target_sha256':[_tensor_id(b.y0),_tensor_id(b.yt)],
            'evaluation_seed':c.seed+991+(i if c.validation_protocol=='balanced_v2' else 0)})
    spec['sha256']=digest(spec);cache[key]=(batches,spec);data._mesh_validation_cache=cache
    return batches,spec


def group_metrics(rows):
    """Equal cases and fixed time weights inside each declared physical quantity."""
    groups={}
    for family,names in (('heat',('u',)),('wave',('u','v'))):
        for name in names:
            selected=[r for r in rows if r['family']==family and name in r['quantity_metrics']]
            if not selected:continue
            values=[r['quantity_metrics'][name] for r in selected];mse=sum(v['physical_mse'] for v in values)/len(values)
            groups[family+'/'+name]={'physical_mse':mse,'physical_rmse':mse**.5,
                'normalized_mse':sum(v['normalized_mse'] for v in values)/len(values),
                'case_count':len(values),'unique_case_count':len({r['id'] for r in selected}),
                'evaluated_points':sum(v['evaluated_points'] for v in values),'normalization_scale':values[0]['normalization_scale'],
                'initial_normalized_mse':sum(v['initial_normalized_mse'] for v in values)/len(values),
                'target_normalized_mse':sum(v['target_normalized_mse'] for v in values)/len(values) if values[0]['target_normalized_mse'] is not None else None}
    return groups


def validate(model,data,stage,cases=None,preview_root=None,update=None):
    c=model.c;device=next(model.parameters()).device;rows=[];images=[];labels=[]
    with _guard(model):
        batches,spec=fixed_batches(model,data,cases)
        for i,held in enumerate(batches):
            batch=held.to(device)
            if stage=='dynamics':state=batch.target_state;scope='Oracle clean physical initial nodes, not observation-only reconstruction'
            elif stage=='central':
                state=model.initial(model.condition(batch.observations,batch.parameter,batch.known,batch.families),batch.parameter,batch.known,batch.families,fgn_noise(c,len(batch.families),torch.Generator(),device,True));scope='Deterministic zero-noise branch, not ensemble mean'
            else:
                context=model.condition(batch.observations,batch.parameter,batch.known,batch.families);state,_=generate(model,context,batch.parameter,batch.known,batch.families,torch.Generator().manual_seed(spec['batches'][i]['evaluation_seed']));scope='Observation-only fixed-seed prediction; no reference field enters model'
            initial=predictions(model,state,batch,True);truth=batch.y0;count=batch.domain_queries;initial_quantities=(initial[...,:count]-truth[...,:count]).square().mean(-1);initial_mse=quantity_mean(initial_quantities,batch.families);constant0=quantity_mean(truth[...,:count].square().mean(-1),batch.families);trajectory=torch.zeros_like(initial_mse);constantt=torch.zeros_like(initial_mse)
            if c.kind['trajectory']:
                pred=predictions(model,state,batch,False);truth=batch.yt;target_quantities=(pred[...,:count]-truth[...,:count]).square().mean(-1);trajectory=quantity_mean(target_quantities,batch.families);constantt=quantity_mean(truth[...,:count].square().mean(-1),batch.families)
            wi=.5 if c.kind['trajectory'] else 1.;wt=.5 if c.kind['trajectory'] else 0.;mse=wi*initial_mse+wt*trajectory;base=wi*constant0+wt*constantt
            if preview_root is not None and i==0:
                from .geometry import regular
                xy=regular((12,12)).to(device)*state.new_tensor(c.domain['lengths'])+state.new_tensor(c.domain['origins']);time=.237*c.interval[1] if c.kind['trajectory'] else 0.;pred=model.decode(state,xy,batch.families,time).cpu().numpy()
                for j in range(min(2,len(state))):
                    item=data.dataset.get('validation',int(batch.ids[j].split(':')[1]));family=item['family']
                    for quantity,name in enumerate(('u','v') if family=='wave' else ('u',)):
                        norm=data.norms[family][quantity];field=pred[j,:,quantity].reshape(12,12)*norm['scale']+norm['mean'];truth=item['solver'].frame(time,item['parameter'],c.interval,(12,12),name);physical_mse=float(__import__('numpy').mean((field-truth)**2));images.append(__import__('numpy').stack([truth,field,abs(field-truth)]));labels.append({'id':item['id'],'family':family,'quantity':name,'physical_time':time,'checkpoint_update':update,'rmse':physical_mse**.5,'normalized_mse':physical_mse/norm['scale']**2,'scope':scope,'stage':stage})
            for j in range(len(state)):
                family=('heat','wave')[int(batch.families[j])];quantities={}
                for k,name in enumerate(('u','v') if family=='wave' else ('u',)):
                    scale=data.norms[family][k]['scale'];n0=float(initial_quantities[j,k]);nt=float(target_quantities[j,k]) if c.kind['trajectory'] else None
                    normalized=wi*n0+(wt*nt if nt is not None else 0.)
                    quantities[name]={'physical_mse':normalized*scale**2,'normalized_mse':normalized,'normalization_scale':scale,
                        'initial_normalized_mse':n0,'target_normalized_mse':nt,'evaluated_points':count*(2 if c.kind['trajectory'] else 1)}
                rows.append({'id':batch.ids[j],'quantity_metrics':quantities,'family':('heat','wave')[int(batch.families[j])],'initial_normalized_mse':float(initial_mse[j]),'trajectory_normalized_mse':float(trajectory[j]) if c.kind['trajectory'] else None,'physical_target_time':float(batch.times[j]) if c.kind['trajectory'] else 0.,'mse':float(mse[j]),'constant_training_mean_mse':float(base[j]),'scope':scope})
    if preview_root is not None and images:
        from pathlib import Path
        import numpy as np
        from pdeflow.runtime import atomic_write,write_json,file_digest
        spec_path=Path(preview_root)/'validation-spec.json'
        if spec_path.exists():
            if read_json(spec_path).get('sha256')!=spec['sha256']:raise ValueError('Fixed validation specification changed; preserve its original cohort and use a NEW run/directory')
        else:write_json(spec_path,spec)
        path=Path(preview_root)/'preview.npz';atomic_write(path,lambda f:np.savez_compressed(f,frames=np.stack(images)));write_json(path.with_suffix('.json'),{'rows':labels,'array_sha256':file_digest(path),'checkpoint_update':update,'stage':stage,'meaning':'Fresh current-stage prediction, scope explicitly distinguishes oracle physical dynamics from observation-only inference'})
    a=sum(r['mse'] for r in rows)/len(rows);b=sum(r['constant_training_mean_mse'] for r in rows)/len(rows)
    groups=group_metrics(rows)
    def balanced(metric):
        families={f:[r[metric] for key,r in groups.items() if key.startswith(f+'/')] for f in c.family_weights}
        return sum(sum(values)/len(values) for values in families.values())/len(families) if all(families.values()) else None
    physical=sum(sum(q['physical_mse'] for q in r['quantity_metrics'].values())/len(r['quantity_metrics']) for r in rows)/len(rows)
    return {'stage':stage,'update':update,'architecture':c.architecture,'records':rows,'groups':groups,
        'decoded_field_mse':physical,'balanced_family_physical_mse':balanced('physical_mse'),
        'balanced_family_normalized_mse':balanced('normalized_mse'),
        'aggregation':'Legacy aggregate: equal evaluated case occurrences, equal active quantities within case; M1 initial only, otherwise equal initial/target-time error. Balanced metrics: equal enabled families, equal active quantities per family, equal cases per quantity. Physical aggregate mixes squared quantity units; compare quantities using fixed-scale normalized errors.',
        'prediction_scope':scope,'case_count':len(rows),'unique_case_count':len({r['id'] for r in rows}),
        'evaluated_points':sum(v['evaluated_points'] for v in groups.values()),
        'generated_samples':len(rows) if c.kind['generative'] and stage not in ('central','dynamics') else 0,
        'state_predictions':len(rows),'validation_spec_sha256':spec['sha256'],'validation_protocol':c.validation_protocol,
        'normalization':spec['normalization'],'time_weighting':spec['time_weighting'],
        'uniform_domain_normalized_mse':a,'constant_training_mean_mse':b,'mse_ratio':a/max(b,1e-12),'passed':a<=c.gate_ratio*b,
        'criterion':'Matched uniform spatial domain MSE, equal active quantities/cases and initial/trajectory terms; not RB-vs-MSE loss comparison',
        'accuracy':'Minimal baseline gate, not scientific adequacy','test_split_used':False}
