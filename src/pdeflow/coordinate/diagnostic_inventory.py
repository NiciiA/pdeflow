"""Checkpoint budget/lineage audit without assuming config budgets were executed."""
from pathlib import Path
import torch
from pdeflow.runtime import file_digest,digest
from .training import read_checkpoint
from .optimization import scientific_source,options
from .model import Autoencoder,Velocity


def inventory(checkpoint):
    cp,c=read_checkpoint(checkpoint,'flow');history=cp.get('history',[])
    with torch.device('meta'):
        ae=Autoencoder(c);velocity=Velocity(c)
    counts={name:sum(p.numel() for p in module.parameters()) for name,module in [('state_encoder',ae.encoder),('spatial_decoder',ae.decoder),('coefficient',ae.coefficient),('graph_encoder',ae.graph_encoder),('evolution',ae.evolution),('flow',velocity)]}
    now=scientific_source();changed=[name for name,sha in cp.get('scientific_source',{}).get('files',{}).items() if now['files'].get(name)!=sha]
    result={'checkpoint':str(Path(checkpoint).resolve()),'sha256':file_digest(checkpoint),'architecture':cp['architecture'],'schema':cp['schema'],'source':cp.get('source'),'scientific_source':cp.get('scientific_source'),'changed_scientific_files':changed,
       'parameter_counts_excluding_buffers':counts,'total_parameters':sum(counts.values()),'portable_inference':cp.get('portable_inference',False),'state_encoder_available':bool(cp.get('autoencoder') and any(k.startswith('encoder.') for k in cp['autoencoder'])),
       'actual_flow_successful_updates':cp['progress'],'attempted_updates':cp.get('attempted_updates'),'skipped_updates':cp.get('skipped_updates'),'effective_examples_processed':cp.get('effective_examples_processed'),
       'declared_stage_budgets_not_proof_of_execution':{k:getattr(c,k) for k in ('ae_steps','evolution_steps','flow_steps')},'preceding_dependency_record':cp.get('dependency'),
       'initial_config':cp.get('initial_config'),'continuation_history':cp.get('continuation_history'),'resolved_config':c.to_dict(),'training_precision':cp.get('actual_precision'),'environment':cp.get('environment'),
       'training_status':cp.get('training_status'),'gate_provenance':cp.get('training_provenance',{}).get('representation_gate','not included in this checkpoint; unknown, not passed'),
       'checkpoint_selection':'latest successful optimizer boundary; no best-validation selection recorded',
       'dataset_identity':cp.get('dataset_identity'),'training_ids_sha256':digest(cp.get('training_ids',[])),'training_id_count':len(cp.get('training_ids',[])),
       'normalizations':cp['normalizations'],'normalization_fitted_ids':cp.get('normalization_fitted_ids'),'latent_calibration':{'mean_shape':list(cp['latent_mean'].shape),'scale_shape':list(cp['latent_scale'].shape),'minimum_scale':float(cp['latent_scale'].min()),'maximum_scale':float(cp['latent_scale'].max()),'all_finite':bool(torch.isfinite(cp['latent_mean']).all() and torch.isfinite(cp['latent_scale']).all())},
       'loss':c.field_loss,'effective_field_supervision':{'recorded_generic_time_groups':c.field_loss.get('time_groups',4),'state_time_groups_per_example':1,'domain_queries_per_quantity':c.query_points,'extra_region_slots_per_quantity':c.field_loss.get('region_queries',0) if c.field_loss.get('type')=='region_balanced_mse' else 0,'policy':'Operator A1/A2 plans force one coherent physical-state time per example/quantity. Initial/early/uniform/final times vary continuously across examples/updates; this is not the generic four-time-group trajectory path.'},'flow_loss':'active-group balanced latent velocity MSE; separate from decoded representation loss','batch_settings':cp.get('batch_settings'),
       'scheduler':{s:options(c)[s] for s in ('autoencoder','evolution','flow')},'scheduler_provenance':'Configuration-derived if optimizer/scheduler removed; history contains executed LR values',
       'history_summary':{'rows':len(history),'first_learning_rate':history[0]['learning_rate'] if history else None,'last_learning_rate':history[-1]['learning_rate'] if history else None,'scales':sorted({r['scaler_scale'] for r in history}),'nonfinite_gradients':sum(not r['gradients_finite'] for r in history),'first_loss':history[0]['training_loss'] if history else None,'last_loss':history[-1]['training_loss'] if history else None}}
    return result


def oracle_model(bundle,evolution=None,representation=None):
    """No random encoder substitution: require the actual frozen dependency."""
    cp=bundle.cp;c=bundle.config
    if not cp.get('portable_inference') and cp.get('autoencoder') and any(k.startswith('encoder.') for k in cp['autoencoder']):return bundle.ae,{'source':'embedded frozen encoder'}
    path=Path(evolution) if evolution else Path(cp.get('dependency',{}).get('path',''))
    if not evolution:path=Path(bundle.checkpoint_path).parent/path
    if not path.is_file():return None,{'available':False,'reason':'Portable inference export omits the state encoder; matching A2 checkpoint required','expected_a2_sha256':cp.get('dependency',{}).get('sha256')}
    if file_digest(path)!=cp['dependency']['sha256']:raise ValueError('Diagnostic A2 checkpoint hash differs from B frozen dependency')
    prior,other=read_checkpoint(path,'evolution')
    if prior['dataset_identity']!=cp['dataset_identity'] or prior['normalizations']!=cp['normalizations']:raise ValueError('Diagnostic stage dataset/normalization mismatch')
    state=prior['autoencoder']
    for name,tensor in cp['inference_state'].items():
        if name not in state or not torch.equal(tensor,state[name]):raise ValueError('Frozen inference tensor differs from diagnostic A2: '+name)
    if representation:
        a=Path(representation)
        if file_digest(a)!=prior['dependency']['sha256']:raise ValueError('Diagnostic A1 hash differs from A2 dependency')
        first,_=read_checkpoint(a,'autoencoder')
        for name,tensor in first['autoencoder'].items():
            if name.startswith(('encoder.','decoder.','coefficient.')) and not torch.equal(tensor,state[name]):raise ValueError('A1 frozen representation changed in A2: '+name)
    ae=Autoencoder(c).to(bundle.device).eval().requires_grad_(False);ae.load_state_dict(state,strict=True)
    return ae,{'available':True,'source':'matching A2 embedded frozen A1 encoder; B decoder/evolution tensors checked exactly','checkpoint':str(path.resolve()),'sha256':file_digest(path),'a2_successful_updates':prior['progress'],'a1_dependency':prior.get('dependency'),'explicit_a1_verified':bool(representation)}


def conditioning_distribution(c,problem,rows):
    times=[r['time'] for r in rows];sigmas=[r.get('sigma',0.) for r in rows]
    return {'training_observation_count_range':list(c.observation_counts),'actual_count':len(rows),'count_outside_training_range':not c.observation_counts[0]<=len(rows)<=c.observation_counts[1],
            'training_explicit_noiseless_probability':c.joint.get('noiseless_probability',0.),'noiseless_mass_known_missing':c.joint.get('noiseless_probability',0.)==0 and c.noise_range[1]>0,'training_absolute_noise_range':list(c.noise_range),'actual_absolute_noise_range':[min(sigmas),max(sigmas)] if sigmas else None,
            'noise_outside_training_range':any(s<c.noise_range[0] or s>c.noise_range[1] for s in sigmas),
            'physical_times':sorted(set(times)),'all_sensors_at_one_time':bool(times and len(set(times))==1),'training_time_sampling':__import__('pdeflow.coordinate.temporal',fromlist=['policies']).policies(c),
            'actual_quantities':sorted({r['field'] for r in rows}),'training_wave_quantities':'Independent u/v random choice per sensor','training_placement':c.observation_placement or {'random':.5,'even':.25,'heat':.25},
            'known_parameter':bool(problem.known_parameters),'training_known_probability':c.joint['known_probability'],
            'interpretation':'Physical support and training-distribution match are different. Dense single-time or single-quantity sets can differ from randomly mixed training sets even when individual queries are supported; no causal attribution follows from this audit alone.'}
