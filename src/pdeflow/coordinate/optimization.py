"""Shared optimizer-step semantics, precision policy and resume compatibility."""
import hashlib
import math
from pathlib import Path
import torch
from pdeflow.runtime import digest


def options(c):
    out={'precision':'fp32','microbatch':c.batch_size,'accumulation':1,'activation_checkpointing':False,
         'workers':0,'pin_memory':True,'validation_every':c.checkpoint_every,'validation_subset':min(c.validation_count,8),
         'broad_every':max(c.checkpoint_every,5000),'broad_subset':min(c.validation_count,24),
         'calibration_count':min(c.train_count,128),'keep_periodic':2,'observation_locations':'grid',
         'autoencoder':{},'evolution':{},'flow':{},'allow_source_change':False,'compile':False,'profile_components':False}
    if set(c.training_options)-out.keys():raise ValueError('Unknown training options: '+', '.join(sorted(set(c.training_options)-out.keys())))
    out.update(c.training_options)
    if out['precision'] not in ('auto','fp32','bf16','fp16'):raise ValueError('Precision: auto, fp32, bf16 or fp16')
    for key in ('microbatch','accumulation','validation_every','validation_subset','broad_every','broad_subset','calibration_count','keep_periodic'):
        if type(out[key])!=int or out[key]<1:raise ValueError(f'{key} must be a positive integer')
    if out['microbatch']>c.batch_size or out['accumulation']!=math.ceil(c.batch_size/out['microbatch']):raise ValueError('Accumulation must equal ceil(effective batch size / microbatch); the final microbatch may be smaller')
    if type(out['workers'])!=int or not 0<=out['workers']<=4:raise ValueError('Use 0–4 bounded data-generation threads')
    if out['compile']:raise ValueError('Compilation is not enabled for this checkpointed dynamic observation architecture; use eager execution')
    if out['observation_locations'] not in ('grid','scattered'):raise ValueError('Observation locations: grid or scattered')
    if out['observation_locations']=='scattered' and c.data_mode!='procedural':raise ValueError('Scattered training observations require procedural Fourier references')
    for stage,total in [('autoencoder',c.ae_steps),('evolution',c.evolution_steps),('flow',c.flow_steps)]:
        value={'learning_rate':c.learning_rate,'weight_decay':0.,'warmup':0,'schedule':'constant','horizon':total,'minimum_ratio':.1,'gradient_clip':10.}
        if set(out[stage])-value.keys():raise ValueError(f'Unknown {stage} optimizer setting')
        value.update(out[stage]);out[stage]=value
        if value['schedule'] not in ('constant','cosine') or type(value['warmup'])!=int or value['warmup']<0 or type(value['horizon'])!=int or value['horizon']<1:raise ValueError('Invalid optimizer schedule')
        if not all(math.isfinite(value[k]) for k in ('learning_rate','weight_decay','minimum_ratio','gradient_clip')) or value['learning_rate']<=0 or value['weight_decay']<0 or not 0<value['minimum_ratio']<=1 or value['gradient_clip']<=0:raise ValueError('Invalid optimizer numerical settings')
    return out


def precision(policy,device):
    if device.type!='cuda':
        if policy not in ('auto','fp32'):raise ValueError('BF16/FP16 training here requires a CUDA device; select fp32 for CPU/MPS')
        return 'fp32',torch.float32
    if policy=='auto':policy='bf16' if torch.cuda.is_bf16_supported() else 'fp16'
    if policy=='bf16' and not torch.cuda.is_bf16_supported():raise ValueError('This GPU does not support BF16; select fp16 or fp32 explicitly')
    return policy,{'fp32':torch.float32,'bf16':torch.bfloat16,'fp16':torch.float16}[policy]


def learning_rate(settings,completed):
    if completed<settings['warmup']:return settings['learning_rate']*(completed+1)/max(1,settings['warmup'])
    if settings['schedule']=='constant':return settings['learning_rate']
    phase=min(1.,max(0.,(completed-settings['warmup'])/max(1,settings['horizon']-settings['warmup'])))
    return settings['learning_rate']*(settings['minimum_ratio']+(1-settings['minimum_ratio'])*(1+math.cos(math.pi*phase))/2)


def scientific_source():
    base=Path(__file__).parents[1]
    names=['coordinate/contracts.py','coordinate/graph.py','coordinate/physics.py','numerics/dynamic.py','coordinate/model.py','coordinate/joint.py','coordinate/consistency.py','coordinate/profiling.py','coordinate/training.py','coordinate/engine.py','coordinate/dataset.py','coordinate/optimization.py','coordinate/temporal.py','heat_continuous.py','heat_placement.py','field_losses.py','numerics/__init__.py','core/__init__.py','equations/__init__.py']
    files={n:hashlib.sha256((base/n).read_bytes()).hexdigest() for n in names}
    return {'files':files,'sha256':digest(files),'policy':'Scientific model/data/loss/numerical source, excluding notebook, docs and unrelated UI'}


def model_contract(c):
    ignored=('animation_frames','times') if c.temporal_mode=='continuous' else ()
    return {**{k:v for k,v in c.to_dict().items() if k not in (*ignored,'runtime','ae_steps','evolution_steps','flow_steps','learning_rate','checkpoint_every','batch_size','training_options')},
            'normalization_calibration_count':min(c.train_count,options(c)['calibration_count'])}


def resume_contract(old,c,stage,extend=False):
    from .contracts import CoordinateConfig
    prior=CoordinateConfig(**old['config']).validate()
    differences=[k for k,v in model_contract(c).items() if digest(v)!=digest(model_contract(prior).get(k))]
    if differences:raise ValueError('Incompatible model/data changes: '+', '.join(differences))
    old_total=getattr(prior,{'autoencoder':'ae_steps','evolution':'evolution_steps','flow':'flow_steps'}[stage]);new_total=getattr(c,{'autoencoder':'ae_steps','evolution':'evolution_steps','flow':'flow_steps'}[stage])
    if new_total!=old_total and not (extend and new_total>old_total):raise ValueError('A changed update budget requires deliberate extend mode and a larger total')
    current=options(c)[stage];saved=old['scheduler']['settings']
    # An extension keeps the original schedule horizon, then holds its terminal LR.
    for key in current:
        # A default horizon derived from the increased budget must not replace
        # the saved horizon, including subsequent ordinary resumes. Explicit
        # horizon changes remain incompatible with continuation.
        if key=='horizon' and 'horizon' not in c.training_options.get(stage,{}):continue
        if current[key]!=saved[key]:raise ValueError(f'Incompatible {stage} optimizer change: {key}')
    changed_source=old.get('scientific_source',{}).get('sha256')!=scientific_source()['sha256']
    if changed_source and not options(c)['allow_source_change']:raise ValueError('Scientific training source changed; use the recorded source or explicitly record allow_source_change (not exact reproducibility)')
    return saved,changed_source


def representation_contract(c):
    """Flow-only auxiliary/cache switches do not change the frozen representation."""
    value=model_contract(c);value['joint']={k:v for k,v in value['joint'].items() if k not in ('auxiliary','target_cache')}
    return value
