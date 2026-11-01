"""Shared optimizer-step semantics, precision policy and resume compatibility."""
import hashlib
import math
from pathlib import Path
import torch
from pdeflow.runtime import digest


def options(c,stage=None):
    out={'precision':'fp32','microbatch':c.batch_size,'accumulation':1,'activation_checkpointing':False,
         'workers':0,'pin_memory':True,'validation_every':c.checkpoint_every,'validation_subset':min(c.validation_count,8),
         'broad_every':max(c.checkpoint_every,5000),'broad_subset':min(c.validation_count,24),
         'calibration_count':min(c.train_count,128),'keep_periodic':2,'observation_locations':'grid',
         'autoencoder':{},'evolution':{},'flow':{},'allow_source_change':False,'compile':False,'profile_components':False,'performance_mode':'optimized','reference_cache_items':32,'frozen_cache_items':128,'execution':'batched','frozen_prefetch':1,'stage_batches':{},'batch_selection':None,'batch_change_authorization':None,'max_consecutive_overflows':8,'initial_scale':65536.}
    if set(c.training_options)-out.keys():raise ValueError('Unknown training options: '+', '.join(sorted(set(c.training_options)-out.keys())))
    out.update(c.training_options)
    if out['execution'] not in ('batched','accumulated'):raise ValueError('Execution: batched or accumulated')
    if type(out['frozen_prefetch'])!=int or out['frozen_prefetch'] not in (0,1):raise ValueError('Frozen prefetch must be0 or1 next chunks')
    if out['performance_mode'] not in ('optimized','legacy'):raise ValueError('Performance mode: optimized or legacy (comparison only)')
    for key in ('reference_cache_items','frozen_cache_items'):
        if type(out[key])!=int or not 1<=out[key]<=256:raise ValueError(key+' must be 1..256')
    if out['precision'] not in ('auto','fp32','bf16','fp16'):raise ValueError('Precision: auto, fp32, bf16 or fp16')
    for key in ('accumulation','validation_every','validation_subset','broad_every','broad_subset','calibration_count','keep_periodic'):
        if type(out[key])!=int or out[key]<1:raise ValueError(f'{key} must be a positive integer')
    from pdeflow.training_settings import resolve_batches
    batches=resolve_batches(c)
    if 'accumulation' not in c.training_options:out['accumulation']=math.ceil(c.batch_size/(min(c.batch_size,8) if out['microbatch']=='auto' else out['microbatch']))
    base=min(c.batch_size,8) if out['microbatch']=='auto' else out['microbatch']
    if out['accumulation']!=math.ceil(c.batch_size/base):raise ValueError('Accumulation must equal ceil(effective batch / GPU microbatch); final microbatch may be smaller')
    out['microbatch']=base
    if stage is not None:out.update(microbatch=batches[stage]['microbatch'],accumulation=batches[stage]['accumulation'])
    if type(out['max_consecutive_overflows'])!=int or out['max_consecutive_overflows']<1:raise ValueError('Overflow limit must be positive')
    if not math.isfinite(out['initial_scale']) or out['initial_scale']<=0:raise ValueError('Initial scaler scale must be finite positive')
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


def cuda_precision_support(device=None):
    device=device or torch.device('cuda')
    with torch.cuda.device(device):
        try:native=bool(torch.cuda.is_bf16_supported(including_emulation=False))
        except TypeError:native=torch.cuda.get_device_capability(device)[0]>=8
        emulated_or_native=bool(torch.cuda.is_bf16_supported())
    return {'bf16_native':native,'bf16_including_emulation':emulated_or_native,'bf16_emulation_only':emulated_or_native and not native,'fp16':True}


def precision(policy,device):
    if device.type!='cuda':
        if policy not in ('auto','fp32'):raise ValueError('BF16/FP16 training here requires a CUDA device; select fp32 for CPU/MPS')
        return 'fp32',torch.float32
    native=cuda_precision_support(device)['bf16_native']
    if policy=='auto':policy='bf16' if native else 'fp16'
    if policy=='bf16' and not native:raise ValueError('This GPU has no native BF16 support (emulation is not a training policy); select fp16 or fp32')
    return policy,{'fp32':torch.float32,'bf16':torch.bfloat16,'fp16':torch.float16}[policy]


def learning_rate(settings,completed):
    if completed<settings['warmup']:return settings['learning_rate']*(completed+1)/max(1,settings['warmup'])
    if settings['schedule']=='constant':return settings['learning_rate']
    phase=min(1.,max(0.,(completed-settings['warmup'])/max(1,settings['horizon']-settings['warmup'])))
    return settings['learning_rate']*(settings['minimum_ratio']+(1-settings['minimum_ratio'])*(1+math.cos(math.pi*phase))/2)


def scientific_source():
    base=Path(__file__).parents[1]
    names=['coordinate/contracts.py','coordinate/graph.py','coordinate/physics.py','numerics/dynamic.py','coordinate/model.py','coordinate/joint.py','coordinate/consistency.py','coordinate/observation_objective.py','coordinate/evolution_objective.py','coordinate/random_state.py','coordinate/profiling.py','coordinate/training.py','coordinate/engine.py','training_settings.py','coordinate/frozen.py','coordinate/batching.py','coordinate/dataset.py','coordinate/optimization.py','coordinate/temporal.py','heat_continuous.py','heat_placement.py','field_losses.py','numerics/__init__.py','core/__init__.py','equations/__init__.py']
    files={n:hashlib.sha256((base/n).read_bytes()).hexdigest() for n in names}
    return {'files':files,'sha256':digest(files),'policy':'Scientific model/data/loss/numerical source, excluding notebook, docs and unrelated UI'}


def model_contract(c):
    ignored=('animation_frames','times') if c.temporal_mode=='continuous' else ()
    return {**{k:v for k,v in c.to_dict().items() if k not in (*ignored,'runtime','ae_steps','evolution_steps','flow_steps','learning_rate','checkpoint_every','batch_size','training_options')},
            'normalization_calibration_count':min(c.train_count,options(c)['calibration_count'])}


def resume_contract(old,c,stage,extend=False):
    from .contracts import CoordinateConfig
    from .training import checkpoint_config
    prior=checkpoint_config(old)
    differences=[k for k,v in model_contract(c).items() if digest(v)!=digest(model_contract(prior).get(k))]
    if differences:raise ValueError('Incompatible model/data changes: '+', '.join(differences))
    from pdeflow.training_settings import batch_identity
    if batch_identity(prior)!=batch_identity(c):
        authorization=options(c)['batch_change_authorization'] or {}
        if digest(batch_identity(prior)) not in [authorization.get('from'),*authorization.get('from_alternatives',[])] or authorization.get('to')!=digest(batch_identity(c)) or not authorization.get('explicit_continuation'):raise ValueError('Batch changes on resume require explicit apply_training_settings(..., continuation=True)')
    old_total=getattr(prior,{'autoencoder':'ae_steps','evolution':'evolution_steps','flow':'flow_steps'}[stage]);new_total=getattr(c,{'autoencoder':'ae_steps','evolution':'evolution_steps','flow':'flow_steps'}[stage])
    if new_total!=old_total and not (extend and new_total>old_total) and not (options(c)['batch_change_authorization'] or {}).get('explicit_continuation'):raise ValueError('A changed update budget requires deliberate extend mode and a larger total')
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
    value=model_contract(c);value['joint']={k:v for k,v in value['joint'].items() if k not in ('auxiliary','observation_auxiliary','target_cache','noiseless_probability')}
    # A2-only physics changes its optimizer objective, not the frozen A1 state
    # encoder/decoder schema. Strict same-stage resume still compares it above.
    value['evolution']={k:v for k,v in value['evolution'].items() if k!='physics'}
    return value


def gradient_diagnostics(named_parameters):
    """Exceptional-path diagnostics on unscaled, unclipped gradients only."""
    invalid=[];count=0
    for name,p in named_parameters:
        if p.grad is None:continue
        bad=~torch.isfinite(p.grad.detach())
        n=int(bad.sum())
        if n:
            count+=1
            if len(invalid)<12:invalid.append({'parameter':name,'nonfinite_elements':n,'elements':p.grad.numel()})
    return {'nonfinite_parameter_count':count,'first_nonfinite_parameters':invalid,
            'scope':'unscaled gradients before clipping; first 12 invalid parameter tensors'}


def optimizer_step(optimizer,scaler,parameters,clip,policy):
    parameters=list(parameters)
    scaler.unscale_(optimizer)
    try:
        # Raising before clipping leaves invalid gradients intact for diagnostics.
        grad=torch.nn.utils.clip_grad_norm_(parameters,clip,error_if_nonfinite=True)
    except RuntimeError as error:
        if policy!='fp16' or 'non-finite' not in str(error):raise
        if all(bool(torch.isfinite(p.grad).all()) for p in parameters if p.grad is not None):
            # GradScaler would NOT skip this: never step after a norm-only overflow.
            raise FloatingPointError('Nonfinite gradient norm with finite gradient elements; optimizer not stepped')
        grad=torch.tensor(float('inf'))
    old_scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
    finite=math.isfinite(float(grad));skipped=policy=='fp16' and scaler.get_scale()<old_scale
    if not finite and not skipped:raise FloatingPointError('Nonfinite gradients without skipped optimizer update')
    return grad,finite,skipped
