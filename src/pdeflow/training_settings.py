"""Shared batch validation, budget conversion and explicit Session application."""
from dataclasses import replace
from pathlib import Path
import math,time,uuid
from pdeflow.runtime import digest,read_json,write_json

STAGES=('autoencoder','evolution','flow')
LABELS={'autoencoder':'A1 representation','evolution':'A2 evolution','flow':'B conditional flow'}
ALIASES={'A1':'autoencoder','A2':'evolution','B':'flow',**{s:s for s in STAGES}}
TOTALS={'autoencoder':'ae_steps','evolution':'evolution_steps','flow':'flow_steps'}

def positive(value,name):
    if type(value)!=int or value<1:raise ValueError(name+' must be a positive integer')
    return value

def micro_value(value):
    if value=='auto':return value
    return positive(value,'GPU microbatch')

def resolve_batches(c):
    raw=c.training_options;effective=positive(c.batch_size,'Effective batch');base=micro_value(raw.get('microbatch',effective));overrides=raw.get('stage_batches',{})
    if not isinstance(overrides,dict) or set(overrides)-set(STAGES):raise ValueError('Stage batch overrides use autoencoder, evolution, flow')
    result={}
    for stage in STAGES:
        override=overrides.get(stage,{})
        if not isinstance(override,dict) or set(override)-{'microbatch'}:raise ValueError('Per-stage batch override supports microbatch only; accumulation is derived')
        requested=micro_value(override.get('microbatch',base));micro=min(effective,8) if requested=='auto' else requested
        if micro>effective:raise ValueError('GPU microbatch cannot exceed effective batch; accumulation cannot create a smaller optimizer batch')
        count=math.ceil(effective/micro);last=effective-(count-1)*micro
        result[stage]={'stage':stage,'label':LABELS[stage],'requested_microbatch':requested,'microbatch':micro,'accumulation':count,'effective_batch':effective,'last_microbatch':last,
          'resolution':'Conservative unqualified Auto: min(effective batch,8); run a benchmark to qualify hardware.' if requested=='auto' else 'Explicit/applied GPU microbatch; hardware fit requires measurement.',
          'weighting':'Sum individual losses / effective batch; final incomplete microbatch uses its actual example count.','planned_successful_updates':getattr(c,TOTALS[stage]),'approximate_examples':getattr(c,TOTALS[stage])*effective}
    return result


def batch_identity(c):return {s:{k:v for k,v in r.items() if k in ('microbatch','effective_batch','accumulation')} for s,r in resolve_batches(c).items()}


def configure(c,settings,budget_policy='keep_updates',progress=None):
    if not isinstance(settings,dict) or set(settings)-{'effective_batch','microbatch','stages'}:raise ValueError('Batch settings support effective_batch, microbatch and stages')
    if budget_policy not in ('keep_updates','keep_examples'):raise ValueError('Choose keep_updates or keep_examples explicitly')
    before=resolve_batches(c);effective=positive(settings.get('effective_batch',c.batch_size),'Effective batch');base=micro_value(settings.get('microbatch',c.training_options.get('microbatch',c.batch_size)))
    stage_values=settings.get('stages',c.training_options.get('stage_batches',{}));normalized={}
    if not isinstance(stage_values,dict):raise ValueError('Stages must be a mapping of stage to GPU microbatch')
    for key,value in stage_values.items():
        if key not in ALIASES:raise ValueError('Unknown training stage: '+str(key))
        if isinstance(value,dict):
            if set(value)-{'microbatch'}:raise ValueError('Stage batch override supports microbatch only')
            value=value.get('microbatch',base)
        normalized[ALIASES[key]]={'microbatch':micro_value(value)}
    global_micro=min(effective,8) if base=='auto' else base
    if global_micro>effective:raise ValueError('GPU microbatch cannot exceed effective batch')
    training={**c.training_options,'microbatch':base,'accumulation':math.ceil(effective/global_micro),'stage_batches':normalized}
    fields={};rounding={};progress=progress or {}
    for stage,name in TOTALS.items():
        old=getattr(c,name);done=progress.get(stage,0)
        total=max(done,math.ceil(max(0,old-done)*c.batch_size/effective)+done) if budget_policy=='keep_examples' else old
        total=max(2 if stage=='flow' else 1,total)
        fields[name]=total;rounding[stage]={'old_remaining_examples':max(0,old-done)*c.batch_size,'new_remaining_examples':max(0,total-done)*effective,'examples_delta':max(0,total-done)*effective-max(0,old-done)*c.batch_size,'rounding_extra_examples':max(0,total-done)*effective-max(0,old-done)*c.batch_size if budget_policy=='keep_examples' else 0,'minimum_stage_updates':2 if stage=='flow' else 1}
        # Keep the schedule horizon and LR, including when budget conversion
        # changes update count. No implicit LR or schedule scaling.
        if progress:
            from pdeflow.coordinate.optimization import options
            training[stage]={**training.get(stage,{}),'horizon':options(c)[stage]['horizon']}
    new=replace(c,batch_size=effective,training_options=training,**fields).validate();after=resolve_batches(new)
    return new,{'before':before,'after':after,'budget_policy':budget_policy,'rounding':rounding,'learning_rate_unchanged':True,'state':'applied; training has not started','exact_reproducibility':'Batch changes preserve the loss estimator, not bitwise-identical optimization trajectories.'}


def apply_training_settings(target,settings,budget_policy='keep_updates',continuation=False,selection=None):
    """Return replacement Session, or (config,report) for the UI's authoritative draft.

    All callers share configure/validation. Session replacement retains its identity
    and storage. Existing provenance is archived; application never launches work.
    """
    from pdeflow.coordinate.contracts import CoordinateConfig
    if isinstance(target,CoordinateConfig):
        new,report=configure(target,settings,budget_policy)
        report['selection']=selection or {'method':'manual settings','hardware_qualification':False}
        new=replace(new,training_options={**new.training_options,'batch_selection':report['selection']}).validate()
        return new,report
    from pdeflow.colab import Session
    from pdeflow.local import run_lock,is_locked
    session=target
    if not isinstance(session,Session):raise TypeError('Expected Session or CoordinateConfig')
    if is_locked(session.root):raise ValueError('Cannot apply settings to an active training worker; stop at an optimizer boundary first')
    with run_lock(session.root):
        prior=session.root/'session.json';started=prior.exists() or any((session.root/s/'resume.pt').exists() for s in STAGES)
        if started and not continuation:raise ValueError('Existing run: choose explicit continuation to change batch settings; original provenance is preserved')
        progress={s:read_json(session.root/s/'run.json').get('progress',0) for s in STAGES if (session.root/s/'run.json').exists()}
        new,report=configure(session.c,settings,budget_policy,progress)
        report['selection']=selection or {'method':'manual settings','hardware_qualification':False}
        new=replace(new,training_options={**new.training_options,'batch_selection':report['selection']}).validate()
        if started:
            from .coordinate.contracts import CoordinateConfig
            from .coordinate.optimization import model_contract
            prior_config=CoordinateConfig(**read_json(prior)['config']).validate() if prior.exists() else session.c
            if model_contract(prior_config)!=model_contract(new):raise ValueError('Architecture/data/loss changes are incompatible with batch continuation')
            alternatives=[]
            for stage in STAGES:
                path=session.root/stage/'run.json'
                if path.exists():alternatives.append(digest(batch_identity(CoordinateConfig(**read_json(path)['config']).validate())))
            authorization={'from':digest(batch_identity(session.c)),'to':digest(batch_identity(new)),'budget_policy':budget_policy,'explicit_continuation':True,'from_alternatives':alternatives}
            new=replace(new,training_options={**new.training_options,'batch_change_authorization':authorization}).validate()
        identifier=time.strftime('%Y%m%dT%H%M%S',time.gmtime())+'-'+uuid.uuid4().hex[:8]
        history=session.root/'settings-history';history.mkdir(exist_ok=True)
        write_json(history/(identifier+'.json'),{'old_config':session.c.to_dict(),'new_config':new.to_dict(),'report':report,'continuation':started,'applied_unix':time.time()})
        for name in ('profile.json',):
            path=session.root/name
            if path.exists():path.rename(history/(identifier+'-'+name))
        # This file, rather than an arbitrary notebook binding, owns the next run.
        write_json(session.root/'applied-settings.json',{'config':new.to_dict(),'config_identity':digest(new.to_dict()),'application':report,'continuation':started,'history_id':identifier})
        write_json(session.root/'resolved-config.json',new.to_dict())
        replacement=Session(new,session.name,session.root.parent,session.persistent.parent,session.allow_cpu_check,session.sync_seconds,session.keep_snapshots)
        replacement.last_sync=session.last_sync;replacement.snapshot=session.snapshot
        return replacement
