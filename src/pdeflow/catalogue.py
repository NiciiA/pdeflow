"""Versioned finite training targets, independent of physical parameter values."""
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from pdeflow import local
from pdeflow.runtime import digest

VERSION='pdeflow-training-v2'
TARGETS=('heat-2d','darcy-2d','chemistry-2d','heat-3d','darcy-3d','chemistry-3d',
         'heat-2d-medium','heat-2d-large','heat-2d-512','heat-2d-1024-cuda','colab-heat-2d')

def signature(config):
    """Capability/distribution/optimizer identity; budget, seeds and device separate.

    Source changes do not invalidate a safely loadable completed inference export.
    Resuming an optimizer still obeys the trainer's strict source identity.
    """
    c=config.to_dict() if hasattr(config,'to_dict') else __import__('copy').deepcopy(config)
    c.pop('runtime',None)
    c.pop('inference',None) # sampler/guidance settings do not train a different prior
    for name in ('seed','steps','checkpoint_every'):c['training'].pop(name,None)
    c['data'].pop('seed',None)
    return digest(c)

def capability_signature(config):
    c=config.to_dict() if hasattr(config,'to_dict') else config
    data={k:c['data'].get(k) for k in ('bandwidth','bandwidth_range','amplitude_range','offset_range','diffusivity_range')}
    return digest({k:c.get(k) for k in ('specialist','formulation','grid','model','chemistry')}|{'distribution':data,'distribution_version':(c.get('provider') or {}).get('distribution_version')})

@lru_cache(maxsize=1)
def catalogue():
    from pdeflow.models import TimeChannelLayout
    from pdeflow.models import make_model,architecture_metadata
    from pdeflow.specialists import get_specialist
    import torch
    rows=[]
    # fork_rng prevents catalogue inspection from changing later model initialization.
    with torch.random.fork_rng():
        for index,name in enumerate(TARGETS):
            c=local.preset_config(name);s=get_specialist(c.specialist);layout=TimeChannelLayout(len(s.fields),c.grid)
            architecture=architecture_metadata(make_model(c.model,layout,s.encoder),c.model,layout)
            rows.append({'id':name,'catalogue_version':VERSION,'index':index,'specialist':s.id,'formulation':c.formulation,
                'generative_method':'conditional flow matching','representation':'direct physical fields with saved transform; no latent decoder',
                'architecture':architecture,'grid':asdict(c.grid),'conditioning':s.conditioning_schema,'distribution_version':c.provider.distribution_version,
                'training_distribution':{'data':asdict(c.data),'chemistry':asdict(c.chemistry) if c.chemistry else None},
                'updates':c.training.steps,'batch_size':c.training.batch_size,'data_mode':c.provider.mode,'resource_profile':asdict(c.runtime),
                'signature':signature(c),'capability_signature':capability_signature(c),'expected_capability':s.capabilities,'command':f'pdeflow train --preset {name}',
                'resolved_config':c.to_dict()})
    return rows

@lru_cache(maxsize=128)
def _export(path,size,mtime):
    from pdeflow.training import _read_checkpoint
    checkpoint,c,_=_read_checkpoint(path)
    import torch
    from pdeflow.models import TimeChannelLayout,make_model,architecture_metadata
    from pdeflow.specialists import get_specialist
    if checkpoint['kind']!='inference':raise ValueError('Not an inference export')
    specialist=get_specialist(c.specialist);layout=TimeChannelLayout(len(specialist.fields),c.grid)
    # Metadata/shape validation must never initialize a CPU thread pool or copy
    # full models during live polling. Meta tensors use the actual model factory.
    with torch.device('meta'):
        model=make_model(c.model,layout,specialist.encoder)
    expected=model.state_dict();actual=checkpoint['model']
    if set(expected)!=set(actual) or any(not isinstance(actual[k],torch.Tensor) or actual[k].shape!=expected[k].shape for k in expected):raise ValueError('Checkpoint state_dict architecture/shape mismatch')
    architecture=architecture_metadata(model,c.model,layout)
    if checkpoint.get('architecture') and checkpoint['architecture']!=architecture:raise ValueError('Checkpoint architecture metadata/parameter count mismatch')
    return {'progress':checkpoint['progress'],'capability_signature':capability_signature(c),'signature':signature(c),'capabilities':checkpoint['capabilities'],'architecture':architecture}

def export_record(path):
    p=Path(path);st=p.stat();return _export(str(p),st.st_size,st.st_mtime_ns)

def successful_run(record,config,updates):
    try:
        if record['state']!='completed' or record['progress']<updates or record['budget']<updates:return False
        if signature(record['resolved_config'])!=signature(config):return False
        p=Path(record['path'])/record['paths']['latest_export'];export=export_record(p)
        return export['progress']>=updates and export['signature']==signature(config)
    except (OSError,ValueError,RuntimeError,KeyError,EOFError):return False

def coverage(workspace=None):
    runs=local.run_records(workspace);rows=[]
    for target in catalogue():
        matching=[r for r in runs if capability_signature(r['resolved_config'])==target['capability_signature']]
        completed=[r for r in matching if successful_run(r,target['resolved_config'],target['updates'])]
        available=[]
        for r in matching:
            try:
                path=Path(r['path'])/r['paths']['latest_export'];export_record(path);available.append(str(path))
            except (OSError,ValueError,RuntimeError,KeyError,EOFError):pass
        evaluated=[]
        for r in matching:
            from pdeflow.runtime import read_json
            if any(read_json(p).get('state')=='completed' for p in (Path(r['path'])/'evaluations').glob('*/evaluation.json')):evaluated.append(r['id'])
        for entry in local.model_entries(workspace):
            try:
                if export_record(entry['path'])['capability_signature']==target['capability_signature'] and entry['path'] not in available:available.append(entry['path'])
            except (OSError,ValueError,RuntimeError,KeyError,EOFError):pass
        rows.append({**target,'state':'completed' if completed else 'partial' if matching else 'no_training_run',
            'configured':True,'completed_run_ids':[r['id'] for r in completed],'run_ids':[r['id'] for r in matching],'run_states':[{'id':r['id'],'state':local.status(r['path'])['state'],'progress':r['progress'],'budget':r['budget']} for r in matching],
            'maximum_updates':max([r['progress'] for r in matching],default=0),'evaluation_available':bool(evaluated),
            'evaluated_run_ids':evaluated,'compatible_discoverable_checkpoint':bool(available),'export_paths':available,
            'quality':'Training completion and checkpoint availability do not establish scientific quality.'})
    obsolete=[{'id':r['id'],'preset':r['preset'],'reason':'Recorded architecture/distribution/layout differs from the current named target; export can still serve its own exact compatible requests'} for r in runs if r['preset'] in TARGETS and capability_signature(r['resolved_config'])!=next(t['capability_signature'] for t in catalogue() if t['id']==r['preset'])]
    return {'catalogue_version':VERSION,'obsolete_targets':obsolete,'targets':rows,'unsupported':'Layouts/methods outside this finite catalogue require an explicit compatible configuration or implementation.'}
