"""Three explicit artifact views over canonical run files, without weight duplication.

A view is a hash manifest, not an extra multi-GB checkpoint copy. Optional ZIPs
are keyed by manifest identity and excluded from routine persistent snapshots.
"""
from pathlib import Path
import json,zipfile
import torch
from pdeflow.runtime import read_json,write_json,file_digest,digest
from pdeflow.coordinate.training import read_checkpoint
from pdeflow.coordinate.joint import ARCHITECTURE,ARCHITECTURES


def _entry(root,p):return {'sha256':file_digest(p),'bytes':p.stat().st_size}


def verify_view(root,kind):
    root=Path(root);record=read_json(root/'bundles'/(kind+'.json'))
    if record['kind']!=kind or record['architecture'] not in ARCHITECTURES:raise ValueError('Artifact schema mismatch')
    for name,e in record['files'].items():
        p=root/name
        if Path(name).is_absolute() or '..' in Path(name).parts or p.is_symlink() or not p.resolve().is_relative_to(root.resolve()):raise ValueError('Unsafe artifact path')
        if p.stat().st_size!=e['bytes'] or file_digest(p)!=e['sha256']:raise ValueError('Artifact checksum mismatch: '+name)
    return record


def build(root,standalone_check=True):
    root=Path(root);stages={};canonical=set()
    for stage in ('autoencoder','evolution','flow'):
        path=root/stage/'resume.pt'
        if not path.exists():continue
        cp,c=read_checkpoint(path,stage)
        required=('optimizer','scheduler','scaler','rng','torch_rng','cuda_rng','ambient_rng','dataset_identity','training_ids','normalization_fitted_ids','effective_examples_processed','attempted_updates','skipped_updates')
        missing=[name for name in required if name not in cp]
        if missing:raise ValueError('Not a complete new resumable checkpoint ('+stage+'): '+', '.join(missing))
        if cp['architecture'] not in ARCHITECTURES or not any(k.startswith('encoder.') for k in cp['autoencoder']):raise ValueError('Exact state encoder missing: '+stage)
        if cp['scheduler']['completed_updates']!=cp['progress'] or cp['attempted_updates']!=cp['progress']+cp['skipped_updates']:raise ValueError('Stage progress/counter mismatch')
        dependency=cp.get('dependency')
        if dependency:
            target=(path.parent/dependency['path']).resolve()
            if not target.is_relative_to(root.resolve()) or file_digest(target)!=dependency['sha256']:raise ValueError('Missing or external frozen dependency; copy its canonical stage into the run before exporting')
            canonical.add(target)
            prior,_=read_checkpoint(target,dependency['stage'])
            if prior['dataset_identity']!=cp['dataset_identity'] or prior['normalizations']!=cp['normalizations']:raise ValueError('Stage dataset/normalization dependency mismatch')
            for name,value in prior['autoencoder'].items():
                if (stage=='flow' or name.startswith(('encoder.','decoder.','coefficient.'))) and not torch.equal(value,cp['autoencoder'][name]):raise ValueError('Frozen representation differs from dependency: '+name)
        stages[stage]={'successful_updates':cp['progress'],'attempted_updates':cp['attempted_updates'],'skipped_updates':cp['skipped_updates'],'completed':cp['completed'],'checkpoint':path.relative_to(root).as_posix(),'checkpoint_sha256':file_digest(path),'dependency':dependency,'scientific_source':cp['scientific_source']['sha256']}
        canonical.add(path)
        if standalone_check:
            # Check shapes/keys without allocating another complete model copy.
            from pdeflow.coordinate.model import Autoencoder,Velocity
            with torch.device('meta'):
                ae=Autoencoder(c);velocity=Velocity(c) if stage=='flow' else None
            ae.load_state_dict(cp['autoencoder'],strict=True,assign=True)
            if velocity is not None:velocity.load_state_dict(cp['velocity'],strict=True,assign=True)
            del ae,velocity
        for pattern in ('*.json','*-preview.npz','history-live.jsonl'):
            for p in (root/stage).glob(pattern):canonical.add(p)
        exported=root/stage/('autoencoder.pt' if stage=='autoencoder' else 'evolution.pt' if stage=='evolution' else 'coordinate.pt')
        if exported.exists():canonical.add(exported)
    if not stages:raise ValueError('No valid stage checkpoint yet; resolved configuration/source snapshot is available, no trained-model artifact exists')
    source=root/'source/source.zip';identity=root/'source/identity.json'
    if not source.exists() or not identity.exists() or file_digest(source)!=read_json(identity)['sha256']:raise ValueError('Exact source package is missing/corrupt')
    from pdeflow.source_bundle import verify as verify_source
    verify_source(source)
    canonical.update((source,identity))
    if (root/'frozen-source/source.zip').exists():canonical.add(root/'frozen-source/source.zip')
    for p in root.glob('*.json'):
        if p.name not in ('sync-status.json','stop-request.json'):canonical.add(p)
    # No untouched test set is exposed for tuning. These fixed validation seeds
    # are procedural identities, not predictions or a scientific-quality claim.
    cases=root/'diagnostic-cases.json'
    if not cases.exists():write_json(cases,{'split':'validation','indices':[0,1],'meaning':'Reproducible procedural validation cases; full recipe/split identity in stage config and dataset metadata'})
    canonical.add(cases)
    inference=root/'export/coordinate.pt';inference_files={inference,inference.with_suffix('.json')} if inference.exists() else set()
    generative='flow' in stages and stages['flow']['successful_updates']>0
    if generative and not inference_files:raise ValueError('Export standalone inference first')
    if standalone_check:
        if generative:
            from pdeflow.coordinate.inference import CoordinateBundle
            CoordinateBundle(inference,device='cpu')
    common={'format':1,'architecture':cp['architecture'],'stages':stages,'generative_model_available':generative,'incomplete_stages':[s for s in ('autoencoder','evolution','flow') if s not in stages or not stages[s]['completed']],
            'accuracy':'Not established by execution or a baseline gate','resume':'Functional replay on compatible source/config; exact CPU replay tested separately. Cross-device/library bitwise replay is not promised.'}
    views={}
    for kind,paths in (('inference',inference_files),('diagnostic',canonical|inference_files),('resumable',canonical|inference_files)):
        record={**common,'kind':kind,'available':bool(paths) if kind=='inference' else True,'files':{p.relative_to(root).as_posix():_entry(root,p) for p in sorted(paths)}}
        write_json(root/'bundles'/(kind+'.json'),record);views[kind]=verify_view(root,kind)
    return views


def load_diagnostic(root,device='cpu'):
    record=verify_view(root,'diagnostic')
    if not record['generative_model_available']:raise ValueError('Partial representation/evolution bundle has no conditional generator yet')
    from pdeflow.coordinate.inference import CoordinateBundle
    return CoordinateBundle(Path(root)/record['stages']['flow']['checkpoint'],device=device,allow_partial=True)


def download_archive(root,kind):
    root=Path(root);record=verify_view(root,kind);identity=digest(record)
    if not record['available']:raise ValueError('Artifact is not available for this partial run: '+kind)
    path=root/'downloads'/(kind+'-'+identity[:16]+'.zip');path.parent.mkdir(exist_ok=True)
    if not path.exists():
        with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_STORED) as archive:
            for name in record['files']:archive.write(root/name,name)
            archive.write(root/'bundles'/(kind+'.json'),'bundles/'+kind+'.json')
    with zipfile.ZipFile(path) as archive:
        if archive.testzip():raise ValueError('Artifact ZIP verification failed')
    return path
