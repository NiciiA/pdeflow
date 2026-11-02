"""Verified, content-addressed safe-boundary snapshots on local/mounted storage.

Drive rename is not assumed atomic. Objects -> manifest -> verified completion
marker -> latest pointer LAST. Recovery always scans complete manifests, so a
stale pointer or interrupted upload cannot replace the last verified snapshot.
"""
from pathlib import Path
import json,uuid,time,shutil
from pdeflow.runtime import read_json,write_json,file_digest,digest


def _safe(root,relative):
    p=root/relative
    if Path(relative).is_absolute() or '..' in Path(relative).parts or any(a.is_symlink() for a in [p,*p.parents]) or not p.resolve().is_relative_to(root.resolve()):raise ValueError('Unsafe snapshot path')
    return p


def probe(root):
    """Non-destructive write/read/delete probe before training; no queued success."""
    parent=Path(root).parent;parent.mkdir(parents=True,exist_ok=True)
    path=parent/('.pdeflow-probe-'+uuid.uuid4().hex);payload=uuid.uuid4().bytes
    try:
        path.write_bytes(payload)
        if path.read_bytes()!=payload:raise OSError('Persistent storage write/read probe mismatch')
    finally:path.unlink(missing_ok=True)
    return {'verified':True,'parent':str(parent),'time':time.time()}


def verify(root,identifier):
    root=Path(root)
    if not isinstance(identifier,str) or len(identifier)!=32 or not identifier.isalnum():raise ValueError('Invalid snapshot ID')
    path=_safe(root,'snapshots/'+identifier+'.json');record=read_json(path)
    if record['id']!=identifier or path.with_suffix('.commit').read_text()!=digest(record):raise ValueError('Missing/invalid completion marker')
    for name,entry in record['files'].items():
        _safe(root,name)  # validate logical destination independently of objects
        sha=entry['sha256']
        if len(sha)!=64 or any(x not in '0123456789abcdef' for x in sha):raise ValueError('Invalid object identity')
        obj=_safe(root,'objects/'+sha)
        if obj.stat().st_size!=entry['bytes'] or file_digest(obj)!=sha:raise ValueError('Incomplete/corrupt object: '+name)
    return record


def snapshots(root):
    root=Path(root)
    try:
        if read_json(root/'.pdeflow-store.json').get('kind')!='pdeflow_coordinate_snapshots':return []
    except (OSError,ValueError):return []
    valid=[]
    for path in (root/'snapshots').glob('*.json'):
        try:valid.append(verify(root,path.stem))
        except (OSError,ValueError,KeyError,TypeError):continue
    return sorted(valid,key=lambda r:(r.get('sequence',0),r['created_unix']),reverse=True)


def latest(root):
    records=snapshots(root)
    return records[0] if records else None


def _names(local):
    for p in local.rglob('*'):
        rel=p.relative_to(local)
        if not p.is_file() or p.is_symlink() or any(x in ('checkpoints','profile-cache','latent-targets','__pycache__','downloads') for x in rel.parts):continue
        if p.name in ('sync-status.json','stop-request.json'):continue
        if 'batch-experiments' in rel.parts and p.suffix in ('.pt','.npz','.npy','.png'):continue
        if (p.suffix in ('.pt','.json','.npz','.png') and p.name!='dataset.npz') or (len(rel.parts)>=2 and rel.parts[-2:] in (('source','source.zip'),('frozen-source','source.zip'))) or (p.name=='parent-source.zip' and 'continuation' in rel.parts) or (p.name=='history-live.jsonl' and (len(rel.parts)==2 and rel.parts[0] in ('autoencoder','evolution','flow') or 'stages' in rel.parts)):yield rel.as_posix()


def publish(local,persistent,keep=3,tag=None):
    """Synchronous call only at a safe trainer boundary; no background copy queue."""
    if type(keep)!=int or not 1<=keep<=20:raise ValueError('Keep 1–20 complete persistent snapshots')
    local=Path(local);root=Path(persistent);_safe(root,'.pdeflow-store.json');root.mkdir(parents=True,exist_ok=True)
    marker=root/'.pdeflow-store.json'
    if not marker.exists():
        if any(root.iterdir()):raise ValueError('Persistent run directory is not an owned PDEFlow snapshot store; choose a new run ID')
        write_json(marker,{'kind':'pdeflow_coordinate_snapshots','schema':2,'id':uuid.uuid4().hex})
    elif read_json(marker).get('kind')!='pdeflow_coordinate_snapshots':raise ValueError('Unrecognized persistent store; no files changed')
    _safe(root,'objects').mkdir(parents=True,exist_ok=True);_safe(root,'snapshots').mkdir(exist_ok=True)
    files={};uploaded=0
    for name in sorted(_names(local)):
        p=_safe(local,name);size=p.stat().st_size;sha=file_digest(p);obj=_safe(root,'objects/'+sha)
        if not obj.exists() or obj.stat().st_size!=size or file_digest(obj)!=sha:
            temporary=_safe(root,'objects/.upload-'+uuid.uuid4().hex)
            try:
                shutil.copyfile(p,temporary)
                if temporary.stat().st_size!=size or file_digest(temporary)!=sha or file_digest(p)!=sha:raise ValueError('Source changed/upload integrity failed; previous committed snapshot retained')
                shutil.copyfile(temporary,obj)
                if obj.stat().st_size!=size or file_digest(obj)!=sha:raise ValueError('Persistent object incomplete; previous snapshot retained')
                uploaded+=size
            finally:temporary.unlink(missing_ok=True)
        files[name]={'sha256':sha,'bytes':size}
    prior=latest(root)
    if prior and prior['files']==files and (tag is None or prior.get('tag')==tag):return prior
    progress=read_json(local/'session.json') if (local/'session.json').exists() else None
    record={'id':uuid.uuid4().hex,'sequence':prior.get('sequence',0)+1 if prior else 1,'parent':prior['id'] if prior else None,'created_unix':time.time(),'files':files,'tag':tag,
            'unique_bytes':sum(e['bytes'] for e in {v['sha256']:v for v in files.values()}.values()),'uploaded_bytes':uploaded,'progress':progress,
            'stage':(progress or {}).get('stage'),'successful_update':(progress or {}).get('progress',0),'source_identity':(progress or {}).get('scientific_source',{}).get('sha256'),'config_identity':digest((progress or {}).get('config'))}
    manifest=_safe(root,'snapshots/'+record['id']+'.json');write_json(manifest,record)
    if read_json(manifest)!=record:raise ValueError('Persistent manifest integrity failed')
    manifest.with_suffix('.commit').write_text(digest(record))
    verify(root,record['id'])  # verify THIS snapshot, never merely any older one
    write_json(root/'latest.json',{'id':record['id'],'manifest_sha256':digest(record)})
    if read_json(root/'latest.json')['id']!=record['id']:raise ValueError('Latest pointer readback failed; complete snapshot remains discoverable by scanning')
    # Owned run only. Final/best snapshots are pinned; all their dependencies
    # remain referenced objects. No other user run is inspected or cleaned.
    committed=snapshots(root);retained={r['id'] for r in committed[:keep]}|{r['id'] for r in committed if r.get('tag') in ('best','final')}
    for r in committed:
        if r['id'] not in retained:
            p=root/'snapshots'/r['id'];p.with_suffix('.commit').unlink(missing_ok=True);p.with_suffix('.json').unlink()
    referenced=set()
    for p in (root/'snapshots').glob('*.json'):
        try:referenced.update(e['sha256'] for e in read_json(p)['files'].values())
        except (OSError,ValueError,KeyError):continue
    for p in (root/'objects').iterdir():
        if p.name not in referenced and len(p.name)==64 and all(x in 'abcdef0123456789' for x in p.name):p.unlink()
    return record


def restore(persistent,local,identifier=None):
    persistent=Path(persistent);local=Path(local)
    record=verify(persistent,identifier) if identifier else latest(persistent)
    if not record:raise ValueError('No complete verified persistent snapshot. Incomplete uploads are ignored; never restarting silently.')
    # Preflight ALL conflicts before writing anything. Restore into a fresh root
    # if local work is newer/different; never replace a recorded source snapshot.
    targets=[]
    for name,e in record['files'].items():
        target=_safe(local,name)
        if target.exists() and file_digest(target)!=e['sha256']:
            if name in ('profile.json','sync-status.json'):continue
            raise ValueError('Local file differs from persistent checkpoint; use a new empty local run directory: '+name)
        targets.append((target,e))
    for target,e in targets:
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():shutil.copyfile(_safe(persistent,'objects/'+e['sha256']),target)
        if target.stat().st_size!=e['bytes'] or file_digest(target)!=e['sha256']:raise ValueError('Restored file checksum mismatch')
    return record
