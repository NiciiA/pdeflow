"""Verified content-addressed snapshots for mounted Drive or a local persistent dir.

No reliance on Drive rename atomicity: recovery accepts only a complete manifest
whose body hash, committed marker and EVERY content object verify.
"""
from pathlib import Path
import json,uuid,time,shutil,os
from pdeflow.runtime import read_json,write_json,file_digest,digest

def _safe(root,relative):
    p=root/relative
    if Path(relative).is_absolute() or '..' in Path(relative).parts or any(a.is_symlink() for a in [p,*p.parents]) or not p.resolve().is_relative_to(root.resolve()):raise ValueError('Unsafe snapshot path')
    return p


def latest(root):
    root=Path(root);valid=[]
    try:
        if read_json(root/'.pdeflow-store.json').get('kind')!='pdeflow_coordinate_snapshots':return None
    except (OSError,ValueError):return None
    for path in (root/'snapshots').glob('*.json'):
        try:
            record=read_json(path)
            if path.with_suffix('.commit').read_text()!=digest(record):continue
            for name,entry in record['files'].items():
                _safe(Path('/'),name) if Path(name).is_absolute() else None
                obj=_safe(root,'objects/'+entry['sha256'])
                if len(entry['sha256'])!=64 or obj.stat().st_size!=entry['bytes'] or file_digest(obj)!=entry['sha256']:raise ValueError('Incomplete object')
                if '..' in Path(name).parts or Path(name).is_absolute():raise ValueError('Unsafe snapshot relative path')
            valid.append(record)
        except (OSError,ValueError,KeyError,TypeError):continue
    return max(valid,key=lambda r:r['created_unix']) if valid else None


def publish(local,persistent,keep=3):
    if type(keep)!=int or not 1<=keep<=20:raise ValueError('Keep 1–20 complete persistent snapshots')
    local=Path(local);root=Path(persistent);_safe(root,'.pdeflow-store.json');root.mkdir(parents=True,exist_ok=True)
    marker=root/'.pdeflow-store.json'
    if not marker.exists():
        if any(root.iterdir()):raise ValueError('Persistent run directory is not an owned PDEFlow snapshot store; choose a new run name')
        write_json(marker,{'kind':'pdeflow_coordinate_snapshots','schema':1,'id':uuid.uuid4().hex})
    elif read_json(marker).get('kind')!='pdeflow_coordinate_snapshots':raise ValueError('Unrecognized persistent store; no files changed')
    _safe(root,'objects').mkdir(parents=True,exist_ok=True);_safe(root,'snapshots').mkdir(exist_ok=True)
    names=[]
    for p in local.rglob('*'):
        if not p.is_file() or p.is_symlink() or any(x in ('checkpoints','profile-cache','__pycache__') for x in p.relative_to(local).parts):continue
        # Only this workflow's artifacts, not arbitrary data or notebook files.
        if p.suffix in ('.pt','.json','.npz','.png') and p.name!='dataset.npz':names.append(p.relative_to(local).as_posix())
    files={}
    for name in names:
        p=_safe(local,name);sha=file_digest(p);obj=_safe(root,'objects/'+sha)
        if not obj.exists() or obj.stat().st_size!=p.stat().st_size or file_digest(obj)!=sha:
            temporary=_safe(root,'objects/.upload-'+uuid.uuid4().hex)
            shutil.copyfile(p,temporary)
            if file_digest(temporary)!=sha:raise ValueError('Persistent upload integrity failed; previous committed snapshot retained')
            # Object publication may be non-atomic on Drive. The commit is last.
            shutil.copyfile(temporary,obj)
            if file_digest(obj)!=sha:raise ValueError('Persistent object incomplete; previous snapshot retained')
            temporary.unlink()
        files[name]={'sha256':sha,'bytes':p.stat().st_size}
    record={'id':uuid.uuid4().hex,'created_unix':time.time(),'files':files,'unique_bytes':sum(e['bytes'] for e in {v['sha256']:v for v in files.values()}.values()),'progress':read_json(local/'session.json') if (local/'session.json').exists() else None}
    manifest=_safe(root,'snapshots/'+record['id']+'.json');write_json(manifest,record)
    if read_json(manifest)!=record:raise ValueError('Persistent manifest integrity failed')
    manifest.with_suffix('.commit').write_text(digest(record))
    if latest(root) is None:raise ValueError('No complete persistent checkpoint; last good snapshot not replaced')
    # Retention is scoped ONLY to this run's content-addressed snapshot store.
    committed=[]
    for p in (root/'snapshots').glob('*.json'):
        try:
            r=read_json(p)
            if p.with_suffix('.commit').read_text()==digest(r):committed.append((p,r['created_unix']))
        except (OSError,ValueError,KeyError):continue
    committed=[p for p,_ in sorted(committed,key=lambda v:v[1],reverse=True)]
    for p in committed[keep:]:p.with_suffix('.commit').unlink(missing_ok=True);p.unlink()
    referenced=set()
    for p in (root/'snapshots').glob('*.json'):
        try:referenced.update(e['sha256'] for e in read_json(p)['files'].values())
        except (ValueError,KeyError):pass
    for p in (root/'objects').iterdir():
        if p.name not in referenced and len(p.name)==64 and all(x in 'abcdef0123456789' for x in p.name):p.unlink()
    return record


def restore(persistent,local):
    persistent=Path(persistent);local=Path(local);record=latest(persistent)
    if not record:raise ValueError('No complete verified persistent snapshot. Incomplete uploads are ignored; never restarting silently.')
    for name,e in record['files'].items():
        target=_safe(local,name);target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and file_digest(target)!=e['sha256']:
            if name in ('profile.json','sync-status.json'):continue # current runtime reports are not training state
            raise ValueError('Local file differs from persistent checkpoint; use a new empty local run directory: '+name)
        if not target.exists():shutil.copyfile(_safe(persistent,'objects/'+e['sha256']),target)
        if file_digest(target)!=e['sha256']:raise ValueError('Restored file checksum mismatch')
    return record
