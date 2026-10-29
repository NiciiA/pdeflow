"""Rebuildable stat inventory and reviewed, reversible cache quarantine.

Scientific artifacts and unknown user files are always protected. This module
never permanently deletes files. Symlinks are inventoried without following them.
"""
from pathlib import Path
import json,os,subprocess,uuid,time,hashlib
from pdeflow.runtime import read_json,write_json,digest
from pdeflow.local import now,project_root,workspace_root,is_locked


def managed_root(root=None):return Path(root).expanduser().absolute() if root else project_root()/'outputs'
def _inside(root,relative):
    p=root/relative
    if Path(relative).is_absolute() or '..' in Path(relative).parts or not p.resolve().is_relative_to(root.resolve()):raise ValueError('Path escapes configured managed root')
    if any(a.is_symlink() for a in [p,*p.parents] if a!=root.parent):raise ValueError('Symlink paths cannot be mutated')
    return p

def _kind(p):
    if p.is_symlink():return 'symlink'
    n=p.name;parts=set(p.parts)
    if '__pycache__' in parts and p.suffix=='.pyc':return 'temporary_cache'
    if ('cache' in parts or 'matplotlib' in parts) and n.startswith('fontlist-') and p.suffix=='.json':return 'temporary_cache'
    if n.endswith('.whl'):return 'wheel'
    if n.endswith('.zip'):return 'source_archive' if 'source' in n else 'archive'
    if p.suffix=='.pt':return 'resume_checkpoint' if n in ('checkpoint.pt','resume.pt') else 'inference_export'
    if n.startswith('generation.') or n.startswith('particle'):return 'sampler_snapshot'
    if p.suffix in ('.png','.jpg','.webp'):return 'qa_image' if 'browser' in parts else 'figure'
    if 'reference' in n or 'reference' in parts:return 'reference'
    if 'observations' in n or 'samplings' in parts:return 'observations'
    if n=='suite.json':return 'suite_manifest'
    if n=='run.json':return 'run_manifest'
    if 'evaluations' in parts:return 'evaluation'
    if n in ('result.npz','metrics.json') or 'reconstruction' in parts:return 'reconstruction'
    return 'other_protected'

def inventory(root=None,refresh=False):
    root=managed_root(root)
    if root.is_symlink():raise ValueError('Managed root cannot be a symlink')
    meta=root/'.storage';cache=meta/'inventory.json'
    if cache.is_file() and not refresh:
        cached=read_json(cache)
        if cached.get('schema_version')==2 and not cached.get('invalidated'):return cached
    _inside(root,'.storage')
    owned=read_json(meta/'owned-caches.json') if (meta/'owned-caches.json').exists() else {}
    pins=read_json(meta/'pins.json') if (meta/'pins.json').exists() else []
    rows=[];active=[];run_paths={};suite_paths={};references=set();errors=[]
    paths=[]
    if root.exists():
        for directory,dirs,files in os.walk(root,followlinks=False):
            dirs[:]=[d for d in dirs if d!='.storage' and not (Path(directory)/d).is_symlink()]
            paths.extend(Path(directory)/f for f in files)
    for p in paths:
        if p.name not in ('run.json','suite.json','reference.json','experiment.json','evaluation-context.json','model-links.json'):continue
        try:
            if p.is_symlink() or p.stat().st_size>2_000_000:continue
            d=read_json(p)
            if p.name=='run.json' and d.get('kind') in ('pdeflow_local_run','coordinate_stage_run'):
                run_paths[p.parent]=d.get('id',p.parent.name)
                if is_locked(p.parent):active.append(p.parent)
            if p.name=='suite.json':
                suite_paths[p.parent]=d.get('id')
                for entry in d.get('entries',[]):
                    child=Path(entry.get('run_path',''))
                    if child.is_absolute() and child.is_relative_to(root):suite_paths[child]=d.get('id')
            def strings(v):
                if isinstance(v,str):yield v
                elif isinstance(v,dict):
                    for a in v.values():yield from strings(a)
                elif isinstance(v,list):
                    for a in v:yield from strings(a)
            for text in strings(d):
                if text.startswith('/'):
                    q=Path(text)
                    if q.is_relative_to(root):references.add(str(q.relative_to(root)))
                elif (p.parent/text).is_file():references.add(str((p.parent/text).relative_to(root)))
                elif len(text)==32 and all(c in '0123456789abcdef' for c in text):references.add(text)
        except (ValueError,OSError,KeyError) as exc:errors.append({'path':str(p),'error':str(exc)})
    tracked=set()
    result=subprocess.run(['git','-C',str(project_root()),'ls-files','-z'],capture_output=True)
    if result.returncode==0:tracked={str((project_root()/n.decode()).absolute()) for n in result.stdout.split(b'\0') if n}
    for p in paths:
        try:
            st=p.lstat();relative=str(p.relative_to(root));kind=_kind(p)
            ancestors=[p,*p.parents]
            relative_ancestors={str(a.relative_to(root)) for a in ancestors if a.is_relative_to(root)}
            pinned=bool(relative_ancestors.intersection(pins))
            running=bool(set(ancestors).intersection(active))
            referenced=bool(relative_ancestors.intersection(references) or set(p.parts).intersection(references))
            ownership=owned.get(relative,{})
            recognized_owned=kind=='temporary_cache' and ownership.get('bytes')==st.st_size and ownership.get('mtime_ns')==st.st_mtime_ns and time.time()-st.st_mtime>86400
            protected=not recognized_owned or pinned or running or referenced or str(p.absolute()) in tracked or p.is_symlink()
            rows.append({'path':relative,'bytes':st.st_size,'created_at_unix':getattr(st,'st_birthtime',None),'updated_at_unix':st.st_mtime,'mtime_ns':st.st_mtime_ns,
                'type':kind,'active':running,'pinned':pinned,'referenced':referenced,'git_tracked':str(p.absolute()) in tracked,'protected':protected,
                'protection':'Scientific/QA/archive/unknown files retained; only registered unchanged disposable caches older than 24h eligible' if protected else 'Registered unchanged disposable cache older than 24h; reviewed quarantine only',
                'run_id':next((run_paths[a] for a in ancestors if a in run_paths),None),'suite_id':next((suite_paths[a] for a in ancestors if a in suite_paths),None)})
        except OSError as exc:errors.append({'path':str(p),'error':str(exc)})
    totals={}
    for r in rows:totals[r['type']]=totals.get(r['type'],0)+r['bytes']
    report={'schema_version':2,'root':str(root),'scanned_at':now(),'entries':rows,'files':len(rows),'total_bytes':sum(r['bytes'] for r in rows),'bytes_by_type':totals,'errors':errors,'size_semantics':'logical stat sizes, not disk allocated size; no symlink traversal','refresh':'explicit; cached report reused on normal UI refresh'}
    write_json(cache,report);return report

def _invalidate(root):
    cache=root/'.storage/inventory.json'
    if cache.exists():
        report=read_json(cache);report['invalidated']=True;write_json(cache,report)

def register_cache(root,path,owner='producer'):
    """Explicit producer/user declaration of a known disposable cache.

    Historical and user-added files are never assigned ownership by filename alone.
    Registration only writes metadata; preview/review/apply remains separate.
    """
    from pdeflow.local import run_lock
    if owner not in ('producer','explicit_user'):raise ValueError('Unknown cache ownership declaration')
    root=managed_root(root);p=_inside(root,path);path=p.relative_to(root).as_posix()
    if _kind(p)!='temporary_cache' or p.stat().st_size>2_000_000:raise ValueError('Only bounded recognized caches can be registered')
    meta=_inside(root,'.storage');meta.mkdir(parents=True,exist_ok=True)
    with run_lock(meta):
        owned=read_json(meta/'owned-caches.json') if (meta/'owned-caches.json').exists() else {}
        st=p.stat();owned[path]={'bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'owner':owner,'registered_at':now()}
        write_json(meta/'owned-caches.json',owned);_invalidate(root)
    return {'path':path,'registered':True,'owner':owner,'files_moved':0,'next':'Preview cleanup after the cache is older than 24h; review exact paths before applying'}

def pin(root,path,pinned=True):
    from pdeflow.local import run_lock
    root=managed_root(root);p=_inside(root,path)
    if not p.exists():raise ValueError('Artifact does not exist')
    path=p.relative_to(root).as_posix()
    meta=_inside(root,'.storage');meta.mkdir(parents=True,exist_ok=True)
    with run_lock(meta):
        old=set(read_json(meta/'pins.json')) if (meta/'pins.json').exists() else set()
        if pinned:old.add(path)
        else:old.discard(path)
        write_json(meta/'pins.json',sorted(old));_invalidate(root)
    return {'path':path,'pinned':pinned}

def preview(root=None):
    report=inventory(root,False);root=Path(report['root']);rows=[r for r in report['entries'] if not r['protected']]
    plan={'schema_version':1,'id':uuid.uuid4().hex,'root':str(root),'created_at':now(),'operation':'reversible_quarantine','inventory_scanned_at':report['scanned_at'],'entries':rows,'total_bytes':sum(r['bytes'] for r in rows),'applied':False,'policy':'No scientific results, QA, model exports, resume state, references or unknown user files eligible.'}
    plan['review_digest']=digest(plan)
    write_json(root/'.storage/plans'/f"{plan['id']}.json",plan);return plan

def apply(root,identifier,confirmation):
    root=managed_root(root)
    if len(identifier)!=32 or any(c not in '0123456789abcdef' for c in identifier):raise ValueError('Invalid cleanup plan ID')
    plan=read_json(root/'.storage/plans'/f'{identifier}.json')
    if confirmation!=plan['review_digest'] or digest({k:v for k,v in plan.items() if k!='review_digest'})!=plan['review_digest']:raise ValueError('Explicit reviewed plan digest required')
    if plan['root']!=str(root) or plan['applied']:raise ValueError('Wrong root or plan already applied')
    from contextlib import ExitStack
    from pdeflow.local import run_lock
    with ExitStack() as locks:
        meta=_inside(root,'.storage')
        locks.enter_context(run_lock(meta))
        if read_json(meta/'plans'/f'{identifier}.json')!=plan:raise ValueError('Cleanup plan changed while acquiring ownership; preview again')
        owners=set()
        for row in plan['entries']:
            p=_inside(root,row['path'])
            for parent in p.parents:
                if parent.is_relative_to(root) and (parent/'run.json').is_file():owners.add(parent)
        for parent in sorted(owners):locks.enter_context(run_lock(parent))
        # Acquired locks intentionally make these caches active; independently
        # recheck pin/reference/content and allow only our own acquired ownership.
        current={r['path']:r for r in inventory(root,True)['entries']}
        owned=read_json(root/'.storage/owned-caches.json') if (root/'.storage/owned-caches.json').exists() else {}
        for row in plan['entries']:
            p=_inside(root,row['path']);fresh=current.get(row['path']);ownership=owned.get(row['path'],{})
            protected_without_our_lock=fresh and fresh['protected'] and not (fresh['active'] and any(p.is_relative_to(owner) for owner in owners) and not fresh['pinned'] and not fresh['referenced'] and not fresh['git_tracked'] and fresh['type']=='temporary_cache')
            if not fresh or protected_without_our_lock or fresh['bytes']!=row['bytes'] or fresh['mtime_ns']!=row['mtime_ns'] or hashlib.sha256(p.read_bytes()).hexdigest()!=ownership.get('sha256'):raise ValueError('Plan is stale or an artifact became protected; preview again')
        quarantine=root/'.storage/quarantine'/identifier;quarantine.mkdir(parents=True,exist_ok=False)
        recovery={'plan':plan,'moved':[],'state':'in_progress'};write_json(quarantine/'recovery.json',recovery)
        for row in plan['entries']:
            p=_inside(root,row['path']);destination=quarantine/'files'/row['path'];destination.parent.mkdir(parents=True,exist_ok=True)
            p.rename(destination);recovery['moved'].append({'original':str(p),'quarantined':str(destination)});write_json(quarantine/'recovery.json',recovery)
        recovery['state']='quarantined';write_json(quarantine/'recovery.json',recovery)
        plan['applied']=True;write_json(root/'.storage/plans'/f'{identifier}.json',plan)
        return {'quarantine':str(quarantine),'files':len(recovery['moved']),'permanent_deletion':False,'recovery_manifest':str(quarantine/'recovery.json')}
