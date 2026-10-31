"""Explicit permanent model deletion, separate from conservative cache quarantine.

Plans are immutable file fingerprints. Unknown files and external imports are protected.
No traversal, symlink following, directory deletion, or filename-only model inference.
"""
from pathlib import Path
from contextlib import ExitStack
from contextlib import contextmanager
import hashlib
import os
import uuid
from pdeflow.runtime import read_json, write_json, file_digest, digest
from pdeflow.local import is_locked, run_lock
from pdeflow.storage import _inside, managed_root


def _files(root):
    for directory,dirs,files in os.walk(root,followlinks=False):
        dirs[:]=[d for d in dirs if d!='.storage' and not (Path(directory)/d).is_symlink()]
        for name in files:
            p=Path(directory)/name
            if not p.is_symlink():yield p


def _json(p):
    try:return read_json(p) if not p.is_symlink() and p.stat().st_size<4_000_000 else {}
    except (OSError,ValueError):return {}


def _locked(directory):
    return (directory/'.run.lock').is_symlink() or is_locked(directory)


@contextmanager
def _parent_fd(root,relative):
    """Hold directory descriptors so a concurrently renamed parent cannot escape."""
    parts=Path(relative).parts
    _inside(root,relative)
    fd=os.open(root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for part in parts[:-1]:
            child=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
            os.close(fd);fd=child
        yield fd,parts[-1]
    finally:os.close(fd)


def _unlink_reviewed(root,entry):
    with _parent_fd(root,entry['path']) as (parent,name):
        fd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=parent)
        with os.fdopen(fd,'rb') as f:
            s=os.fstat(f.fileno());sha=hashlib.file_digest(f,'sha256').hexdigest()
            if [s.st_dev,s.st_ino]!=entry['inode'] or s.st_size!=entry['bytes'] or s.st_mtime_ns!=entry['mtime_ns'] or sha!=entry['sha256']:raise ValueError('An artifact changed since preview; preview again')
            current=os.stat(name,dir_fd=parent,follow_symlinks=False)
            if (s.st_dev,s.st_ino)!=(current.st_dev,current.st_ino):raise ValueError('Artifact changed during deletion')
            os.unlink(name,dir_fd=parent)
            return s


def inventory(root):
    root=managed_root(root);_inside(root,'.storage');paths=list(_files(root));groups={};claims={};active_roots=[]
    active_ids=set()
    for p in paths:
        if p.name=='job.json' and _json(p).get('status') in ('queued','running'):
            request=_json(p.parent/'request.json');active_ids.update([request.get('model_id'),request.get('refinement_model_id')])
    for p in paths:
        if p.name in ('run.json','workflow.json'):
            d=_json(p)
            if d.get('kind') in ('coordinate_stage_run','pdeflow_local_run') or p.name=='workflow.json':
                if _locked(p.parent):active_roots.append(p.parent)
    # Every schema-validated model, including historical bare exports and backups,
    # is classified. A .pt extension by itself never grants deletion authority.
    import torch
    for p in paths:
        if p.suffix!='.pt':continue
        try:
            cp=torch.load(p,map_location='cpu',weights_only=True)
            if not isinstance(cp,dict):continue
            known=(cp.get('schema_version')==1 and cp.get('kind') in ('inference','resume') and {'model','config'}<=cp.keys()) or (cp.get('stage') in ('flow','autoencoder') and {'autoencoder','config','normalization','dependency'}<=cp.keys())
            toy=_json(p.parent/'toy.json') if p.name=='toy.pt' else {}
            known|=set(cp)=={'model','samples','initial'} and toy.get('formulation')=='independent_gaussian_linear_cfm' and 'actual_optimizer_updates' in toy and str(toy.get('source',{}).get('version','')).startswith('pdeflow-')
            if not known:continue
        except Exception:continue
        owner=next((a for a in p.parents if a.is_relative_to(root) and (a/'run.json').exists()),p.parent)
        if owner==root:owner=p.parent
        key=str(owner.relative_to(root)) or '.'
        d=_json(owner/'run.json');group=groups.setdefault(key,{'id':key,'name':f'{d.get("specialist","Heat" if cp.get("stage") else "PDE")} · {d.get("stage", "fixed grid")} · {owner.name}',
            'files':[],'active':False,'dependencies':[],'model_ids':[],'dataset_ids':[],'owners':[str(owner),str(p.parent)]})
        group['architecture_status']='Current joint model' if cp.get('architecture')=='operator_aware_physical_latent_v2' else 'Legacy architecture — incompatible with current model';
        if group['architecture_status'].startswith('Legacy') and 'Legacy' not in group['name']:group['name']+=' · '+group['architecture_status']
        group['files'].append(str(p.relative_to(root)));group['model_ids'].append(file_digest(p)[:32])
        group['active']|=any(p.is_relative_to(a) for a in active_roots) or _locked(p.parent)
        group['active']|=bool(active_ids.intersection(group['model_ids']))
        if cp.get('dataset_identity'):group['dataset_ids'].append(cp['dataset_identity'])
        dep=cp.get('dependency') or d.get('dependency') or {}
        if dep.get('path'):
            q=Path(dep['path'])
            if not q.is_absolute():q=(p.parent/q).resolve()
            if q.is_absolute() and q.is_relative_to(root):group['dependencies'].append(str(q.relative_to(root)))
        claims[str(p.relative_to(root))]=key
    for g in groups.values():
        g['dependencies']=sorted(set(claims[p] for p in g['dependencies'] if p in claims and claims[p]!=g['id']))
        g['bytes']=_unique_bytes(root,g['files'])
    selected=set()
    for p in paths:
        if p.name=='selected-model.json':
            d=_json(p);selected.add(d.get('id'))
    for g in groups.values():g['selected']=bool(selected.intersection(g['model_ids']))
    categories={'models':0,'data':0,'results_logs':0,'protected_unknown':0};seen=set()
    model_files=set(claims)
    for p in paths:
        s=p.stat();inode=(s.st_dev,s.st_ino)
        if inode in seen:continue
        seen.add(inode)
        category='models' if str(p.relative_to(root)) in model_files else _optional_category(p,root)
        categories[category]+=s.st_size
    return {'root':str(root),'models':list(groups.values()),'categories':categories,'total_bytes':sum(categories.values()),
            'meaning':'Unique logical bytes inside managed root; hard links counted once. External linked imports, source and unknown files are protected.'}


def _optional_category(p,root):
    # Only output files with a producer manifest are eligible; arbitrary uploads
    # or personal files with matching extensions remain protected.
    for a in p.parents:
        if not a.is_relative_to(root):break
        d=_json(a/'dataset.json') if (a/'dataset.json').exists() else {}
        if d.get('kind') in ('coordinate_training_data','generated_dataset') and p.name in ('dataset.npz','dataset.json'):return 'data'
        run=_json(a/'run.json') if (a/'run.json').exists() else {}
        if run.get('kind') in ('coordinate_stage_run','pdeflow_local_run') and p.name in ('train.log','worker.log','history.json','metrics.json','preview.npz','comparison.json','result.npz'):return 'results_logs'
        if (a/'job.json').exists() and p.name in ('worker.log','result.npz','metrics.json','comparison.json','evaluation-reference.npz'):return 'results_logs'
    return 'protected_unknown'


def _unique_bytes(root,paths):
    seen={}
    for name in paths:
        s=_inside(root,name).stat();seen[(s.st_dev,s.st_ino)]=s.st_size
    return sum(seen.values())


def preview(root,selected=None,all_models=False,data=False,results=False):
    root=managed_root(root);report=inventory(root);groups={g['id']:g for g in report['models']}
    if not all_models and (not selected or set(selected)-groups.keys()):raise ValueError('Choose existing managed models before previewing deletion')
    chosen=set(groups) if all_models else set(selected);shared=[]
    # Include stage prerequisites only when no retained flow needs them.
    for key in list(chosen):
        for dependency in groups[key]['dependencies']:
            users={g['id'] for g in groups.values() if dependency in g['dependencies']}
            if users<=chosen:chosen.add(dependency)
            else:shared.append({'id':dependency,'reason':'Retained: shared stage dependency used by another model'})
    for key in chosen:
        users={g['id'] for g in groups.values() if key in g['dependencies']}
        if users-chosen:raise ValueError('This stage is needed by retained models; select those dependent models as well')
    selected_groups=[groups[k] for k in sorted(chosen)]
    if any(g['active'] for g in selected_groups):raise ValueError('An active training or reconstruction job is using these model artifacts; stop it safely first')
    paths={p for g in selected_groups for p in g['files']}
    owner_roots={Path(o) for g in selected_groups for o in g['owners']}
    if data or results:
        dataset_ids={i for g in selected_groups for i in g['dataset_ids']}
        retained_dataset_ids={i for g in groups.values() if g['id'] not in chosen for i in g['dataset_ids']}
        data_roots=set()
        if data:
            for p in _files(root):
                if p.name=='dataset.json':
                    meta=_json(p)
                    if meta.get('kind')=='coordinate_training_data' and meta.get('sha256') in dataset_ids-retained_dataset_ids:
                        candidate=p.parent/'dataset.npz'
                        if candidate.is_file() and not candidate.is_symlink() and file_digest(candidate)==meta['sha256']:data_roots.add(p.parent)
        for p in _files(root):
            if not any(p.is_relative_to(a) for a in owner_roots|data_roots):continue
            category=_optional_category(p,root)
            if (data and category=='data') or (results and category=='results_logs'):paths.add(str(p.relative_to(root)))
    entries=[]
    for name in sorted(paths):
        p=_inside(root,name);s=p.stat()
        entries.append({'path':name,'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'inode':[s.st_dev,s.st_ino],'links':s.st_nlink,'allocated':s.st_blocks*512,'sha256':file_digest(p)})
    confirmation='DELETE ALL MODEL ARTIFACTS' if all_models else 'DELETE SELECTED MODEL ARTIFACTS'
    inodes={}
    for e in entries:inodes.setdefault(tuple(e['inode']),[]).append(e)
    expected_allocated=sum(rows[0]['allocated'] for rows in inodes.values() if len(rows)==rows[0]['links'])
    plan={'id':uuid.uuid4().hex,'root':str(root),'entries':entries,'models':selected_groups,'shared_retained':shared,'unique_bytes':_unique_bytes(root,paths),
          'expected_last_link_allocated_bytes':expected_allocated,
          'confirmation':confirmation,'data_included':data,'results_included':results,'data_results_policy':'Only associated producer-owned recognized files included when selected; unknown and imported data protected.',
          'selected_becoming_unavailable':[g['name'] for g in selected_groups if g['selected']],'applied':False}
    plan['digest']=digest(plan);write_json(_inside(root,f'.storage/model-plans/{plan["id"]}.json'),plan)
    return plan


def apply(root,identifier,confirmation,review_digest):
    root=managed_root(root)
    if len(identifier)!=32 or any(c not in 'abcdef0123456789' for c in identifier):raise ValueError('Invalid deletion plan')
    path=_inside(root,f'.storage/model-plans/{identifier}.json');plan=read_json(path)
    if plan['root']!=str(root) or plan['applied'] or confirmation!=plan['confirmation'] or review_digest!=plan['digest'] or digest({k:v for k,v in plan.items() if k!='digest'})!=plan['digest']:raise ValueError('Confirm the exact current deletion preview')
    with ExitStack() as locks:
        _inside(root,'.storage/.run.lock')
        locks.enter_context(run_lock(_inside(root,'.storage')))
        if read_json(path)!=plan:raise ValueError('Deletion plan already used or changed')
        # Locks are also used by trainers and the workbench inference worker.
        owners=sorted({o for g in plan['models'] for o in g['owners']})
        for owner in owners:
            relative=str(Path(owner).relative_to(root));_inside(root,str(Path(relative)/'.run.lock'))
            locks.enter_context(run_lock(_inside(root,relative)))
        for entry in plan['entries']:
            p=_inside(root,entry['path']);s=p.stat()
            if [s.st_dev,s.st_ino]!=entry['inode'] or s.st_size!=entry['bytes'] or s.st_mtime_ns!=entry['mtime_ns'] or file_digest(p)!=entry['sha256']:raise ValueError('An artifact changed since preview; preview again')
        removed=[];failures=[];freed=set();allocated=0
        for entry in plan['entries']:
            try:
                s=_unlink_reviewed(root,entry);inode=(s.st_dev,s.st_ino)
                removed.append(entry['path'])
                if s.st_nlink==1 and inode not in freed:allocated+=s.st_blocks*512;freed.add(inode)
            except (OSError,ValueError) as e:failures.append({'path':entry['path'],'error':str(e)})
        ids={e['sha256'][:32] for e in plan['entries'] if e['path'] in removed}
        cleared=[]
        for p in list(_files(root)):
            if p.name=='selected-model.json' and _json(p).get('id') in ids:
                p.unlink();cleared.append(str(p.relative_to(root)))
        plan['applied']=True;plan['removed']=removed;plan['failures']=failures
        write_json(path,plan)
    after=inventory(root)
    return {'removed':removed,'failures':failures,'cleared_selections':cleared,'removed_model_ids':sorted(ids),
            'actual_unique_bytes_removed':_unique_removed(plan,removed),'last_link_allocated_bytes_released':allocated,'inventory':after,
            'meaning':'Permanently unlinked; no trash copy. Unique managed bytes removed and allocated last-link blocks reported. Filesystem snapshots/compression may affect free disk space.'}


def _unique_removed(plan,removed):
    seen={}
    for e in plan['entries']:
        if e['path'] in removed:seen[tuple(e['inode'])]=e['bytes']
    return sum(seen.values())
