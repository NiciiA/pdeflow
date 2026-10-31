"""Passive stage discovery uses real manifests plus existing kernel ownership locks."""
from pathlib import Path
from pdeflow.local import workspace_root,is_locked
from pdeflow.runtime import read_json,file_digest
from .training import read_checkpoint

def runs(workspace=None):
    from pdeflow.training_workflow import known_workspaces
    result=[];seen=set()
    paths=[p for root in known_workspaces(workspace) for p in sorted((root/'coordinate').glob('*/run.json'))]
    for p in paths:
        try:
            row=read_json(p);active=is_locked(p.parent)
            row['state']='running' if active else 'interrupted' if row['state'] in ('running','preparing') else row['state']
            row['active']=active;row['path']=str(p.parent);row['id']=p.parent.name
            export=Path(row['export']);row['available']=False
            if export.is_file():
                if row.get('registered_sha256') and file_digest(export)!=row['registered_sha256']:raise ValueError('Registered coordinate export changed; explicitly register its new content')
                cp,c=read_checkpoint(export,row['stage']);row['available']=bool(cp['completed'] or cp.get('inference_ready'));row['model_id']=file_digest(export)[:32]
                row['capabilities']=cp['capabilities'];row['checkpoint_progress']=cp['progress']
            key=(row.get('model_id'),row.get('stage')) if row.get('available') else str(p)
            if key not in seen:result.append(row);seen.add(key)
        except (ValueError,RuntimeError,OSError,KeyError) as error:result.append({'id':p.parent.name,'state':'incompatible','error':str(error),'available':False})
    return result

def model(workspace,identifier):
    for row in runs(workspace):
        if row.get('model_id')==identifier and row.get('stage')=='flow' and row['available']:return row
    raise ValueError('Coordinate flow export unavailable. Complete both stages under WORKSPACE/coordinate/<run>, then refresh.')

def register_checkpoint(path,workspace=None):
    """Explicit link to a completed portable stage export; never moves user weights."""
    from pdeflow.runtime import write_json,code_identity
    from pdeflow.local import now
    path=Path(path).expanduser().resolve();cp,c=read_checkpoint(path)
    if not (cp['completed'] or cp.get('inference_ready')):raise ValueError('Complete the stage or explicitly export the latest valid partial inference state')
    fingerprint=file_digest(path);identifier=f'imported-{cp["stage"]}-{fingerprint[:12]}'
    root=workspace_root(workspace)/'coordinate'/identifier;root.mkdir(parents=True,exist_ok=True)
    record={'schema_version':1,'kind':'coordinate_stage_run','id':identifier,'stage':cp['stage'],'architecture':cp['architecture'],'specialist':'+'.join(c.family_weights),'state':'completed','progress':cp['progress'],'budget':getattr(c,{'autoencoder':'ae_steps','evolution':'evolution_steps','flow':'flow_steps'}[cp['stage']]),'config':c.to_dict(),'dataset_identity':cp['dataset_identity'],'normalization':cp['normalization'],'dependency':cp['dependency'],'export':str(path),'registered_sha256':fingerprint,'created_at':now(),'source':cp['source'],'registration_source':code_identity(),'device':'portable export; inference device chosen separately','parameters':sum(v.numel() for v in (cp.get('autoencoder') or cp.get('inference_state') or {}).values())+sum(v.numel() for v in (cp.get('velocity') or {}).values()),'resume_command':'Imported inference export has no optimizer resume state. Resume the original stage directory with its matching recorded source.','history':cp.get('history',[])[-200:],'history_total':len(cp.get('history',[])),'meaning':'Linked completed stage export; budget completion does not establish reconstruction quality'}
    if not cp['completed']:record.update(state='partial',training_status=cp.get('training_status','Partial training'))
    manifest=root/'run.json'
    if not manifest.exists():write_json(manifest,record)
    else:
        previous=read_json(manifest)
        # An explicit re-import of identical content can repair a registration
        # whose managed copy was permanently removed. Never replace a live link.
        if previous.get('kind')=='coordinate_stage_run' and previous.get('registered_sha256')==fingerprint and not Path(previous['export']).is_file():
            record['previous_missing_export']=previous['export']
            write_json(manifest,record)
    return record
