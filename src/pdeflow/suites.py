"""Sequential finite collections of existing managed presets; no second trainer."""
from dataclasses import replace
from datetime import datetime,timezone
from pathlib import Path
import uuid
from pdeflow import local
from pdeflow.runtime import Config,code_identity,digest,read_json,write_json

from pdeflow.catalogue import TARGETS,VERSION,successful_run
MEMBERS=TARGETS
SUITES={'local-pilot':MEMBERS[:6],'quickstart':MEMBERS[:6],'capacity-local':MEMBERS[6:8],'full':MEMBERS,'all':MEMBERS}

def plan(suite='full',device='auto',workspace=None,include=(),exclude=(),seed=37,budgets=None,missing=False):
    if suite not in SUITES:raise ValueError('Suites: '+', '.join(SUITES))
    unknown=(set(include)|set(exclude))-set(MEMBERS)
    if unknown:raise ValueError(f'Unknown suite presets: {sorted(unknown)}')
    if set(budgets or {})-set(SUITES[suite]):raise ValueError('Budget override must name a member of the selected suite; no unused overrides are ignored')
    if type(seed)!=int or not 0<=seed<2**63-len(MEMBERS):raise ValueError('Suite seed must be a nonnegative bounded integer')
    root=local.workspace_root(workspace);entries=[];runs=local.run_records(root)
    for index,name in enumerate(MEMBERS):
        if name not in SUITES[suite] or (include and name not in include) or name in exclude:continue
        c=local.preset_config(name)
        steps=(budgets or {}).get(name,2 if suite=='quickstart' else c.training.steps)
        if type(steps)!=int or steps<1:raise ValueError('Per-preset update budgets must be positive integers')
        selected_device=device if device!='auto' else c.runtime.device
        c=replace(c,training=replace(c.training,steps=steps,batch_size=1 if suite=='quickstart' else c.training.batch_size,seed=seed+index),data=replace(c.data,seed=seed+index),runtime=replace(c.runtime,device=selected_device)).validate(check_device=False)
        matches=[r['id'] for r in runs if successful_run(r,c,steps)]
        if missing and matches:continue
        entries.append({'preset':name,'target_version':VERSION,'specialist':c.specialist,'grid':list(c.grid.shape),'physical_times':list(c.grid.times) if c.grid.temporal_kind!='static' else None,'batch_size':c.training.batch_size,'updates':steps,'data_mode':c.provider.mode,'device':selected_device,'seed':seed+index,'resolved_config':c.to_dict(),'existing_exact_matches':matches,'cost_estimate':None,'cost_note':'No full-budget cost estimate measured; plan never launches training'})
    if not entries and not missing:raise ValueError('Suite selection is empty')
    return {'schema_version':2,'catalogue_version':VERSION,'suite':suite,'workspace':str(root),'entries':entries,'execution':'sequential','missing_only':missing,'output_template':str(root/'suites/<suite-id>'),'children_template':str(root/'runs/<suite-id>-<preset>'),'seed_policy':'root seed + stable catalogue index; original six indices retained','scope':'Frozen finite catalogue; no training implied by planning'}

def records(workspace=None):
    root=local.workspace_root(workspace)
    result=[]
    for path in (root/'suites').glob('*/suite.json'):
        try:result.append({**read_json(path),'path':str(path.parent)})
        except (ValueError,OSError):continue
    return sorted(result,key=lambda r:(r['created_at'],r['id']),reverse=True)

def resolve(selector='latest',workspace=None):
    root=local.workspace_root(workspace)
    if selector=='latest':
        rows=records(root)
        if not rows:raise ValueError('No suites. Run pdeflow train-all --plan, then train-all --suite quickstart --device cpu')
        path=Path(rows[0]['path'])
    else:path=Path(selector).resolve() if Path(selector).exists() else root/'suites'/selector
    return path,read_json(path/'suite.json')

def status(selector='latest',workspace=None):
    path,record=resolve(selector,workspace)
    entries=[]
    for entry in record['entries']:
        p=Path(entry['run_path'])
        row=read_json(p/'run.json') if (p/'run.json').exists() else None
        entries.append({**entry,'latest_update':row.get('progress') if row else None,'run_id':row.get('id') if row else entry.get('run_id'),'checkpoint_ready':bool(row and row.get('checkpoint_ready'))})
    current=set(MEMBERS);frozen={e['preset'] for e in record['entries']}
    return {**record,'original_catalogue_completed':all(e['state']=='completed' for e in record['entries']),'current_targets_not_in_snapshot':sorted(current-frozen),'entries':entries,'path':str(path),'locked':local.is_locked(path),'resume_command':f'pdeflow resume-suite {path}'}

def _execute(path,record,stop_after=None):
    print(f"Suite {record['id']} · {path} · sequential {len(record['entries'])} children",flush=True)
    def save(**changes):
        record.update(changes,updated_at=local.now());write_json(path/'suite.json',record)
    if record['source']['source_sha256']!=code_identity()['source_sha256']:raise ValueError('Suite source changed. Resume with the recorded source, or start a new suite; completed children are preserved.')
    save(state='running')
    try:
        for index,entry in enumerate(record['entries']):
            if digest(entry['resolved_config'])!=entry['config_identity']:raise ValueError('Suite configuration changed; restore the recorded manifest or start a new suite')
            if entry['state']=='completed':
                child_record=read_json(Path(entry['run_path'])/'run.json')
                if (not entry.get('reused') and (digest(child_record['resolved_config'])!=entry['config_identity'] or child_record['source']['source_sha256']!=record['source']['source_sha256'])) or not successful_run({**child_record,'path':entry['run_path']},entry['resolved_config'],entry['updates']):raise ValueError('Completed child configuration/provenance changed; preserve it and start a new suite')
                continue
            if entry['state']=='failed':
                if record.get('continue_on_error'):continue
                save(state='failed');return status(str(path))
            save(active_child=index)
            c=Config.from_dict(entry['resolved_config'])
            child=Path(entry['run_path'])
            existing=read_json(child/'run.json') if (child/'run.json').exists() else None
            try:
                if existing:
                    if digest(existing['resolved_config'])!=digest(c.to_dict()) or existing['source']['source_sha256']!=record['source']['source_sha256']:raise ValueError('Child configuration/provenance mismatch; preserve this child and start a new suite')
                    if existing['state']=='completed':result=local.status(str(child))
                    else:result=local.resume_training(str(child),workspace=record['workspace'],stop_after=stop_after)
                else:
                    result=local.start_training(preset=entry['preset'],workspace=record['workspace'],output=child,device=c.runtime.device,steps=c.training.steps,batch_size=c.training.batch_size,seed=c.training.seed,name=f"{record['id']}/{entry['preset']}",stop_after=stop_after)
                entry.update(run_id=result['id'],state='completed' if result['state']=='completed' else 'interrupted',error=None)
                if entry['state']!='completed':
                    save(state='interrupted');return status(str(path))
            except KeyboardInterrupt:
                entry['state']='interrupted';save(state='interrupted');return status(str(path))
            except (ValueError,OSError,RuntimeError) as error:
                entry.update(state='failed',error=f'{type(error).__name__}: {error}')
                save(state='failed')
                if not record['continue_on_error']:return status(str(path))
            save()
        save(state='completed_with_failures' if any(e['state']=='failed' for e in record['entries']) else 'completed',active_child=None,finished_at=local.now())
    except KeyboardInterrupt:save(state='interrupted')
    return status(str(path))

def start(suite='full',device='auto',workspace=None,include=(),exclude=(),seed=37,budgets=None,reuse=False,force=False,continue_on_error=False,stop_after=None,missing=False):
    planned=plan(suite,device,workspace,include,exclude,seed,budgets,missing)
    if not planned['entries']:return {**planned,'state':'completed','message':'All selected target budgets already covered; no jobs launched'}
    root=local.initialize(workspace);identifier=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+suite+'-'+uuid.uuid4().hex[:8]
    path=root/'suites'/identifier;path.mkdir(parents=True)
    record={**planned,'id':identifier,'source':code_identity(),'created_at':local.now(),'state':'pending','continue_on_error':continue_on_error,'active_child':None}
    runs=local.run_records(root)
    for entry in record['entries']:
        match=next((r for r in runs if r['id'] in entry['existing_exact_matches']),None) if reuse and not force else None
        entry.update(config_identity=digest(entry['resolved_config']),state='completed' if match else 'pending',run_path=match['path'] if match else str(root/'runs'/(identifier+'-'+entry['preset'])),reused=bool(match),run_id=match['id'] if match else None)
    with local.run_lock(path):
        write_json(path/'suite.json',record)
        return _execute(path,record,stop_after)

def resume(selector='latest',workspace=None,stop_after=None,retry_failed=False):
    path,record=resolve(selector,workspace)
    with local.run_lock(path):
        if retry_failed:
            for entry in record['entries']:
                if entry['state']=='failed':entry['state']='pending'
        return _execute(path,record,stop_after)
