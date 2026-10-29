"""Bounded visual payloads derived only from actual inputs, results and histories."""
from pathlib import Path
import hashlib
import math
import json
import re
import numpy as np
from pdeflow.core import ProblemSpec
from pdeflow.observations import ObservationSet
from pdeflow.runtime import digest,read_json
from pdeflow.workbench.observations import parse_rows


def input_preview(problem,rows,field=None,time_index=0,axis=2,index=0,noise_std=.03):
    p=ProblemSpec.from_dict(problem);g=p.grid
    if math.prod(g.shape)>1048576:raise ValueError('Input preview supports at most1048576 grid points; use a smaller explicit request')
    choices=[{'name':f.name,'role':'unknown target','units':f.units} for f in p.fields]+[{'name':k,'role':'known input','units':'nondimensional'} for k in p.known_fields]
    field=field or choices[0]['name']
    if field not in [c['name'] for c in choices]:raise ValueError('Choose a declared field or known input')
    if not 0<=time_index<len(g.times):raise ValueError('Physical time index outside saved times')
    if g.dimension==3 and (not 0<=axis<3 or not 0<=index<g.shape[axis]):raise ValueError('Input slice outside physical grid')
    spatial=[i for i in range(g.dimension) if g.dimension!=3 or i!=axis]
    stride=[max(1,math.ceil(g.shape[a]/128)) for a in spatial]
    coordinates={g.axis_names[a]:g.coordinates()[a][::step].tolist() for a,step in zip(spatial,stride)}
    shape=[len(coordinates[g.axis_names[a]]) for a in spatial]
    values=None
    if field in p.known_fields:
        full=np.asarray(p.known_fields[field],dtype=float)
        if full.shape!=g.shape or not np.isfinite(full).all():raise ValueError(f'Known field {field} must have shape{g.shape} and finite values')
        sliced=np.take(full,index,axis=axis) if g.dimension==3 else full
        values=sliced[tuple(slice(None,None,s) for s in stride)].tolist()
    issues=[];points=[]
    if rows:
        try:parse_rows(p,rows,noise_std)
        except (ValueError,TypeError) as error:
            match=re.search(r'Row (\d+)',str(error));issues=[{'row':int(match[1])-1 if match else None,'message':str(error)}]
        if not issues:
            for row_id,row in enumerate(rows):
                if row['field']!=field:continue
                if g.temporal_kind=='transient' and not math.isclose(float(row['time']),g.times[time_index],abs_tol=1e-9):continue
                if g.dimension==3 and not math.isclose(float(row[g.axis_names[axis]]),float(g.coordinates()[axis][index]),abs_tol=1e-8):continue
                points.append({'row_id':row_id,**row})
    displayed=points[::max(1,math.ceil(len(points)/512))]
    return {'field':field,'choices':choices,'shape':shape,'axes':[g.axis_names[a] for a in spatial],
            'coordinates':coordinates,'values':values,'points':displayed,'point_count':len(points),'total_observations':len(rows),
            'physical_time':g.times[time_index] if g.temporal_kind=='transient' else None,
            'slice_coordinate':float(g.coordinates()[axis][index]) if g.dimension==3 else None,
            'boundary':g.domain.boundary,'units':next(c['units'] for c in choices if c['name']==field),
            'role':'known input' if values is not None else 'unknown target; only measured locations are supplied',
            'valid':not issues,'issues':issues,'downsampled':any(s>1 for s in stride) or len(displayed)!=len(points),
            'interpretation':'Actual supplied arrays and on-grid observations only. Unmeasured regions are unknown; no interpolation or reference field is used.'}


def bounded_rows(rows,limit=600):
    if len(rows)<=limit:return rows,False
    # Preserve real extrema in loss bins; no smoothing or invented samples.
    indices={0,len(rows)-1};width=math.ceil(len(rows)/(limit//4))
    for start in range(0,len(rows),width):
        group=list(range(start,min(start+width,len(rows))))
        for key in ['training_loss','validation_loss']:
            valid=[i for i in group if key in rows[i]]
            if valid:indices.update([min(valid,key=lambda i:rows[i][key]),max(valid,key=lambda i:rows[i][key])])
    return [rows[i] for i in sorted(indices)],True


def training_runs(workspace):
    from pdeflow.local import run_records,status
    result=[]
    for record in run_records(workspace)[:50]:
        path=Path(record['path']);record=status(str(path));live=read_json(path/'progress.json') if (path/'progress.json').exists() else {};heartbeat=read_json(path/'heartbeat.json') if (path/'heartbeat.json').exists() else {};history_path=path/record['paths']['history']
        history=read_json(history_path) if history_path.is_file() else []
        architecture=None
        export=path/record['paths']['latest_export']
        if export.is_file():
            try:
                from pdeflow.catalogue import export_record
                architecture=export_record(export)['architecture']
            except (OSError,ValueError,RuntimeError):pass
        rows,downsampled=bounded_rows(history)
        events_path=path/record["paths"]["events"]
        events=[json.loads(line) for line in events_path.read_text().splitlines() if line.strip()] if events_path.is_file() else []
        result.append({'id':hashlib.sha256(str(path).encode()).hexdigest()[:32],'run_id':record['id'],'name':record.get('name'),
                       'specialist':record['specialist'],'preset':record['preset'],'profile':record['profile'],'state':record['state'],
                       'data_mode':record['data_mode'],'progress':record['progress'],'budget':record['budget'],'device':record['device'],'grid':record['resolved_config']['grid'],
                       'history':[{k:row[k] for k in ['step','training_loss','validation_loss','data_wait_seconds','neural_host_seconds'] if k in row} for row in rows],
                       'checkpoint_positions':sorted({event['step'] for event in events if event.get('kind')=='checkpoint'}), 'history_total':len(history),'downsampled':downsampled,'checkpoint_every':record['resolved_config']['training']['checkpoint_every'],
                       'meaning':'Generative CFM objective, not reconstruction error; validation is separate from test.',
                       'resume_command':f"pdeflow resume {path}",'resolved_config':record['resolved_config'],'dtype':record['dtype'],'live':live,'heartbeat':heartbeat,'active':record['locked'],'architecture':architecture,'log_tail':(path/'train.log').read_text()[-5000:] if (path/'train.log').exists() else '', 'checkpoint_updated_at_unix':(path/'training/checkpoint.pt').stat().st_mtime if (path/'training/checkpoint.pt').exists() else None,'provider_summary':{k:v for k,v in (read_json(path/'training/manifest.json').get('data_provider',{}) if (path/'training/manifest.json').exists() else {}).items() if k in ('mode','unique_training_problems_consumed','unique_training_problems_solved','accounting')},'evaluate_command':f'pdeflow evaluate --run {path}','command_workspace':'Use the same --workspace setting as this backend'})
    return result[:50]


def result_plots(directory):
    directory=Path(directory);m=read_json(directory/'metrics.json');p=ProblemSpec.from_dict(read_json(directory/'problem.json'))
    obs=ObservationSet.from_dict(read_json(directory/'observations.json'),p.grid)
    import torch
    with np.load(directory/'result.npz',allow_pickle=False) as data:
        samples=torch.from_numpy(data['samples'][0].copy())
        weights=torch.from_numpy(data['weights'][0].copy()).double() if 'weights' in data else torch.full((samples.shape[0],),1/samples.shape[0],dtype=torch.float64)
    predicted=(obs.operator.apply(samples).double()*weights[:,None]).sum(0)
    valid=torch.nonzero(obs.mask[0]).flatten().tolist()
    chosen=valid[::max(1,math.ceil(len(valid)/256))]
    points=[{'index':i,'field':p.fields[int(obs.operator.indices[i,0])].name,'observed':float(obs.values[0,i]),'predicted':float(predicted[i]),'residual':float(predicted[i]-obs.values[0,i])} for i in chosen]
    process=m.get('cost',{}).get('process',{})
    history=process.get('diagnostic_history',[])
    traces=[]
    estimated=[row for row in history if row.get('label')=='estimated_endpoint']
    positions=[a['generation_position'] for a in process.get('actions',[]) if 'generation_position' in a]
    position_iterator=iter(positions) if len(estimated)==len(positions) else iter([])
    for row in history:
        entries=row.get('entries',{})
        pos=row.get('generation_position')
        if pos is None:pos=row.get('generation_coordinate')
        if pos is None and row.get('label')=='estimated_endpoint':pos=next(position_iterator,None)
        if pos is None and row.get('label')!='estimated_endpoint':pos=m.get('cost',{}).get('inference_state',{}).get('generation_position')
        traces.append({'position':pos,'label':row.get('label','diagnostic'),'values':{k:e.get('value') for k,e in entries.items() if k.startswith(('observation.','pde.')) and isinstance(e.get('value'),(int,float))}})
    return {'points':points,'total_observations':len(valid),'downsampled':len(chosen)!=len(valid),'weighted':bool(m.get('provenance',{}).get('mode') in ['fk','importance']),
            'traces':traces[-512:],'units':'field-specific physical units; nondimensional in current specialists','reference_used':False,
            'cost':{k:m.get('cost',{}).get(k) for k in ['seconds','model_evaluations','model_particle_evaluations','gradient_evaluations','energy_particle_evaluations']}}


def compare_results(directories):
    if not 2<=len(directories)<=4:raise ValueError('Select2–4 completed runs for comparison')
    records=[]
    for identifier,directory in directories:
        d=Path(directory);p=read_json(d/'problem.json');o=read_json(d/'observations.json');m=read_json(d/'metrics.json')
        with np.load(d/'result.npz',allow_pickle=False) as a:reference=hashlib.sha256(a['reference'].tobytes()).hexdigest() if 'reference' in a else None
        records.append({'id':identifier,'identity':digest({'problem':p,'observations':o}),'reference_identity':reference,'metrics':m})
    if len({r['identity'] for r in records})!=1:raise ValueError('Runs have different equations/parameters, grid, units or measurement identities/noise. Direct quality/cost comparison requires the same actual problem and observations.')
    same_reference=len({r['reference_identity'] for r in records})==1 and records[0]['reference_identity'] is not None
    return {'compatible':True,'identity':records[0]['identity'],'same_reference':same_reference,'reference_note':'Common attached evaluation truth' if same_reference else 'Reference accuracy unavailable for comparison: attach the same truth to every run',
            'runs':[{'id':r['id'],'mode':r['metrics']['provenance'].get('mode'),'checkpoint':r['metrics']['provenance'].get('checkpoint_sha256'),'observation_rmse':r['metrics'].get('observation_rmse'),'pde_fd_rmse':r['metrics'].get('pde_fd_rmse'),'relative_l2_unobserved_mean':r['metrics'].get('relative_l2_unobserved_mean') if same_reference else None,'seconds':r['metrics'].get('cost',{}).get('seconds'),'model_evaluations':r['metrics'].get('cost',{}).get('model_evaluations'),'stop_reason':r['metrics']['stop_reason']} for r in records],
            'warning':'Settings/checkpoints may differ; no automatic accuracy ranking, test tuning or calibrated uncertainty claim.'}
