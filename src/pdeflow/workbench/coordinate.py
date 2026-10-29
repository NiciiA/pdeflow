"""Coordinate specialist routes reuse the existing single workbench job queue."""
from pathlib import Path
import uuid
import numpy as np
from fastapi import Query
from fastapi.responses import FileResponse
from pydantic import BaseModel,ConfigDict,Field
from pdeflow.runtime import read_json,write_json
from pdeflow.core import ProblemSpec
from .jobs import contained

class Inverse(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    model_id:str=Field(pattern=r'^[a-f0-9]{32}$')
    problem:dict
    mathematical_request:dict|None=None
    require_continuous:bool=False
    rows:list[dict]=Field(default_factory=list,max_length=20000)
    queries:list[list[float]]|None=None
    steps:int=Field(default=12,ge=1,le=512)
    particles:int=Field(default=2,ge=1,le=16)
    seed:int=Field(default=91,ge=0,lt=2**63)
    chunk_size:int=Field(default=1024,ge=1,le=65536)
    device:str=Field(default='cpu',pattern=r'^(cpu|mps|cuda|auto)$')
    submission_token:str=Field(default_factory=lambda:uuid.uuid4().hex,max_length=80)

class ImportReference(BaseModel):
    model_config=ConfigDict(extra='forbid')
    artifact:dict


def register(app,workspace,data,output):
    from pdeflow.coordinate.registry import runs,model
    @app.get('/api/coordinate/runs')
    def coordinate_runs():return runs(workspace)

    @app.post('/api/references/import',status_code=201)
    def reference_import(payload:ImportReference):
        from pdeflow.reference.imports import import_reference
        identifier=uuid.uuid4().hex
        report=import_reference(payload.artifact,contained(data,identifier),identifier)
        return {k:v for k,v in report.items() if k!='path'}

    @app.post('/api/coordinate/validate')
    def validate(payload:Inverse):
        from pdeflow.coordinate.training import read_checkpoint
        from pdeflow.coordinate.contracts import validate_rows,validate_queries,grid_queries
        from pdeflow.runtime import RuntimeSpec
        entry=model(workspace,payload.model_id);cp,c=read_checkpoint(entry['export'],'flow');p=ProblemSpec.from_dict(payload.problem)
        if payload.mathematical_request is not None:
            from pdeflow.symbolic import analyze
            # JSON/JavaScript serializes0.0 as0; structural numeric equality matches physical semantics.
            parsed=analyze(payload.mathematical_request,observation_kinds=('coordinate_point_set',))
            if parsed['problem'] is None:raise ValueError('Current mathematical source is unsupported: '+'; '.join(parsed['reasons']))
            source=ProblemSpec.from_dict(parsed['problem'])
            if source.grid.domain!=p.grid.domain or source.grid.dimension!=p.grid.dimension or source.grid.convention!=p.grid.convention or source.grid.temporal_kind!=p.grid.temporal_kind or {**source.to_dict(),'grid':p.to_dict()['grid']}!=p.to_dict():raise ValueError('Current mathematical source differs from the executable physical problem; reparse the current draft')
            if payload.mathematical_request['measurements']['rows']!=payload.rows:raise ValueError('Coordinate source measurement rows differ from the submitted scalar set')
        if payload.require_continuous and c.temporal_mode!='continuous':raise ValueError('Legacy snapshot checkpoint is not continuously trained; train/import a continuous heat model or use legacy inference with its original support')
        c.compatible(p);validate_rows(c,payload.rows,p);q=validate_queries(c,payload.queries if payload.queries is not None else grid_queries(p.grid))
        if not payload.rows and not cp['capabilities']['unconditional_trained']:raise ValueError('This model needs at least one measurement; its training did not include a zero-observation prior')
        if len(q)*payload.particles>16_777_216:raise ValueError('Decoded output exceeds64MiB budget')
        RuntimeSpec(device=payload.device).resolve()
        return {'valid':True,'temporal':__import__('pdeflow.coordinate.temporal',fromlist=['capability']).capability(c),'query_count':len(q),'observations':len(payload.rows),'architecture':'set_latent_coordinate','training_reference_shape':c.reference_shape,'query_shape':p.grid.shape if payload.queries is None else None,'warning':'New query resolution does not establish learned fine-scale accuracy','zero_observation_prior_trained':cp['capabilities']['unconditional_trained']}

    @app.post('/api/coordinate/reconstruct',status_code=202)
    def reconstruct(payload:Inverse):
        validate(payload);entry=model(workspace,payload.model_id)
        request={'kind':'coordinate','model_id':payload.model_id,'problem':payload.problem,'rows':payload.rows,'queries':payload.queries,'inference':{'steps':payload.steps,'particles':payload.particles,'seed':payload.seed,'chunk_size':payload.chunk_size},'runtime':{'device':payload.device},'mode':'learned_observation_conditioning'}
        job=app.state.jobs.submit(request,entry['export'],payload.submission_token)
        if payload.mathematical_request is not None:write_json(contained(output,job['id'])/'mathematical-source.json',payload.mathematical_request)
        return job

    def result_directory(identifier):
        directory=contained(output,identifier)
        if app.state.jobs.get(identifier)['status']!='succeeded':raise ValueError('Wait for successful reconstruction before inspecting results')
        return directory/'coordinate-result'

    @app.get('/api/coordinate/jobs/{identifier}/result')
    def result(identifier:str):
        directory=result_directory(identifier);report=read_json(directory/'metrics.json')
        with np.load(directory/'result.npz',allow_pickle=False) as f:table=[{'query':q.tolist(),'samples':f['samples'][:,i].tolist()} for i,q in enumerate(f['queries'][:256])]
        return {'table_first256':table,'metrics':report,'comparison':read_json(directory/'comparison.json') if (directory/'comparison.json').exists() else None}

    @app.post('/api/coordinate/jobs/{identifier}/compare')
    def compare(identifier:str,payload:dict):
        if set(payload)!={'reference_id'}:raise ValueError('Comparison requires only a saved reference ID, after reconstruction')
        from pdeflow.coordinate.evaluation import attach_reference
        return attach_reference(result_directory(identifier),contained(data,payload['reference_id']))

    @app.get('/api/coordinate/jobs/{identifier}/slice')
    def view(identifier:str,time_index:int=Query(0,ge=0),axis:int=Query(2,ge=0,le=2),index:int=Query(0,ge=0),statistic:str='mean'):
        directory=result_directory(identifier);p=ProblemSpec.from_dict(read_json(directory/'problem.json'));m=read_json(directory/'metrics.json')
        if not m['regular_grid']:raise ValueError('Scattered queries export an ordered table; select a regular query grid for slice plotting')
        if time_index>=len(p.grid.times):raise ValueError('Physical-time index out of range')
        with np.load(directory/'result.npz',allow_pickle=False) as f:samples=f['samples'].reshape(-1,len(p.grid.times),*p.grid.shape)
        from pdeflow.reference import slice_payload
        if statistic not in ('mean','spread') and not (statistic.startswith('sample:') and statistic[7:].isdigit() and int(statistic[7:])<len(samples)):raise ValueError('Choose mean, spread or valid sample:index')
        field=samples.mean(0) if statistic=='mean' else samples.std(0) if statistic=='spread' else samples[int(statistic[7:])]
        result=slice_payload(p,field[time_index],'u',axis,index,time_index)
        result['statistic']='ensemble mean' if statistic=='mean' else 'empirical spread, not calibrated uncertainty' if statistic=='spread' else statistic
        rows=read_json(directory/'observations.json');visible=[r for r in rows if abs(r['time']-p.grid.times[time_index])<1e-8 and (p.grid.dimension!=3 or abs(r[p.grid.axis_names[axis]]-float(p.grid.coordinates()[axis][index]))<1e-8)]
        result['observations']=visible;result['observation_count']=len(rows);result['off_slice_observation_count']=len(rows)-len(visible)
        ref=directory/'evaluation-reference.npz'
        if ref.exists() and statistic!='spread':
            with np.load(ref,allow_pickle=False) as f:truth=f['reference'].reshape(len(p.grid.times),*p.grid.shape)[time_index]
            reference_slice=slice_payload(p,truth,'u',axis,index,time_index)['values'];result['reference']=reference_slice
            result['error']=(np.asarray(result['values'])-np.asarray(reference_slice)).__abs__().tolist()
            result['vmin']=min(result['vmin'],float(np.min(reference_slice)));result['vmax']=max(result['vmax'],float(np.max(reference_slice)))
        return result

    @app.get('/api/coordinate/jobs/{identifier}/artifacts/{name}')
    def artifact(identifier:str,name:str):
        if name not in ['result.npz','metrics.json','comparison.json','problem.json','observations.json']:raise ValueError('Unknown coordinate artifact')
        path=result_directory(identifier)/name
        if not path.is_file():raise ValueError('Artifact unavailable; attach a reference for comparison metrics')
        return FileResponse(path)
