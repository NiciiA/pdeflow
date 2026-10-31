"""Local scientific API; no training starts on import or UI load."""
from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import uuid

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
import torch

from pdeflow.core import ProblemSpec, ReconstructionResult
from pdeflow.equations import Expr, heat_expression, heat_problem
from pdeflow.inference import validate_compatibility,compatibility_report,selection_key,validate_inference_budget
from pdeflow.specialists import get_specialist
from pdeflow.numerics import heat_reference, smooth_initial
from pdeflow.observations import ObservationSet, sensor_observations
from pdeflow.runtime import InferenceConfig, RuntimeSpec, atomic_write, digest, environment, file_digest, read_json, write_json
from pdeflow.training import _read_checkpoint
from pdeflow.evaluation import export_result
from .jobs import JobManager, contained
from .observations import import_text, observation_rows, parse_rows


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ModelSelection(StrictModel):
    mode: str = Field(pattern=r"^(auto|explicit)$")
    id: str | None = Field(default=None,pattern=r"^[a-f0-9]{32}$")

class ReconstructionRequest(StrictModel):
    model_id: str | None = Field(default=None,pattern=r"^[a-f0-9]{32}$")
    model_selection: ModelSelection | None = None
    problem: dict
    rows: list[dict] = Field(min_length=1, max_length=20000)
    noise_std: float = Field(default=.03, ge=0)
    tolerance: float = Field(default=.1, gt=0)
    inference: dict = Field(default_factory=dict)
    runtime: dict = Field(default_factory=lambda: {"device": "cpu", "dtype": "float32", "precision": "full", "threads": 2})
    mode: str = "observation_pde"
    submission_token: str = Field(default_factory=lambda: uuid.uuid4().hex, max_length=80, min_length=1)
    mathematical_request: dict | None = None


class MathematicalRequest(StrictModel):
    request: dict
    model_id: str | None = None


class ProblemRequest(StrictModel):
    problem: dict


class ObservationRequest(ProblemRequest):
    rows: list[dict] = Field(default_factory=list, max_length=20000)
    noise_std: float = .03
    tolerance: float = .1
    text: str | None = Field(default=None, max_length=1_000_000)
    format: str = "csv"


class DemoRequest(StrictModel):
    problem: dict | None = None
    model_id: str | None = None
    diffusivity: float = Field(default=.07, gt=0)
    seed: int = Field(default=53, ge=0, le=2**31)
    sensors: int = Field(default=10, ge=1, le=512)
    noise_std: float = Field(default=.03, ge=0)


class ReferenceRequest(StrictModel):
    reference_id: str | None = None
    sampling_id: str | None = None
    policy: str = "native"
    shared_times_only: bool = False
    metrics: list[str] = Field(default_factory=lambda:["mse","rmse","mae","relative_l2"])

class ReferenceSolveRequest(StrictModel):
    request: dict
    inputs: dict
    solver: str = "auto"
    accuracy: dict | None = Field(default_factory=dict)
    seed: int = Field(default=53,ge=0,le=2**31)

class ReferenceInputPreview(ReferenceSolveRequest):
    field: str
    axis: int = Field(default=2,ge=0,le=2)
    index: int = Field(default=0,ge=0)

class ReferenceSamplingRequest(StrictModel):
    sampling: dict


class PreviewRequest(StrictModel):
    problem: dict
    rows: list[dict] = Field(default_factory=list,max_length=20000)
    field: str | None = None
    time_index: int = Field(default=0,ge=0)
    axis: int = Field(default=2,ge=0,le=2)
    index: int = Field(default=0,ge=0)
    noise_std: float = Field(default=.03,ge=0)


class CompareRequest(StrictModel):
    ids: list[str] = Field(min_length=2,max_length=4)


def create_app(model_roots=None, output_root=None, data_root=None, worker=None, workspace=None):
    from pdeflow.local import workspace_root, model_entries, run_lock
    local_workspace=workspace_root(workspace)
    explicit_roots=model_roots if model_roots is not None else os.environ.get('PDEFLOW_MODEL_ROOTS','').split(os.pathsep)
    roots = [Path(p).resolve() for p in explicit_roots if p]
    managed=model_roots is None and not os.environ.get('PDEFLOW_MODEL_ROOTS')
    output = Path(output_root or os.environ.get("PDEFLOW_JOB_ROOT", local_workspace/'workbench/jobs')).resolve()
    data = Path(data_root or os.environ.get("PDEFLOW_DATA_ROOT", local_workspace/'workbench/data')).resolve()
    data.mkdir(parents=True, exist_ok=True)
    models = {}

    def discover():
        from pdeflow.training_workflow import fixed_model_entries
        current = {}
        candidates=[(Path(entry['path']),entry) for entry in fixed_model_entries(local_workspace)] if managed else []
        for root in roots:
            if not root.is_dir(): continue
            for path in root.rglob("inference.pt"):
                if not path.resolve().is_relative_to(root): continue
                candidates.append((path,{}))
        for path,entry in candidates:
            try:
                identifier = file_digest(path)[:32]
                if identifier in current:
                    current[identifier]['export_kinds']=sorted(set(current[identifier].get('export_kinds',[])+[entry.get('export_kind','external')]))
                elif identifier in models:
                    current[identifier] = {**models[identifier],**entry,'path':path,'export_kinds':[entry.get('export_kind','external')]}
                else:
                    checkpoint, config, _ = _read_checkpoint(path)
                    if checkpoint['kind']!='inference':continue
                    current[identifier] = {**entry,"path": path, "config": config, "checkpoint": checkpoint,'export_kinds':[entry.get('export_kind','external')]}
            except (ValueError, RuntimeError, EOFError, KeyError, TypeError, OSError):
                continue
        # A trainer can atomically replace inference.pt. Retire the old content
        # ID rather than executing new weights under a stale selection.
        models.clear()
        models.update(current)
        return models

    def model(identifier):
        discover()
        if identifier not in models: raise ValueError("This checkpoint is unavailable. Refresh the model list and select a compatible export.")
        return models[identifier]

    @asynccontextmanager
    async def lifespan(app):
        discover()
        output.mkdir(parents=True,exist_ok=True)
        with run_lock(output):
            app.state.jobs = JobManager(output, **({"worker": worker} if worker else {}))
            try:yield
            finally:app.state.jobs.close()

    app = FastAPI(title="PDEFlow local workbench", lifespan=lifespan)
    from .coordinate import register as register_coordinate
    register_coordinate(app,local_workspace,data,output)
    from .training import register as register_training
    register_training(app,local_workspace)
    from .playground import register as register_playground
    register_playground(app,local_workspace,data,output)
    app.add_middleware(CORSMiddleware, allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"], allow_methods=["GET", "POST"], allow_headers=["Content-Type"])

    @app.middleware("http")
    async def limits(request: Request, call_next):
        request.state.request_id=uuid.uuid4().hex
        if request.url.path=='/api/playground/import':
            # The upload route streams and bounds large model files itself.
            return await call_next(request)
        chunks, length = [], 0
        async for chunk in request.stream():
            length += len(chunk)
            if length > (64*1024**2 if request.url.path=='/api/playground/import' else 2_000_000):
                return JSONResponse(status_code=413, content={"error": {"code": "TooLarge", "message": "Request exceeds 2 MB"}})
            chunks.append(chunk)
        request._body = b"".join(chunks)
        return await call_next(request)

    from fastapi.exceptions import RequestValidationError
    @app.exception_handler(RequestValidationError)
    async def schema_invalid(request,error):
        details=[{'path':list(e['loc']),'code':e['type'],'message':e['msg']} for e in error.errors()]
        message='Check the highlighted request fields before submitting.'
        if any('model_id' in e['path'] or 'model_selection' in e['path'] for e in details):message='Choose Automatic or a registered compatible checkpoint. Model selection cannot be empty.'
        return JSONResponse(status_code=422,content={'error':{'code':'InvalidRequest','message':message,'request_id':request.state.request_id,'details':details}})

    @app.exception_handler(RuntimeError)
    @app.exception_handler(FloatingPointError)
    @app.exception_handler(ValueError)
    @app.exception_handler(TypeError)
    @app.exception_handler(KeyError)
    @app.exception_handler(IndexError)
    async def invalid(request, error):
        return JSONResponse(status_code=422, content={"error": {"code": "ValidationError", "message": str(error),"request_id":request.state.request_id,"details":[{"code":type(error).__name__,"message":str(error)}]}})

    @app.exception_handler(FileNotFoundError)
    async def missing(request, error):
        return JSONResponse(status_code=404, content={"error": {"code": "NotFound", "message": "Unknown job or unavailable artifact"}})

    @app.get("/api/health")
    def health():
        return {"status": "ok", "runtime": environment(RuntimeSpec()), "backends": {"cpu": True, "mps": torch.backends.mps.is_available(), "cuda": torch.cuda.is_available()}}

    @app.get("/api/capabilities")
    def capabilities():
        return {"implemented": ["periodic scalar heat", "periodic variable-permeability Darcy 2D/3D", "periodic A+B reversible C chemistry 2D/3D", "on-grid observations", "linear CFM", "euler", "heun", "uniform", "quadratic"],
                "planned": ["waves", "fluids", "diffusion", "temporal/other-family refinement"], "fk": {"implemented":True,"stepper":"euler_maruyama","resampler":"systematic","refinement_combination":False}, "posterior_guarantee": False, "training_data_modes":["offline","online","hybrid"], "training_launch":"Explicit Train model action or CLI; opening the workbench never starts training"}

    @app.get("/api/models")
    def list_models():
        result=[]
        for key,entry in discover().items():
            c=entry['config'];sp=get_specialist(c.specialist)
            if 'architecture' not in entry:
                from pdeflow.models import make_model,TimeChannelLayout,architecture_metadata
                layout=TimeChannelLayout(len(sp.fields),c.grid)
                entry['architecture']=entry['checkpoint'].get('architecture') or architecture_metadata(make_model(c.model,layout,sp.encoder),c.model,layout)
            p=sp.default_problem(c.grid)
            label=f"{entry.get('run_name') or entry.get('run_id') or sp.id.title()} · {sp.id} {c.grid.dimension}D {list(c.grid.shape)} / {len(c.grid.times)} {'static frame' if c.grid.temporal_kind=='static' else 'physical frames'} · {entry['checkpoint']['progress']} updates · {'/'.join(entry['export_kinds'])} · {entry.get('status','bounded; scientifically unevaluated')}"
            result.append({'id':key,'name':label,'run_id':entry.get('run_id'),'run_name':entry.get('run_name'),'export_kinds':entry['export_kinds'],'architecture':entry['architecture'],'specialist':sp.id,'problem':p.to_dict(),'config':c.to_dict(),'capabilities':entry['checkpoint']['capabilities'],'steering':{'fk':c.formulation=='independent_gaussian_linear_cfm','sde_stepper':'euler_maruyama'},'training':{'updates':entry['checkpoint']['progress'],'diffusivity_range':c.data.diffusivity_range if sp.id=='heat' else None,'forcing_range':'unknown; generator is a smooth random mean-zero field' if sp.id=='darcy' else None,'status':entry.get('status','bounded training; scientifically unevaluated'),'data_mode':c.provider.mode if c.provider else 'offline','evaluation_identity':entry['checkpoint'].get('provider_state',{}).get('evaluation_identity')}})
        return result

    @app.post('/api/input-preview')
    def preview(payload: PreviewRequest):
        from pdeflow.workbench.visuals import input_preview
        return input_preview(**payload.model_dump())

    @app.get('/api/reference-capabilities')
    def reference_capabilities():
        from pdeflow.reference import capabilities
        return capabilities()

    @app.post('/api/reference-input-preview')
    def reference_input(payload: ReferenceInputPreview):
        from pdeflow.reference import ReferenceProblem,preview_inputs
        args=payload.model_dump();field=args.pop('field');axis=args.pop('axis');index=args.pop('index')
        return preview_inputs(ReferenceProblem(**args),field,axis,index)

    @app.post('/api/references',status_code=201)
    def reference_solve(payload: ReferenceSolveRequest):
        from pdeflow.reference import ReferenceProblem,create
        identifier=uuid.uuid4().hex
        manifest=create(ReferenceProblem(**payload.model_dump()),contained(data,identifier),identifier)
        return reference_summary(identifier)

    def reference_summary(identifier):
        manifest=read_json(contained(data,identifier)/'reference.json')
        p=manifest['problem']
        return {**manifest,'problem':{**p,'known_fields':{name:'Known spatial array; exported with inverse request' for name in p.get('known_fields',{})}}}

    @app.get('/api/references/{identifier}')
    def get_reference(identifier: str):
        return reference_summary(identifier)

    @app.get('/api/references/{identifier}/slice')
    def reference_slice(identifier: str,field_index:int=Query(0,ge=0),time_index:int=Query(0,ge=0),axis:int=Query(2,ge=0,le=2),index:int=Query(0,ge=0),sampling_id:str|None=None):
        from pdeflow.reference import preview
        directory=contained(data,identifier);result=preview(directory,field_index,time_index,axis,index)
        if sampling_id:
            rows=read_json(contained(directory/'samplings',sampling_id)/'rows.json');total_rows=len(rows)
            p=ProblemSpec.from_dict(read_json(directory/'problem.json'))
            rows=[r for r in rows if r['field']==p.fields[field_index].name and (p.grid.temporal_kind=='static' or r['time']==p.grid.times[time_index]) and (p.grid.dimension!=3 or abs(r[p.grid.axis_names[axis]]-float(p.grid.coordinates()[axis][index]))<1e-8)]
            result.update(off_slice_observation_count=total_rows-len(rows),observation_count=len(rows),observations=rows[::max(1,math.ceil(len(rows)/512))])
        return result

    @app.post('/api/references/{identifier}/sample',status_code=201)
    def reference_sample(identifier:str,payload:ReferenceSamplingRequest):
        from pdeflow.reference.observations import sample,ObservationSamplingSpec
        directory=contained(data,identifier);sampling_id=uuid.uuid4().hex
        result=sample(directory,ObservationSamplingSpec(**payload.sampling),contained(directory/'samplings',sampling_id))
        with (directory/'samplings'/sampling_id/'observations.json').open() as stream:indices=json.load(stream)['indices']
        return {**{k:v for k,v in result.items() if k not in ['reference_path','path']},'sampling_id':sampling_id,'indices':indices}

    @app.get('/api/references/{identifier}/artifacts/{name}')
    def reference_artifact(identifier:str,name:str,sampling_id:str|None=None):
        directory=contained(data,identifier)
        if name=='reference-field.json':
            from pdeflow.reference.imports import export_reference
            return JSONResponse(export_reference(directory))
        choices={'reference.npz':directory/'reference.npz','reference-problem.json':directory/'reference-problem.json','reference.json':directory/'reference.json'}
        if sampling_id:
            sampling=contained(directory/'samplings',sampling_id)
            choices.update({name:sampling/name for name in ['observations.json','inverse-request.json','sampling.json','rows.json']})
            if name=='observations.csv':
                import csv,io
                def export_csv(rows):
                    stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=list(rows[0]) if rows else ["field","value"])
                    writer.writeheader();writer.writerows(rows);return stream.getvalue()
                from fastapi.responses import Response
                return Response(export_csv(read_json(sampling/'rows.json')),media_type='text/csv',headers={'Content-Disposition':'attachment; filename="observations.csv"'})
        if name not in choices:raise ValueError('Unknown reference artifact name')
        return FileResponse(choices[name],filename=name)

    def storage_root():
        from pdeflow.local import project_root
        root=project_root()/'outputs'
        return root if local_workspace.is_relative_to(root) else local_workspace

    @app.get('/api/storage')
    def storage_inventory(refresh:bool=False):
        from pdeflow.storage import inventory
        return inventory(storage_root(),refresh)

    @app.post('/api/storage/preview')
    def storage_preview():
        from pdeflow.storage import preview
        return preview(storage_root())

    @app.post('/api/storage/pin')
    def storage_pin(payload:dict):
        from pdeflow.storage import pin
        if set(payload)!={'path','pinned'} or type(payload['pinned'])!=bool:raise ValueError('Provide path and boolean pinned')
        return pin(storage_root(),payload['path'],payload['pinned'])

    @app.post('/api/analytical-examples')
    def analytical_examples(payload:ProblemRequest):
        from pdeflow.reference.analytical import templates
        return templates(ProblemSpec.from_dict(payload.problem))

    @app.post('/api/analytical-verify')
    def analytical_verify(payload:ReferenceSolveRequest):
        from pdeflow.reference.solvers import compile_problem,ReferenceProblem
        from pdeflow.reference.analytical import verify
        p,_=compile_problem(ReferenceProblem(payload.request,payload.inputs))
        _,report=verify(p,payload.inputs['initial_u'],payload.inputs['analytical_reference']['expression'])
        return report

    @app.get('/api/training-coverage')
    def training_coverage():
        from pdeflow.catalogue import coverage
        return coverage(local_workspace)

    @app.get('/api/training-plan')
    def training_plan(suite:str='full'):
        from pdeflow.suites import plan
        return plan(suite,workspace=local_workspace)

    @app.post('/api/checkpoint-compatibility')
    def checkpoint_compatibility(payload:ProblemRequest):
        p=ProblemSpec.from_dict(payload.problem)
        return [compatibility_report(p,e['config'],candidate=i) for i,e in discover().items()]

    @app.get('/api/training-suites')
    def training_suites():
        from pdeflow.suites import records,status
        return [status(row['path']) for row in records(local_workspace)[:30]]

    @app.get('/api/training-runs')
    def training_history():
        from pdeflow.workbench.visuals import training_runs
        return training_runs(local_workspace)

    @app.get('/api/training-presets')
    def local_training_presets():
        from pdeflow.local import presets
        return presets()

    @app.get('/api/jobs/{identifier}/input-preview')
    def saved_input(identifier: str,field: str|None=None,time_index:int=Query(0,ge=0),axis:int=Query(2,ge=0,le=2),index:int=Query(0,ge=0)):
        from pdeflow.workbench.visuals import input_preview
        directory,_=finished(identifier)
        p=ProblemSpec.from_dict(read_json(directory/'result/problem.json'))
        obs=ObservationSet.from_dict(read_json(directory/'result/observations.json'),p.grid)
        return input_preview(p.to_dict(),observation_rows(obs,p),field,time_index,axis,index,obs.noise_std)

    @app.get('/api/jobs/{identifier}/plots')
    def plots(identifier: str):
        from pdeflow.workbench.visuals import result_plots
        directory,_=finished(identifier)
        return result_plots(directory/'result')

    @app.post('/api/compare')
    def compare(payload: CompareRequest):
        from pdeflow.workbench.visuals import compare_results
        if len(set(payload.ids))!=len(payload.ids):raise ValueError('Choose distinct completed runs')
        return compare_results([(identifier,finished(identifier)[0]/'result') for identifier in payload.ids])

    @app.get("/api/mathematics/presets")
    def equation_presets():
        from pdeflow.symbolic import presets
        return presets()

    @app.post("/api/mathematics/parse")
    def parse_mathematics(payload: MathematicalRequest):
        from pdeflow.symbolic import analyze
        try:
            report = analyze(payload.request)
        except (KeyError, TypeError, IndexError, RecursionError) as error:
            raise ValueError("Incomplete or malformed mathematical request") from error
        if report['problem'] and payload.request['measurements'].get('rows'):
            try:
                p=ProblemSpec.from_dict(report['problem']);obs=parse_rows(p,payload.request['measurements']['rows'],payload.request['measurements']['noise_std'])
                get_specialist(p.family).validate_observations(p,obs)
            except ValueError as error:
                report['reasons'].append(str(error));report['problem']=None
        choices=[]; exclusions=[]
        p=ProblemSpec.from_dict(report['problem']) if report['problem'] else None
        for identifier,entry in sorted(discover().items(),key=lambda item:selection_key(*item)):
            try:
                if p is None: raise ValueError('No executable problem for this draft')
                validate_compatibility(p,entry['config'],InferenceConfig())
                choices.append(identifier)
            except ValueError as error: exclusions.append({'id':identifier,'specialist':entry['config'].specialist,'reason':str(error),'compatibility':compatibility_report(p,entry['config'],candidate=identifier) if p else None})
        selected=None
        if payload.model_id:
            if payload.model_id in choices: selected=payload.model_id
            else: report['reasons'].append('Explicit checkpoint override is incompatible; select Automatic to route again')
        elif choices: selected=choices[0]
        report['status']['checkpoint']=bool(selected)
        report['routing']={'recognized_formulation':report['canonical_source'],'candidate_specialists':[report['recognized_family']] if report['recognized_family']!='custom' else [],'compatible_checkpoints':choices,'exclusions':exclusions,'selected_specialist':report['recognized_family'] if p else None,'selected_checkpoint':selected,'selection_reason':'Explicit compatible override' if selected and payload.model_id else 'Compatible exports prioritized by optimizer updates, completed budget, then validation-selected role; no truth or quality guarantee' if selected else 'No compatible checkpoint selected','unknown_ranges':['Darcy forcing/permeability generalization unvalidated'] if report['recognized_family']=='darcy' else ['Chemistry parameter/initial-condition generalization unvalidated'] if report['recognized_family']=='chemistry' else [],'range_warnings':[]}
        if p:
            from pdeflow.catalogue import coverage
            from pdeflow.runtime import Config
            report['routing']['targets']=[{k:t[k] for k in ('id','grid','state','maximum_updates','updates','compatible_discoverable_checkpoint','command')} for t in coverage(local_workspace)['targets'] if compatibility_report(p,Config.from_dict(t['resolved_config'],check_device=False))['compatible']]
        if p and p.family=='heat':
            for identifier in choices:
                lo,hi=models[identifier]['config'].data.diffusivity_range
                if not lo<=p.known_parameters['diffusivity']<=hi:report['routing']['range_warnings'].append({'id':identifier,'reason':'Diffusivity outside recorded training range; generalization unvalidated'})

        if p and p.family=='chemistry':
            for identifier in choices:
                chemistry=models[identifier]['config'].chemistry
                for name,value in p.known_parameters.items():
                    bounds=chemistry.diffusivity_range if name.startswith('D_') else chemistry.forward_rate_range if name=='k_f' else chemistry.reverse_rate_range
                    if not bounds[0]<=value<=bounds[1]:report['routing']['range_warnings'].append({'id':identifier,'reason':f'{name} outside recorded chemistry training range; generalization unvalidated'})
        return report

    @app.post("/api/problems/validate")
    def validate_problem(payload: ProblemRequest):
        p=ProblemSpec.from_dict(payload.problem)
        try:
            get_specialist(p.family).validate(p,'independent_gaussian_linear_cfm','heun','uniform'); implemented=True
        except ValueError: implemented=False
        return {'syntax_valid':True,'canonical':p.to_dict(),'residual_implemented':implemented,'reference_solver':implemented,'equation_preview':p.family,'checkpoint_compatibility_requires_validation':True}

    @app.post("/api/observations/validate")
    def validate_observations(payload: ObservationRequest):
        p = ProblemSpec.from_dict(payload.problem)
        rows = import_text(p, payload.text, payload.format, payload.noise_std, payload.tolerance) if payload.text is not None else payload.rows
        obs = parse_rows(p, rows, payload.noise_std, payload.tolerance)
        return {"rows": observation_rows(obs, p), "count": len(rows), "observations": obs.to_dict()}

    def resolve(payload):
        raise ValueError("Legacy architecture — incompatible with current model. Use the heat playground joint model workflow. Existing artifacts remain available in storage.")
        p = ProblemSpec.from_dict(payload.problem)
        validate_inference_budget(p,InferenceConfig(**payload.inference))
        selection=payload.model_selection
        if selection:
            if payload.model_id is not None:raise ValueError('Use model_selection or legacy model_id, not both')
            if selection.mode=='explicit':
                if not selection.id:raise ValueError('Explicit selection needs a registered checkpoint ID')
                identifier=selection.id
            else:
                if selection.id is not None:raise ValueError('Automatic selection has no explicit ID')
                candidates=[];exclusions=[]
                for key,candidate in sorted(discover().items(),key=lambda item:selection_key(*item)):
                    try:validate_compatibility(p,candidate['config'],InferenceConfig(**payload.inference));candidates.append(key)
                    except ValueError as error:
                        if candidate['config'].specialist==p.family:exclusions.append(compatibility_report(p,candidate['config'],candidate=key)['summary'])
                if not candidates:raise ValueError('No compatible trained model is available for this equation and grid. Train or import the matching layout. '+(' '.join(exclusions[:2])))
                identifier=candidates[0]
        else:
            if payload.model_id is None:raise ValueError('Choose Automatic or a registered compatible checkpoint before submission')
            identifier=payload.model_id
        entry = model(identifier)
        payload.model_id=identifier
        if payload.mathematical_request is not None:
            report = parse_mathematics(MathematicalRequest(request=payload.mathematical_request, model_id=payload.model_id))
            if report["problem"] is None or ProblemSpec.from_dict(report["problem"]) != p:
                raise ValueError("Current mathematical source is unsupported or differs from the executable problem; parse and validate the current draft")
            measurements = payload.mathematical_request["measurements"]
            if measurements.get("rows") != payload.rows or measurements.get("noise_std") != payload.noise_std:
                raise ValueError("Mathematical request measurements/noise differ from reconstruction inputs")
        if len(json.dumps(payload.problem)) > 1_000_000: raise ValueError("Problem specification too large")
        get_specialist(p.family).validate(p, entry["config"].formulation, "heun", "uniform")
        inf = InferenceConfig(**payload.inference).validate()
        runtime = RuntimeSpec(**payload.runtime)
        runtime.resolve()
        if inf.steps > 512 or inf.particles > (256 if payload.mode in ("fk","importance") else 16) or runtime.threads > 8:
            raise ValueError("Local job limit: <=512 steps, <=16 samples, <=8 CPU threads")
        validate_compatibility(p, entry["config"], inf)
        obs = parse_rows(p, payload.rows, payload.noise_std, payload.tolerance)
        get_specialist(p.family).validate_observations(p,obs)
        size = inf.particles * len(p.fields) * len(p.grid.times) * math.prod(p.grid.shape) * (8 if runtime.dtype == "float64" else 4)
        if size > 128 * 1024**2: raise ValueError("Field sample storage exceeds 128 MiB local limit")
        warnings = []
        r = entry["checkpoint"]["config"].get("data", {}).get("diffusivity_range")
        if p.family == "heat" and r and not r[0] <= p.known_parameters["diffusivity"] <= r[1]: warnings.append("Diffusivity lies outside recorded training range; parameter generalization is unvalidated")
        if payload.mode not in ("unguided", "observation", "observation_pde", "fk", "importance"): raise ValueError("Unsupported guidance mode; FK is planned")
        if payload.mode in ("fk","importance"):
            from pdeflow.process import ProcessConfig
            if inf.fk is None:raise ValueError("FK requires explicit inference.fk configuration")
            process=ProcessConfig(**(inf.controller or {})).validate()
            if process.refine or process.fine_model_id or process.max_attempts!=1:raise ValueError("FK plus refinement/retry is unsupported")
            if inf.particles>process.max_particles:raise ValueError("FK particle count exceeds controller max_particles")
            if entry["config"].formulation!="independent_gaussian_linear_cfm":raise ValueError("Checkpoint lacks FK score capability")
        fine_entry=None
        if inf.controller and inf.controller.get('fine_model_id'):
            fine_entry=model(inf.controller['fine_model_id'])
            from pdeflow.refinement import refine_request
            from types import SimpleNamespace
            refine_request(p,obs,SimpleNamespace(config=fine_entry['config']),inf,inf.controller.get('max_grid_points',32768))
            fine_size=inf.particles*len(p.fields)*len(p.grid.times)*math.prod(fine_entry['config'].grid.shape)*(8 if runtime.dtype=='float64' else 4)
            if fine_size>128*1024**2:raise ValueError('Fine field exceeds local storage limit')
        request = {"model_id": identifier,"selection":{"mode":selection.mode if selection else "explicit","resolved_checkpoint":identifier,"rationale":"Compatible update-count/completed-budget priority; no quality ranking" if selection and selection.mode=="auto" else "Explicit registered compatible checkpoint"}, "problem": p.to_dict(), "observations": obs.to_dict(), "inference": asdict(inf), "runtime": asdict(runtime), "mode": payload.mode}
        if fine_entry is not None:
            request['refinement_checkpoint']=str(fine_entry['path'])
            request['refinement_model_id']=inf.controller['fine_model_id']
        if payload.mathematical_request is not None:
            request["mathematical_request"] = payload.mathematical_request
        return entry, request, size, warnings

    @app.post("/api/reconstruction/validate")
    def validate_reconstruction(payload: ReconstructionRequest):
        _, resolved, size, warnings = resolve(payload)
        return {"valid": True, "resolved": resolved, "sample_storage_bytes": size, "storage_note": "Array storage only, not full CPU/GPU memory prediction", "warnings": warnings}

    @app.post("/api/jobs", status_code=202)
    def submit(payload: ReconstructionRequest):
        entry, resolved, _, _ = resolve(payload)
        return app.state.jobs.submit(resolved, entry["path"], payload.submission_token)

    @app.get("/api/jobs")
    def jobs(): return app.state.jobs.list()

    @app.get("/api/jobs/{identifier}")
    def job(identifier: str): return app.state.jobs.get(identifier)

    @app.post("/api/jobs/{identifier}/cancel")
    def cancel(identifier: str): return app.state.jobs.cancel(identifier)

    @app.post("/api/demos")
    def synthetic(payload: DemoRequest):
        entry = model(payload.model_id) if payload.model_id else None
        if entry is None and payload.problem is None:raise ValueError('Supply a current physical problem, or select a registered checkpoint')
        grid = entry['config'].grid if entry else ProblemSpec.from_dict(payload.problem).grid
        if payload.sensors > math.prod(grid.shape): raise ValueError("More sensors than grid nodes")
        if payload.problem is not None: p=ProblemSpec.from_dict(payload.problem)
        else:
            p=get_specialist(entry["config"].specialist).default_problem(grid)
            if p.family=="heat":p=heat_problem(grid,payload.diffusivity)
        if entry:validate_compatibility(p,entry["config"],InferenceConfig())
        from pdeflow.reference import create
        from pdeflow.reference.examples import from_problem
        from pdeflow.reference.observations import sample,ObservationSamplingSpec
        identifier=uuid.uuid4().hex;directory=contained(data,identifier)
        create(from_problem(p,payload.seed),directory,identifier)
        result=sample(directory,ObservationSamplingSpec(count=payload.sensors,seed=payload.seed+1,noise_std=payload.noise_std),directory/'legacy-observations')
        return {"problem":p.to_dict(),"rows":result['rows'],"reference_id":identifier,
                "label":"Synthetic observations from explicit reference generation; reference is detached from inference"}

    def finished(identifier):
        record = app.state.jobs.get(identifier)
        if record["status"] != "succeeded": raise ValueError("Results require a succeeded job")
        directory = contained(output, identifier)
        return directory, read_json(directory / "result/metrics.json")

    import threading
    evaluation_lock=threading.RLock()
    @app.post('/api/jobs/{identifier}/evaluation-context')
    def associate_evaluation(identifier:str,payload:ReferenceRequest):
        directory=contained(output,identifier);app.state.jobs.get(identifier)
        if payload.reference_id:
            from pdeflow.reference import load
            refproblem,_,_=load(contained(data,payload.reference_id))
            original=ProblemSpec.from_dict(read_json(directory/'request.json')['problem'])
            if refproblem!=original:raise ValueError('Reference association is stale or incompatible with this saved inverse problem')
            write_json(directory/'evaluation-context.json',payload.model_dump())
        else:(directory/'evaluation-context.json').unlink(missing_ok=True)
        return {'associated':bool(payload.reference_id),'reference_in_worker':False}

    @app.get('/api/jobs/{identifier}/inspection')
    def spatial_inspection(identifier:str,field_index:int=0,time_index:int=0,point:str|None=None,sample_index:int=-1):
        from .inspection import inspect
        directory,_=finished(identifier)
        return inspect(directory/'result',field_index,time_index,[int(k) for k in point.split(',')] if point else None,sample_index)

    @app.get('/api/jobs/{identifier}/generation')
    def generation_frames(identifier:str,frame:int|None=None,particle:int=0,kind:str='current_physical',axis:int=2,index:int=0,field_index:int|None=None,time_index:int|None=None):
        from .inspection import generation
        directory,_=finished(identifier)
        return generation(directory/'result',frame,particle,kind,axis,index,field_index,time_index)

    @app.get("/api/jobs/{identifier}/result")
    def result(identifier: str):
        directory, report = finished(identifier)
        with evaluation_lock:
            if (directory/'evaluation-context.json').exists() and not (directory/'reference.json').exists():
                try:
                    attach_reference(identifier,ReferenceRequest(**read_json(directory/'evaluation-context.json')))
                    report=read_json(directory/'result/metrics.json')
                except (ValueError,OSError) as error:
                    report={**report,'evaluation_error':str(error)}
        original=read_json(directory / "request.json")
        resolved={**original,'problem':read_json(directory/'result/problem.json'),'observations':read_json(directory/'result/observations.json')}
        if resolved['problem']!=original['problem']:
            resolved['model_id']=report['provenance']['checkpoint_sha256'][:32]
            resolved['inference']={**original['inference'],'seed':report['provenance']['sampling_seed']}
            if original.get('mathematical_request'):
                resolved['mathematical_request']={**original['mathematical_request'],'domain':resolved['problem']['grid']}
        # This is a final-state view. original_request is the reproducible process input.
        return {"metrics": report, "request": resolved, "original_request": original, "job": app.state.jobs.get(identifier),
                "reference_available": (directory / "reference.json").exists(), "baselines":{p.name.removeprefix("baseline_"):read_json(p/"metrics.json") for p in directory.glob("baseline_*") if (p/"metrics.json").exists()}}

    @app.post("/api/jobs/{identifier}/reference")
    def attach_reference(identifier: str, payload: ReferenceRequest):
        directory, report = finished(identifier)
        p = ProblemSpec.from_dict(read_json(directory / "result/problem.json"))
        obs = ObservationSet.from_dict(read_json(directory / "result/observations.json"), p.grid)
        reference = None
        evaluation_report = None
        if payload.reference_id:
            refdir = contained(data, payload.reference_id)
            refproblem = ProblemSpec.from_dict(read_json(refdir / "problem.json"))
            from pdeflow.reference import load
            if (refdir/'reference.json').exists():refproblem,refvalues,refmanifest=load(refdir)
            else:
                # Read-only compatibility for pre-artifact synthetic demonstrations.
                with np.load(refdir/'reference.npz',allow_pickle=False) as a:refvalues=a['values'].copy()
                refmanifest={'diagnostics':{'classification':'legacy numerical demonstration; detailed solver provenance not recorded'}}
            if not payload.sampling_id:
                for candidate in sorted((refdir/'samplings').glob('*/observations.json')):
                    if read_json(candidate)==obs.to_dict():payload.sampling_id=candidate.parent.name;break
            if payload.sampling_id:
                original=ObservationSet.from_dict(read_json(contained(refdir/'samplings',payload.sampling_id)/'observations.json'),p.grid)
                if original.to_dict()!=obs.to_dict():raise ValueError('Sampling artifact does not match the saved reconstruction observations')
            from pdeflow.evaluation.comparison import align,EvaluationSpec
            evaluation_spec=EvaluationSpec(payload.policy,payload.shared_times_only,metrics=tuple(payload.metrics))
            aligned,time_indices,_=align(refvalues,refproblem,p,evaluation_spec)
            if len(time_indices)==len(p.grid.times):reference=torch.from_numpy(aligned[None])
        with np.load(directory / "result/result.npz", allow_pickle=False) as a:
            samples = torch.from_numpy(a["samples"].copy())
            population_weights=torch.from_numpy(a["weights"].copy()) if "weights" in a else None
        reconstruction = ReconstructionResult(samples, report["cost"], report["provenance"], report["stop_reason"], report["corrected"],weights=population_weights)
        if payload.reference_id:
            from pdeflow.evaluation.comparison import compare
            evaluation_report=compare(samples.numpy(),p,obs,refvalues,refproblem,population_weights.numpy() if population_weights is not None else None,evaluation_spec)
        updated=export_result(directory / "result", reconstruction, p, obs, reference)
        if evaluation_report:
            updated['evaluation']=evaluation_report;updated['reference_diagnostics']=refmanifest['diagnostics'];updated['reference_id']=payload.reference_id
            write_json(directory/'result/metrics.json',updated);write_json(directory/'result/comparison.json',evaluation_report)
        for baseline_dir in directory.glob('baseline_*'):
            if not (baseline_dir/'metrics.json').exists():continue
            previous=read_json(baseline_dir/'metrics.json')
            with np.load(baseline_dir/'result.npz',allow_pickle=False) as a:baseline_samples=torch.from_numpy(a['samples'].copy())
            baseline=ReconstructionResult(baseline_samples,previous['cost'],previous['provenance'],previous['stop_reason'],previous['corrected'])
            baseline_metrics=export_result(baseline_dir,baseline,p,obs,reference)
            if evaluation_report:
                baseline_metrics['evaluation']=compare(baseline_samples.numpy(),p,obs,refvalues,refproblem,spec=evaluation_spec)
                write_json(baseline_dir/'metrics.json',baseline_metrics);write_json(baseline_dir/'comparison.json',baseline_metrics['evaluation'])
        if payload.reference_id: write_json(directory / "reference.json", {"id": payload.reference_id,"sampling_id":payload.sampling_id,"evaluation_spec":evaluation_spec.__dict__})
        else:
            (directory / "reference.json").unlink(missing_ok=True)
            (directory / "evaluation-context.json").unlink(missing_ok=True)
        return {"attached": bool(payload.reference_id)}

    @app.get("/api/jobs/{identifier}/slice")
    def slice_preview(identifier: str, time_index: int = Query(0, ge=0), axis: int = Query(2, ge=0, le=2),
                      index: int = Query(0, ge=0), field_index: int = Query(0, ge=0), sample: int = Query(-1, ge=-2, le=255)):
        directory, report = finished(identifier)
        p = ProblemSpec.from_dict(read_json(directory / "result/problem.json")); grid = p.grid
        if field_index >= len(p.fields): raise ValueError("Field index out of bounds")
        if time_index >= len(grid.times) or (grid.dimension == 3 and index >= grid.shape[axis]): raise ValueError("Slice/time index out of bounds")
        with np.load(directory / "result/result.npz", allow_pickle=False) as a:
            ensemble = a["samples"][0, :, field_index, time_index]
            if sample >= len(ensemble): raise ValueError("Sample index out of bounds")
            weights=a["weights"][0] if "weights" in a else np.full(len(ensemble),1/len(ensemble))
            weighted_mean=np.einsum("p,p...->...",weights,ensemble)
            field = np.sqrt(np.einsum("p,p...->...",weights,(ensemble-weighted_mean)**2)) if sample == -2 else weighted_mean if sample == -1 else ensemble[sample]
            reference = a["reference"][0, field_index, time_index] if "reference" in a else None
        spatial_axes = list(grid.axis_names)
        if grid.dimension == 3:
            field = np.take(field, index, axis=axis)
            if reference is not None: reference = np.take(reference, index, axis=axis)
            spatial_axes.pop(axis)
        native_shape=list(grid.shape)
        strides=[max(1,math.ceil(n/128)) for n in field.shape]
        field=field[tuple(slice(None,None,k) for k in strides)]
        if reference is not None:reference=reference[tuple(slice(None,None,k) for k in strides)]
        obs = ObservationSet.from_dict(read_json(directory / "result/observations.json"), grid)
        points = [r for r in observation_rows(obs, p) if r["field"] == p.fields[field_index].name and (grid.temporal_kind == "static" or abs(r["time"] - grid.times[time_index]) < 1e-9) and (grid.dimension != 3 or abs(r[grid.axis_names[axis]] - float(grid.coordinates()[axis][index])) < 1e-8)]
        arrays = [field] + ([reference] if reference is not None and sample != -2 else [])
        return {"values": field.tolist(), "reference": reference.tolist() if reference is not None and sample != -2 else None,
                "error": np.abs(field - reference).tolist() if reference is not None and sample != -2 else None,
                "shape": list(field.shape), "axes": spatial_axes, "native_shape":native_shape,"display_downsampled":any(k>1 for k in strides),"coordinates": {a: grid.coordinates()[grid.axis_names.index(a)][::k].tolist() for a,k in zip(spatial_axes,strides)},
                "field": p.fields[field_index].name, "physical_time": grid.times[time_index] if grid.temporal_kind == "transient" else None, "slice_coordinate": grid.coordinates()[axis][index].item() if grid.dimension == 3 else None,
                "vmin": min(float(a.min()) for a in arrays), "vmax": max(float(a.max()) for a in arrays), "observations": points[::max(1,math.ceil(len(points)/512))],
                "observation_count":len(points),"observations_downsampled":len(points)>512, "units": "nondimensional", "statistic": "empirical spread (not calibrated uncertainty)" if sample == -2 else "field"}

    @app.get("/api/jobs/{identifier}/experiment.zip")
    def experiment_archive(identifier:str):
        import io,zipfile
        from fastapi.responses import StreamingResponse
        directory,_=finished(identifier);buffer=io.BytesIO()
        manifest={'schema_version':1,'kind':'workbench_experiment_export','job_id':identifier,'inference_request':'reconstruction-request.json','reference_in_worker':False,'created_at':__import__('pdeflow.local',fromlist=['now']).now()}
        with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
            archive.write(directory/'request.json','reconstruction-request.json')
            for file in (directory/'result').iterdir():
                if file.is_file() and not file.name.startswith('.'):archive.write(file,'reconstruction/'+file.name)
            for baseline in directory.glob('baseline_*'):
                for file in baseline.iterdir():
                    if file.is_file() and not file.name.startswith('.'):archive.write(file,baseline.name+'/'+file.name)
            if (directory/'reference.json').exists():
                attached=read_json(directory/'reference.json');ref=contained(data,attached['id'])
                manifest['reference']=attached
                for name in ['reference.json','reference-problem.json','reference.npz','problem.json']:
                    if (ref/name).exists():archive.write(ref/name,'reference/'+name)
                if attached.get('sampling_id'):
                    sampling=contained(ref/'samplings',attached['sampling_id'])
                    for name in ['sampling.json','observations.json','inverse-request.json','rows.json','clean-samples-evaluation-only.npz']:archive.write(sampling/name,'sampling/'+name)
            archive.writestr('experiment.json',json.dumps(manifest,indent=2))
        buffer.seek(0)
        return StreamingResponse(buffer,media_type='application/zip',headers={'Content-Disposition':'attachment; filename="pdeflow-experiment.zip"'})

    @app.get("/api/jobs/{identifier}/artifacts/{name}")
    def artifact(identifier: str, name: str):
        directory, _ = finished(identifier)
        choices = {"samples.npz": directory / "result/result.npz", "metrics.json": directory / "result/metrics.json",
                   "configuration.json": directory / "request.json", "figure.png": directory / "result/reconstruction.png","comparison.json":directory/"result/comparison.json"}
        if name not in choices: raise ValueError("Unknown artifact name")
        return FileResponse(choices[name], filename=name)

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument('--host',default='127.0.0.1');parser.add_argument('--workspace');parser.add_argument('--reload',action='store_true')
    args = parser.parse_args()
    from pdeflow.local import serve
    serve(args.workspace,args.host,args.port,args.reload)


if __name__ == "__main__": main()
