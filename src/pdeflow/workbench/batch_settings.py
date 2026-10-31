"""Batch experiment/application control plane; no numerical trainer implementation."""
from pathlib import Path
import uuid
from pydantic import BaseModel,ConfigDict,Field
from pdeflow import training_workflow as workflow,batch_experiments as experiments
from pdeflow.coordinate.contracts import CoordinateConfig
from pdeflow.training_settings import apply_training_settings,resolve_batches

class Request(BaseModel):
    model_config=ConfigDict(extra='forbid')
    request:dict
    settings:dict=Field(default_factory=dict)
    budget_policy:str='keep_updates'
    experiment_id:str|None=None
    recommended:bool=False
    selections:dict|None=None

class Experiment(BaseModel):
    model_config=ConfigDict(extra='forbid')
    request:dict
    options:dict=Field(default_factory=dict)


def register(app,workspace):
    root=Path(workspace)/'batch-experiments'
    def location(identifier):
        if not identifier.isalnum() or len(identifier)!=32:raise ValueError('Invalid batch experiment ID')
        path=root/identifier
        if not (path/'experiment.json').exists():raise ValueError('Unknown batch experiment')
        return path
    @app.post('/api/training/batch/apply')
    def apply(payload:Request):
        resolved=workflow.plan(payload.request);c=CoordinateConfig(**resolved['config']).validate()
        settings=payload.settings
        if payload.experiment_id:settings=experiments.selected_settings(c,experiments.status(location(payload.experiment_id)),payload.selections,payload.recommended)
        selection={'method':'benchmark recommendation' if payload.recommended else 'manual benchmark selection','experiment_id':payload.experiment_id} if payload.experiment_id else None
        new,report=apply_training_settings(c,settings,payload.budget_policy,selection=selection)
        request={**payload.request,'advanced':{**payload.request.get('advanced',{}),'batch_size':new.batch_size,'ae_steps':new.ae_steps,'evolution_steps':new.evolution_steps,'flow_steps':new.flow_steps,'training_options':new.training_options}}
        return {'request':request,'resolved':workflow.plan(request),'application':report,'state':'applied for next NEW training submission; training has not started'}
    @app.post('/api/training/batch/estimate')
    def estimate(payload:Experiment):return experiments.plan(CoordinateConfig(**workflow.plan(payload.request)['config']).validate(),**payload.options)
    @app.get('/api/training/batch/benchmarks')
    def saved():
        from pdeflow.runtime import read_json
        rows=[read_json(p) for p in root.glob('*/experiment.json')]
        return [{'id':r['id'],'state':r['state'],'mode':r['plan']['mode'],'created_unix':r.get('created_unix',0)} for r in sorted(rows,key=lambda r:r.get('created_unix',0),reverse=True)[:20]]
    @app.post('/api/training/batch/benchmark',status_code=202)
    def benchmark(payload:Experiment):
        if any(r['state'] in workflow.ACTIVE for r in workflow.runs(workspace)):raise ValueError('Stop the active training worker before benchmarking batch sizes')
        root.mkdir(parents=True,exist_ok=True)
        from pdeflow.runtime import read_json
        for existing in root.glob('*/experiment.json'):
            if read_json(existing)['state'] in experiments.ACTIVE:raise ValueError('A batch benchmark is active; inspect/cancel it first')
        path=root/uuid.uuid4().hex
        return experiments.start(CoordinateConfig(**workflow.plan(payload.request)['config']).validate(),path,**payload.options)
    @app.get('/api/training/batch/benchmark/{identifier}')
    def status(identifier:str):return experiments.status(location(identifier))
    @app.post('/api/training/batch/benchmark/{identifier}/stop')
    def stop(identifier:str):return experiments.stop(location(identifier))
