"""Thin UI adapter to the same persistent training workflow used by the CLI."""
from fastapi import Query
from pydantic import BaseModel, ConfigDict, Field
from pdeflow import training_workflow as workflow
from pdeflow.runtime import read_json
from pdeflow.local import workspace_root


class TrainingRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    problem: str = 'heat-wave-2d'
    model: str = 'resolution-flexible'
    budget: str = 'pipeline-check'
    device: str = 'auto'
    advanced: dict = Field(default_factory=dict)
    allow_poor_representation: bool = False
    representation_max_mse_ratio: float = Field(default=1., gt=0, le=1)


class StartRequest(TrainingRequest):
    submission_token: str = Field(min_length=1, max_length=100)


class ResumeRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    allow_poor_representation: bool = False


class ModelContext(BaseModel):
    model_config = ConfigDict(extra='forbid')
    problem: dict | None = None


class Selection(ModelContext):
    id: str
    adopt_training_problem: bool = False


class Extension(BaseModel):
    model_config = ConfigDict(extra='forbid')
    additional_updates: int = Field(ge=1, le=1_000_000)


def register(app, workspace):
    from .batch_settings import register as register_batches
    register_batches(app,workspace)
    @app.get('/api/training/options')
    def options():
        return workflow.options()

    @app.post('/api/training/plan')
    def plan(payload: TrainingRequest):
        return workflow.plan(payload.model_dump())

    preview_scales={}
    @app.post('/api/training/loss-preview')
    def loss_preview(payload: TrainingRequest):
        plan=workflow.plan(payload.model_dump())
        if plan['model']!='resolution-flexible':raise ValueError('Field-loss preview requires the shared physical-state representation')
        from pdeflow.coordinate.contracts import CoordinateConfig
        from pdeflow.coordinate.dataset import example,state_samples,identity
        from pdeflow.field_losses import analyze,settings
        from dataclasses import replace
        import numpy as np
        c=CoordinateConfig(**plan['config']).validate();cfg=settings(c.field_loss);key=identity(c);result=[]
        if key not in preview_scales:
            pc=replace(c,encoder_points=min(32,c.encoder_points));samples={f:[[],[]] for f in c.family_weights}
            for i in range(min(12,c.train_count)):
                item=example(pc,'train',i)
                for t in (0.,.137,.4,.8):
                    values=state_samples(item,item['context'],t,pc).numpy()
                    for j in range(1+int(item['family']=='wave')):samples[item['family']][j].append(values[:,j])
            preview_scales[key]={f:[max(float(np.concatenate(a).std()),1e-3) if a else 1. for a in arrays] for f,arrays in samples.items()}
            if len(preview_scales)>16:preview_scales.pop(next(iter(preview_scales)))
        for i in range(min(12,c.train_count)):
            item=example(c,'train',i)
            for j,quantity in enumerate(('u','v')[:1+int(item['family']=='wave')]):
                scale=preview_scales[key][item['family']][j];u=item['solver'].frame(0.,item['parameter'],c.interval,(cfg.analysis_resolution,)*2,quantity);a=analyze(u,c.lengths,c.field_loss,scale);effective=a['contribution_weights'] if cfg.type=='region_balanced_mse' else a['contrast_weights']/a['contrast_weights'].mean() if cfg.type=='weighted_mse' else np.ones_like(a['reference'])
                result.append({'id':item['id'],'family':item['family']+' '+item['case']+' '+quantity,'quantity':quantity,'reference':a['reference'].tolist(),'labels':a['labels'].tolist(),'weights':effective.tolist(),'regions':a['regions'],'region_count':a['region_count'],'training_calibration_scale':scale})
        return {'examples':result,'settings':__import__('dataclasses').asdict(cfg),'split':'train','meaning':'Actual training realizations at t=0, separate signed u/v masks. Preview-only scale estimate uses up to12 training realizations and32 spatial samples at four calibration times; exact run normalization is fitted and recorded by the trainer. No test examples.'}

    @app.post('/api/training/start', status_code=202)
    def start(payload: StartRequest):
        request = payload.model_dump(exclude={'submission_token'})
        return workflow.start(request, workspace, payload.submission_token)

    @app.get('/api/training/runs')
    def runs():
        return workflow.runs(workspace)

    @app.get('/api/training/runs/{identifier}')
    def status(identifier: str):
        return workflow.status(identifier, workspace)

    @app.post('/api/training/runs/{identifier}/stop')
    def stop(identifier: str):
        return workflow.stop(identifier, workspace)

    @app.post('/api/training/runs/{identifier}/resume', status_code=202)
    def resume(identifier: str, payload: ResumeRequest):
        return workflow.resume(identifier, workspace, payload.allow_poor_representation)

    @app.post('/api/training/runs/{identifier}/extend', status_code=202)
    def extend(identifier: str, payload: Extension):
        return workflow.extend(identifier, payload.additional_updates, workspace)

    @app.get('/api/training/runs/{identifier}/preview')
    def preview(identifier: str, case: int = Query(0, ge=0, le=1), field: int = Query(0, ge=0),
                time_index: int = Query(0, ge=0), axis: int = Query(2, ge=0, le=2), index: int = Query(0, ge=0)):
        return workflow.preview(identifier, workspace, case, field, time_index, axis, index)

    @app.post('/api/training/models')
    def models(payload: ModelContext):
        return workflow.model_list(workspace, payload.problem)

    @app.post('/api/training/use')
    def use(payload: Selection):
        return workflow.select_model(payload.id, workspace, payload.problem, payload.adopt_training_problem)

    @app.get('/api/training/selection')
    def selection():
        path = workspace_root(workspace) / 'selected-model.json'
        return read_json(path) if path.exists() else None
