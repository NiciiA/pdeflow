"""Thin UI adapter to the same persistent training workflow used by the CLI."""
from fastapi import Query
from pydantic import BaseModel, ConfigDict, Field
from pdeflow import training_workflow as workflow
from pdeflow.runtime import read_json
from pdeflow.local import workspace_root


class TrainingRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    problem: str = 'heat-2d'
    model: str = 'fixed-grid'
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
        if plan['model']!='resolution-flexible' or plan['problem']!='heat-2d':raise ValueError('Field-loss preview requires the heat2D representation')
        from pdeflow.coordinate.contracts import CoordinateConfig
        from pdeflow.coordinate.dataset import example
        from pdeflow.field_losses import analyze,settings
        c=CoordinateConfig(**plan['config']).validate();cfg=settings(c.field_loss);result=[]
        if c.data_mode!='procedural':raise ValueError('Live loss preview requires the procedural training distribution; saved materialized runs retain their original data')
        from pdeflow.coordinate.optimization import options
        from pdeflow.coordinate.dataset import identity
        from dataclasses import replace
        import numpy as np
        count=min(c.train_count,options(c)['calibration_count']);key=identity(c)
        if count>512:raise ValueError('Live preview is bounded to512 calibration solutions; inspect saved training calibration for a larger budget')
        if key not in preview_scales:
            # Encoder sample count changes no physical realization or dataset identity.
            pc=replace(c,encoder_points=8);total=total2=points=0
            for i in range(count):
                u=example(pc,'train',i)['values'].astype(np.float64);total+=u.sum();total2+=(u*u).sum();points+=u.size
            mean=total/points;preview_scales[key]=max(float(np.sqrt(max(0,total2/points-mean*mean))),1e-6)
            if len(preview_scales)>16:preview_scales.pop(next(iter(preview_scales)))
        scale=preview_scales[key]
        for i in range(min(6,c.train_count)):
            item=example(c,'train',i)
            u=item['solver'].frame(0.,item['D'],c.interval,shape=(cfg.analysis_resolution,)*2) if 'solver' in item else item['values'].reshape(len(c.times),*c.reference_shape)[0]
            a=analyze(u,c.lengths,c.field_loss,scale)
            effective=a['contribution_weights'] if cfg.type=='region_balanced_mse' else a['contrast_weights']/a['contrast_weights'].mean() if cfg.type=='weighted_mse' else np.ones_like(a['reference'])
            result.append({'id':item['id'],'family':item['family'],'reference':a['reference'].tolist(),'labels':a['labels'].tolist(),'weights':effective.tolist(),'regions':a['regions'],'region_count':a['region_count'],'training_calibration_scale':scale})
        return {'examples':result,'settings':__import__('dataclasses').asdict(cfg),'split':'train','meaning':'Actual training distribution at t=0; exact frozen training-calibration scale from the same physical realizations/recipe. Never test examples.'}

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
