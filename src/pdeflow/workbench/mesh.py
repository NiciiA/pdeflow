"""Mesh inference routes validate capabilities independently of the browser."""
from pathlib import Path
from fastapi import Request
from pydantic import Field
from pdeflow.runtime import RuntimeSpec,read_json
from pdeflow.core import ProblemSpec
from pdeflow.coordinate.contracts import validate_rows,validate_queries,grid_queries
from .coordinate import Inverse

class MeshInverse(Inverse):
    solver:str|None=Field(default=None,pattern=r'^(euler|heun|ddim)$')


def register(app,workspace,data,output):
    from pdeflow.mesh.registry import model
    from pdeflow.mesh.provenance import read_checkpoint
    @app.post('/api/mesh/validate')
    def validate(payload:MeshInverse):
        entry=model(workspace,payload.model_id);cp,c=read_checkpoint(entry['export']);p=ProblemSpec.from_dict(payload.problem);reference=c.reference_config();reference.compatible(p);validate_rows(reference,payload.rows,p)
        if payload.mathematical_request is not None:
            from pdeflow.symbolic import analyze
            parsed=analyze(payload.mathematical_request,observation_kinds=('coordinate_point_set',))
            if parsed['problem'] is None:raise ValueError('Current mathematical source is unsupported: '+'; '.join(parsed['reasons']))
            source=ProblemSpec.from_dict(parsed['problem'])
            if source.grid.domain!=p.grid.domain or source.grid.dimension!=p.grid.dimension or source.grid.convention!=p.grid.convention or source.grid.temporal_kind!=p.grid.temporal_kind or {**source.to_dict(),'grid':p.to_dict()['grid']}!=p.to_dict():raise ValueError('Current mathematical source differs from the executable problem; reparse current draft')
            if payload.mathematical_request['measurements']['rows']!=payload.rows:raise ValueError('Source measurement rows differ from submitted observations')
        if payload.correction is not None:raise ValueError('Correction is unsupported for mesh models; no silently changed sampling law')
        if not payload.rows and reference.observation_counts[0]>0 and reference.unconditional_probability==0:raise ValueError('This model was not trained for zero observations')
        if c.architecture=='mesh_m4_flow_v1' and payload.solver not in (None,'euler','heun'):raise ValueError('Flow requires Euler or Heun')
        if c.architecture=='mesh_m5_diffusion_v1' and payload.solver not in (None,'ddim'):raise ValueError('Diffusion uses deterministic DDIM, not a flow solver')
        if c.architecture=='mesh_m5_diffusion_v1' and payload.steps>c.sampler['diffusion_steps']:raise ValueError('Reverse steps exceed the recorded diffusion schedule')
        q=validate_queries(reference,payload.queries if payload.queries is not None else grid_queries(p.grid) if c.kind['trajectory'] else grid_queries(__import__('dataclasses').replace(p.grid,times=(0.,))))
        if not c.kind['trajectory'] and q[:,-1].ne(0).any():raise ValueError('M1 only reconstructs t=0; later sensor timestamps remain valid conditioning')
        if len(q)*payload.particles>16_777_216:raise ValueError('Decoded output exceeds64MiB budget')
        RuntimeSpec(device=payload.device).resolve()
        return {'valid':True,'architecture':c.architecture,'initial_only':not c.kind['trajectory'],'query_count':len(q),'measurement_time_support':c.interval,'no_hidden_reference':True,'warning':'Trained budget does not establish accuracy; internal node count limits represented detail'}
    @app.post('/api/mesh/reconstruct',status_code=202)
    def reconstruct(payload:MeshInverse):
        validate(payload);entry=model(workspace,payload.model_id);request={'kind':'mesh','model_id':payload.model_id,'problem':payload.problem,'rows':payload.rows,'queries':payload.queries,'inference':{'steps':payload.steps,'particles':payload.particles,'seed':payload.seed,'chunk_size':payload.chunk_size,'solver':payload.solver,'correction':payload.correction},'runtime':{'device':payload.device},'mode':'mesh_observation_conditioning'}
        job=app.state.jobs.submit(request,entry['export'],payload.submission_token)
        if payload.mathematical_request is not None:
            from pdeflow.runtime import write_json
            write_json(output/job['id']/'mathematical-source.json',payload.mathematical_request)
        return job
    @app.post('/api/mesh/preflight')
    def preflight(payload:dict):
        from pdeflow.mesh.workflow import plan
        from pdeflow.mesh.contracts import MeshConfig
        from pdeflow.mesh.benchmark import profile
        config=MeshConfig(**plan(payload)['config']).validate();root=workspace/'mesh-profiles';root.mkdir(exist_ok=True);result=profile(config,root,updates=8,warmup=2)
        from pdeflow.runtime import write_json,digest
        write_json(root/(digest(config.to_dict())+'.json'),result);return result
    @app.post('/api/mesh/microbatch-benchmark')
    def microbatch_benchmark(payload:dict):
        from pdeflow.mesh.workflow import plan
        from pdeflow.mesh.contracts import MeshConfig
        from pdeflow.mesh.benchmark import microbatch_benchmark
        from pdeflow.local import is_locked
        if any(is_locked(p) for p in (workspace/'training-workflows').glob('*')):
            raise ValueError('Stop active training before benchmarking on this device')
        config=MeshConfig(**plan(payload['request'])['config']).validate()
        root=workspace/'mesh-profiles';root.mkdir(exist_ok=True)
        return microbatch_benchmark(config,root,payload.get('candidates',[1,2,4,8]),payload.get('updates',8),payload.get('warmup',2))
