"""Small heat playground facade. Full reference arrays never cross the inverse boundary."""
from pathlib import Path
from functools import lru_cache
import uuid
import numpy as np
from fastapi import Request
from pydantic import BaseModel, ConfigDict, Field
from pdeflow.core import DomainSpec, GridSpec
from pdeflow.equations import heat_problem
from pdeflow.numerics import heat_reference, smooth_initial
from pdeflow.runtime import read_json, write_json, atomic_write
from .jobs import contained


class Heat(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    preset: str = 'cosine'
    D: float = Field(default=.07, ge=.0001, le=2)
    end: float = Field(default=1., gt=0, le=5)
    frame_count: int = Field(default=31,ge=2,le=101)
    lengths: list[float] = Field(default_factory=lambda:[1.,1.], min_length=2, max_length=2)
    origins: list[float] = Field(default_factory=lambda:[0.,0.], min_length=2, max_length=2)
    amplitude: float = Field(default=.8, ge=-10, le=10)
    background: float = Field(default=.2, ge=-10, le=10)
    kx: int = Field(default=1, ge=0, le=15)
    ky: int = Field(default=1, ge=0, le=15)
    phix: float = Field(default=0, ge=-7, le=7)
    phiy: float = Field(default=0, ge=-7, le=7)
    seed: int = Field(default=73, ge=0, lt=2**32)
    expression: str = Field(default='0.2+0.8*cos(2*pi*x)*cos(2*pi*y)', max_length=2000)
    initial: list[list[float]] | None = None
    complete: str = Field(default='', max_length=2000)



class ErrorMetrics(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    reference_id: str
    time: float
    prediction: list[list[float]]
    field_loss: dict = Field(default_factory=dict)
    normalization_scale: float = Field(default=1.,gt=0)


def reference(spec: Heat, root: Path):
    from pdeflow.reference import create
    from pdeflow.reference.solvers import ReferenceProblem
    from pdeflow.reference.fields import field_array
    from pdeflow.reference.analytical import normalize_expression, verify
    if any(not .05 <= L <= 20 for L in spec.lengths):
        raise ValueError('Domain lengths must be between 0.05 and 20')
    grid=GridSpec((64,64),DomainSpec(tuple(spec.lengths),tuple(spec.origins)),tuple(np.linspace(0,spec.end,spec.frame_count)))
    p=heat_problem(grid,spec.D)
    x,y=np.meshgrid(*[c.numpy() for c in grid.coordinates()],indexing='ij')
    X=(x-spec.origins[0])/spec.lengths[0];Y=(y-spec.origins[1])/spec.lengths[1]
    exact=None
    if spec.preset=='cosine':
        mode=spec.amplitude*np.cos(2*np.pi*spec.kx*X+spec.phix)*np.cos(2*np.pi*spec.ky*Y+spec.phiy)
        initial=spec.background+mode
        decay=spec.D*((2*np.pi*spec.kx/spec.lengths[0])**2+(2*np.pi*spec.ky/spec.lengths[1])**2)
        exact=spec.background+np.exp(-decay*np.asarray(grid.times))[:,None,None]*mode
    elif spec.preset in ('spot','two'):
        def spot(a,b):return np.exp(-(np.sin(np.pi*(X-a))**2+np.sin(np.pi*(Y-b))**2)/.065)
        initial=spec.background+spec.amplitude*(spot(.35,.5)+(spot(.7,.65) if spec.preset=='two' else 0))
    elif spec.preset=='random':initial=spec.background+spec.amplitude*smooth_initial(np.random.default_rng(spec.seed),grid)
    elif spec.preset=='draw':initial=field_array(spec.initial if spec.initial is not None else 0.,grid)
    elif spec.preset=='formula':
        import ast
        from pdeflow.reference.fields import FUNCTIONS
        expression=normalize_expression(spec.expression)
        try:tree=ast.parse(expression,mode='eval')
        except SyntaxError as error:raise ValueError('This formula is incomplete; check parentheses and operators') from error
        unknown={n.id for n in ast.walk(tree) if isinstance(n,ast.Name)}-{'x','y','pi','D','t',*FUNCTIONS}
        if unknown:raise ValueError('This formula contains an unknown symbol: '+', '.join(sorted(unknown)))
        initial=field_array({'expression':expression},grid,{'D':spec.D,'t':0.})
    else:raise ValueError('Choose a supported starting field')
    if np.max(np.abs(initial))>10000:raise ValueError('Starting field magnitude must be at most 10000')
    identifier=uuid.uuid4().hex;directory=contained(root,identifier)
    from pdeflow.reference.examples import from_problem
    from dataclasses import replace
    manifest=create(replace(from_problem(p,spec.seed),inputs={'initial_u':initial.tolist()},solver='numerical'),directory,identifier)
    write_json(directory/'heat-config.json',spec.model_dump())
    with np.load(directory/'reference.npz',allow_pickle=False) as saved:frames=saved['values'][0,0]
    verification=None
    if spec.complete:
        _,verification=verify(p,initial,spec.complete)
    # UI receives cached frames once, never a network request per animation frame.
    return {'id':identifier,'problem':p.to_dict(),'frames':frames.tolist(),'initial':initial.tolist(),
            'exact':exact.tolist() if exact is not None else None,'exact_max_error':float(np.max(np.abs(frames-exact))) if exact is not None else None,
            'range':[float(frames.min()),float(frames.max())],'diagnostics':manifest['diagnostics'],'verification':verification}


@lru_cache(maxsize=8)
def _reference(root,identifier):
    from pdeflow.reference import load
    from pdeflow.heat_continuous import FourierHeat
    p,v,_=load(contained(Path(root),identifier))
    if p.family!='heat' or p.grid.dimension!=2 or p.grid.times[0]!=0 or 'diffusivity' not in p.known_parameters:raise ValueError('Choose an unforced periodic heat2D reference starting at physical time0')
    return p,FourierHeat(v[0,0,0],p.grid.domain.lengths,p.grid.domain.origins)


@lru_cache(maxsize=64)
def _analysis(root,identifier,time_value):
    p,solver=_reference(root,identifier)
    return solver.frame(time_value,p.known_parameters['diffusivity'],(0.,p.grid.times[-1]),shape=(32,32))


@lru_cache(maxsize=64)
def _features(root,identifier,time_value):
    from pdeflow.heat_placement import features
    p,_=_reference(root,identifier)
    return features(_analysis(root,identifier,time_value),p.grid.domain.lengths,p.grid.domain.origins)[0]


class Measurements(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    reference_id: str
    count: int = Field(default=50,ge=1,le=2000)
    noise: float = Field(default=0,ge=0,le=100)
    seed: int = Field(default=91,ge=0,lt=2**32)
    time: float
    pattern: str = 'current'
    placement: str = 'random'
    background: float = Field(default=.25,ge=.1,le=1)
    spread: float = Field(default=.08,ge=.04,le=.4)
    minimum_distance: float = Field(default=.005,ge=0,le=.1)
    times: list[float] = Field(default_factory=list,max_length=100)
    adaptive: bool = False
    positions: list[list[float]] = Field(default_factory=list,max_length=2000)


def candidates(spec: Measurements,root:Path):
    from pdeflow.heat_continuous import bounded_time
    from pdeflow.heat_placement import place
    p,solver=_reference(str(root),spec.reference_id);g=p.grid;interval=(0.,g.times[-1])
    t=float(bounded_time(spec.time,interval))
    if spec.pattern not in ('current','across','fixed','manual'):raise ValueError('Unknown time sampling pattern')
    selected=bounded_time(spec.times,interval) if spec.times else np.array([0.,interval[1]/2,interval[1]])
    if len(set(selected))!=len(selected):raise ValueError('Choose distinct physical measurement times')
    policy='manual' if spec.pattern=='manual' else spec.placement
    count=spec.count
    times=np.full(count,t)
    if spec.pattern=='across':
        times=np.random.default_rng(spec.seed+271).uniform(*interval,count) if not spec.times else np.resize(selected,count)
    if spec.pattern=='fixed':times=np.resize(selected,count)
    sensors=int(np.ceil(count/len(selected))) if spec.pattern=='fixed' else count
    field=_analysis(str(root),spec.reference_id,t) if policy=='heat' else None
    xy,metadata=place(sensors,policy,spec.seed,g.domain.lengths,g.domain.origins,field=field,centres=_features(str(root),spec.reference_id,t) if policy=='heat' else None,background=spec.background,spread=spec.spread,minimum_distance=spec.minimum_distance,manual=spec.positions)
    if spec.pattern=='fixed':xy=np.repeat(xy,len(selected),axis=0)[:count]
    if spec.adaptive and spec.pattern=='fixed':raise ValueError('Fixed sensors cannot adapt their positions over time')
    if spec.adaptive and spec.pattern=='across':
        if not spec.times:raise ValueError('Choose explicit times for time-adaptive placement (bounded field analysis)')
        designs=[]
        for j,at in enumerate(selected):
            ix=np.flatnonzero(times==at)
            if len(ix):
                positions,report=place(len(ix),policy,spec.seed+j,g.domain.lengths,g.domain.origins,field=_analysis(str(root),spec.reference_id,float(at)) if policy=='heat' else None,centres=_features(str(root),spec.reference_id,float(at)) if policy=='heat' else None,background=spec.background,spread=spec.spread,minimum_distance=spec.minimum_distance,manual=spec.positions)
                xy[ix]=positions;designs.append({'time':float(at),'placement':report})
        metadata.update(time_designs=designs,retained_regions=[],allocation=[])
    metadata.update(design_time=t,reference_id=spec.reference_id,time_policy=spec.pattern,time_adaptive=spec.adaptive,
      spatial_sensors=len(set(map(tuple,xy))),sampled_times=len(set(times)),total_scalar_observations=count)
    return {'positions':xy.tolist(),'times':times.tolist(),'placement':metadata}


def measurements(spec: Measurements,root:Path):
    design=candidates(spec,root);p,solver=_reference(str(root),spec.reference_id)
    xy=np.asarray(design['positions']);times=np.asarray(design['times']);points=np.c_[xy,times]
    if len(set(map(tuple,points)))!=spec.count:raise ValueError('Measurements need unique space-time points')
    clean=solver.evaluate(points,p.known_parameters['diffusivity'],(0.,p.grid.times[-1]))
    noise=np.random.default_rng(np.random.SeedSequence([spec.seed,983]));noisy=clean+noise.normal(size=spec.count)*spec.noise
    rows=[{'field':'u','x':float(x),'y':float(y),'time':float(t),'value':float(v),'sigma':spec.noise} for (x,y),t,v in zip(xy,times,noisy)]
    identifier=uuid.uuid4().hex;directory=contained(root,spec.reference_id)/'samplings'/identifier;directory.mkdir(parents=True)
    record={'id':identifier,'count':spec.count,'rows':rows,'seed':spec.seed,'noise_std':spec.noise,'reference_id':spec.reference_id,'pattern':spec.pattern,'placement':design['placement'],
            'definition':'N total scalar observations at actual floating physical times; finite Fourier reference, independent absolute Gaussian noise; selection separate from values'}
    write_json(directory/'rows.json',rows);write_json(directory/'sampling.json',record)
    atomic_write(directory/'clean-evaluation-only.npz',lambda f:np.savez_compressed(f,clean=clean))
    return record


def register(app,workspace,data,output):
    from pdeflow.local import project_root
    managed=project_root()/'outputs'
    managed=managed if workspace.is_relative_to(managed) else workspace

    @app.get('/api/playground/storage')
    def storage():
        from pdeflow.model_storage import inventory
        return inventory(managed)

    def idle():
        if any(j['status'] in ('queued','running') for j in app.state.jobs.list()):
            raise ValueError('A reconstruction is active; wait or cancel it before deleting models')

    @app.post('/api/playground/storage/preview')
    def deletion_preview(payload:dict):
        from pdeflow.model_storage import preview
        with app.state.jobs.lock:
            idle()
            return preview(managed,**payload)

    @app.post('/api/playground/storage/delete')
    def deletion_apply(payload:dict):
        from pdeflow.model_storage import apply
        with app.state.jobs.lock:
            idle()
            return apply(managed,**payload)

    @app.post('/api/playground/reference')
    def make_reference(payload:Heat):return reference(payload,data)

    @app.post('/api/playground/candidates')
    def preview_candidates(payload:Measurements):return candidates(payload,data)

    @app.get('/api/playground/reference/{identifier}/at')
    def reference_at(identifier:str,time:float):
        p,solver=_reference(str(data),identifier)
        from pdeflow.heat_continuous import bounded_time
        time=float(bounded_time(time,(0.,p.grid.times[-1])))
        spec_path=contained(data,identifier)/'heat-config.json'
        spec=read_json(spec_path) if spec_path.exists() else None
        frame=solver.frame(time,p.known_parameters['diffusivity'],(0.,p.grid.times[-1]))
        exact=(spec['background']+(solver.initial-spec['background'])*np.exp(-spec['D']*((2*np.pi*spec['kx']/spec['lengths'][0])**2+(2*np.pi*spec['ky']/spec['lengths'][1])**2)*time)).tolist() if spec and spec['preset']=='cosine' else None
        return {'time':time,'frame':frame.tolist(),'exact':exact,'policy':'Finite Fourier evolution at exact requested physical time; no saved-frame interpolation'}

    @app.get('/api/playground/results/{identifier}/at')
    def result_at(identifier:str,time:float):
        from pdeflow.coordinate.inference import decode_saved
        from pdeflow.coordinate.registry import model
        directory=contained(output,identifier)
        if app.state.jobs.get(identifier)['status']!='succeeded':raise ValueError('Wait for completed reconstruction')
        request=read_json(directory/'request.json');entry=model(workspace,request['model_id'])
        return decode_saved(directory/'coordinate-result',entry['export'],time)

    @app.post('/api/playground/error-metrics')
    def error_metrics(payload:ErrorMetrics):
        from pdeflow.field_losses import field_metrics
        p,solver=_reference(str(data),payload.reference_id)
        prediction=np.asarray(payload.prediction,dtype=float)
        if prediction.ndim!=2 or min(prediction.shape)<2 or max(prediction.shape)>128:raise ValueError('Error metrics require a bounded2D query grid up to128²')
        reference=solver.frame(payload.time,p.known_parameters['diffusivity'],(0.,p.grid.times[-1]),shape=prediction.shape)
        result=field_metrics(prediction,reference,p.grid.domain.lengths,payload.field_loss,payload.normalization_scale)
        a=result.pop('analysis');labels=result['labels'];sq=(prediction-reference)**2
        weights=np.ones_like(sq)
        if result['region_count'] and not result['unrepresented_region_count']:
            alpha=result['settings']['alpha'];weights.fill(1-alpha)
            for region in result['regions']:
                mask=labels==region['label'];weights[mask]+=alpha/(result['region_count']*mask.mean())
        contrast=__import__('pdeflow.field_losses',fromlist=['project']).project(a['contrast_weights'],prediction.shape)
        result.update(labels=labels.tolist(),region_contribution=(sq*weights).tolist() if result['region_balanced_mse'] is not None else None,signed_error=(prediction-reference).tolist(),squared_error=sq.tolist(),weighted_contribution=(sq*contrast/contrast.mean()).tolist(),time=payload.time,reference_id=payload.reference_id)
        return result

    @app.post('/api/playground/measurements')
    def sample(payload:Measurements):return measurements(payload,data)

    @app.post('/api/playground/models')
    def models(payload:dict):
        from pdeflow.coordinate.registry import runs
        from pdeflow.coordinate.contracts import CoordinateConfig
        from pdeflow.core import ProblemSpec
        p=ProblemSpec.from_dict(payload['problem']);result=[]
        for row in runs(workspace):
            if row.get('stage')!='flow' or not row.get('available') or row['config']['dimension']!=2:continue
            c=CoordinateConfig(**row['config']).validate();reason='';compatible=True
            try:c.compatible(p)
            except ValueError as e:compatible=False;reason=str(e)
            if not c.diffusivity_range[0]<=p.known_parameters['diffusivity']<=c.diffusivity_range[1]:reason=f'This model supports D between {c.diffusivity_range[0]} and {c.diffusivity_range[1]}.'
            if c.temporal_mode!='continuous':compatible=False;reason='Legacy snapshot checkpoint is not continuously trained. Use Legacy tools within its original interval, or train/import a continuous model.'
            partial=row.get('state')=='partial'
            result.append({'id':row['model_id'],'name':f'Heat model · {row["progress"]} flow updates'+(' · partial training' if partial else ''),'compatible':compatible,'reason':reason,
                           'pipeline_check':partial or max(c.ae_steps,row['progress'])<=80,'temporal':__import__('pdeflow.coordinate.temporal',fromlist=['capability']).capability(c),'status':'Scientifically unevaluated','range':list(c.diffusivity_range)})
        return result

    @app.post('/api/playground/import')
    async def upload(request:Request):
        from pdeflow.coordinate.training import read_checkpoint
        from pdeflow.coordinate.registry import register_checkpoint
        directory=workspace/'model-imports'/uuid.uuid4().hex;directory.mkdir(parents=True)
        path=directory/'coordinate.pt';temporary=directory/'upload.part';length=0
        try:
            with temporary.open('wb') as stream:
                async for chunk in request.stream():
                    length+=len(chunk)
                    if length>2*1024**3:raise ValueError('Model exceeds the 2 GiB local import limit')
                    stream.write(chunk)
            if not length:raise ValueError('Choose a coordinate flow export')
            temporary.rename(path)
            cp,c=read_checkpoint(path,'flow')
            if c.dimension!=2 or not (cp['completed'] or cp.get('inference_ready')):raise ValueError('Choose a completed or explicitly exported partial resolution-flexible 2D heat model')
            row=register_checkpoint(path,workspace)
            write_json(directory/'model-import.json',{'kind':'playground_model_import','export':str(path),'model_id':row['id']})
            return {'id':row['id']}
        except Exception:
            path.unlink(missing_ok=True)
            temporary.unlink(missing_ok=True)
            raise

    @app.get('/api/playground/results/{identifier}')
    def results(identifier:str):
        from pdeflow.core import ProblemSpec
        directory=contained(output,identifier)/'coordinate-result'
        if app.state.jobs.get(identifier)['status']!='succeeded':raise ValueError('Reconstruction has not finished')
        p=ProblemSpec.from_dict(read_json(directory/'problem.json'))
        with np.load(directory/'result.npz',allow_pickle=False) as f:samples=f['samples'].reshape(-1,len(p.grid.times),*p.grid.shape)
        metrics=read_json(directory/'metrics.json')
        return {'frames':samples.mean(0).tolist(),'spread':samples.std(0).tolist(),'problem':p.to_dict(),'metrics':metrics,
                'comparison':read_json(directory/'comparison.json') if (directory/'comparison.json').exists() else None}
