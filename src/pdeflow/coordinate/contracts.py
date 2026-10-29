"""Heat coordinate specialist contracts; no reference in the inverse schema."""
from dataclasses import asdict, dataclass, field
import math
import numpy as np
import torch
from pdeflow.core import GridSpec, ProblemSpec

@dataclass(frozen=True)
class CoordinateConfig:
    dimension: int = 2
    reference_shape: tuple[int,...] = (8,8)
    times: tuple[float,...] = (0., .05, .1)
    lengths: tuple[float,...] = (1.,1.)
    origins: tuple[float,...] = (0.,0.)
    diffusivity_range: tuple[float,float] = (.03,.12)
    width: int = 32
    tokens: int = 8
    layers: int = 2
    heads: int = 4
    frequencies: int = 3
    encoder_points: int = 256
    query_points: int = 256
    train_count: int = 12
    validation_count: int = 4
    test_count: int = 4
    ae_steps: int = 20
    flow_steps: int = 20
    batch_size: int = 2
    learning_rate: float = .001
    checkpoint_every: int = 10
    seed: int = 73
    observation_counts: tuple[int,int] = (0,24)
    noise_range: tuple[float,float] = (0.,.05)
    unconditional_probability: float = .1
    runtime: dict = field(default_factory=lambda: {'device':'cpu','dtype':'float32','threads':2})
    temporal_mode: str = 'snapshots'
    time_interval: tuple[float,float] = (0.,1.)
    time_sampling: dict = field(default_factory=dict)
    observation_placement: dict = field(default_factory=dict)
    animation_frames: int = 31
    data_mode: str = 'materialized'
    distribution: dict = field(default_factory=dict)
    training_options: dict = field(default_factory=dict)
    field_loss: dict = field(default_factory=dict)

    @property
    def interval(self):return tuple(self.time_interval) if self.temporal_mode=='continuous' else (self.times[0],self.times[-1])

    def validate(self):
        for name in ('reference_shape','times','lengths','origins','diffusivity_range','observation_counts','noise_range','time_interval'):
            object.__setattr__(self,name,tuple(getattr(self,name)))
        if self.dimension not in (1,2,3) or len(self.reference_shape)!=self.dimension or len(self.lengths)!=self.dimension or len(self.origins)!=self.dimension:raise ValueError('Coordinate heat requires explicit matching 1D–3D coordinates/domain')
        if self.temporal_mode not in ('snapshots','continuous'):raise ValueError('Temporal mode must be snapshots or continuous')
        if type(self.animation_frames)!=int or not 2<=self.animation_frames<=101:raise ValueError('Animation frame count must be 2..101; this is not training-time support')
        if self.temporal_mode=='continuous':
            if self.data_mode!='procedural' or self.dimension!=2:raise ValueError('Continuous training requires procedural heat2D; migrate to a new run, do not resume a snapshot model')
            if len(self.time_interval)!=2 or not np.isfinite(self.time_interval).all() or self.time_interval[0]!=0 or not 0<self.time_interval[1]<=5:raise ValueError('Continuous physical interval must start at0 and end in (0,5]')
            from .temporal import policies
            policies(self)
            placement=self.observation_placement or {'random':.5,'even':.25,'heat':.25}
            if set(placement)-{'random','even','heat'} or not placement or not all(np.isfinite(v) and v>=0 for v in placement.values()) or not np.isclose(sum(placement.values()),1):raise ValueError('Observation-placement mixture must sum to one over random/even/heat')
        from pdeflow.core import DomainSpec
        GridSpec(self.reference_shape,DomainSpec(self.lengths,self.origins),self.times)
        if len(self.times)<2 or self.times[0]!=0:raise ValueError('Transient physical times must start at0 and include an interval')
        integers=('width','tokens','layers','heads','frequencies','encoder_points','query_points','train_count','validation_count','test_count','ae_steps','flow_steps','batch_size','checkpoint_every')
        if any(type(getattr(self,k))!=int or getattr(self,k)<1 for k in integers) or self.width%self.heads or self.width>2048 or self.tokens>512 or self.layers>24:raise ValueError('Invalid bounded architecture/training sizes; width must divide heads (limits: width2048, tokens512, layers24)')
        if not all(math.isfinite(x) for x in (*self.diffusivity_range,*self.noise_range,self.learning_rate,self.unconditional_probability)) or not 0<self.diffusivity_range[0]<=self.diffusivity_range[1] or not 0<=self.noise_range[0]<=self.noise_range[1] or self.learning_rate<=0 or not 0<=self.unconditional_probability<1:raise ValueError('Invalid parameter/noise/dropout ranges')
        if len(self.observation_counts)!=2 or any(type(v)!=int for v in self.observation_counts) or not 0<=self.observation_counts[0]<=self.observation_counts[1]<=20000 or self.observation_counts[1]<1:raise ValueError('Observation-count distribution must be bounded0..20000')
        if self.temporal_mode!='continuous' and self.observation_counts[1]>math.prod(self.reference_shape)*len(self.times):raise ValueError('Training observation count exceeds unique reference field-space-time points; reduce the configured count distribution')
        if self.flow_steps<2:raise ValueError('Flow requires at least2 optimizer updates; zero-observation training is enabled only by its explicit distribution settings')
        if math.prod(self.reference_shape)*len(self.times)>2_000_000:raise ValueError('Reference field-time budget exceeded')
        if self.data_mode not in ('materialized','procedural'):raise ValueError('Data mode must be materialized or procedural')
        from pdeflow.field_losses import settings
        loss=settings(self.field_loss)
        if loss.estimator!='legacy_uniform_points' and self.dimension!=2:raise ValueError('Grouped field reconstruction losses currently require scalar periodic heat2D')
        if loss.estimator!='legacy_uniform_points' and self.query_points < (loss.time_groups if self.temporal_mode=='continuous' else len(self.times)):raise ValueError('Query budget must cover all field-loss physical time groups')
        from .optimization import options
        options(self)
        if self.data_mode=='procedural':
            from .dataset import recipe
            recipe(self)
            if self.temporal_mode=='continuous':
                if min(self.encoder_points,self.query_points)<8:raise ValueError('Continuous encoder/query budgets need at least8 scalar space-time points')
                return self
            if min(self.encoder_points,self.query_points)<len(self.times):raise ValueError('Encoder/query points must cover every saved physical time')
            if max(self.encoder_points,self.query_points)>math.prod(self.reference_shape)*len(self.times):raise ValueError('Encoder/query point counts exceed unique saved space-time nodes; choose an explicit smaller count')
        return self

    def to_dict(self):return asdict(self)

    def compatible(self,problem: ProblemSpec):
        from pdeflow.specialists import get_specialist
        from pdeflow.equations import heat_problem
        expected=heat_problem(problem.grid,.07)
        if problem.operator_spec!=expected.operator_spec or problem.fields!=expected.fields or problem.known_fields or problem.unknown_parameters or set(problem.known_parameters)!={'diffusivity'} or problem.initial_condition!=expected.initial_condition:raise ValueError('Coordinate specialist supports only the declared unknown scalar heat field and known constant diffusivity, with no forcing/known initial field')
        if problem.family!='heat' or problem.grid.dimension!=self.dimension or tuple(problem.grid.domain.lengths)!=self.lengths or tuple(problem.grid.domain.origins)!=self.origins or problem.grid.convention!='periodic_nodes_endpoint_excluded':raise ValueError('Coordinate heat checkpoint requires its exact family, field schema, dimension, domain, periodic boundary and node convention; only query grid/time counts may vary')
        if problem.grid.times[0]<self.interval[0]-1e-7 or problem.grid.times[-1]>self.interval[1]+1e-7 or problem.grid.temporal_kind!='transient':raise ValueError('Physical-time query outside checkpoint support '+str(self.interval)+('. Legacy snapshot checkpoint is not continuously trained over [0,1]; train/import a continuous model.' if self.temporal_mode!='continuous' else '; no extrapolation'))
        if not self.diffusivity_range[0]<=problem.known_parameters['diffusivity']<=self.diffusivity_range[1]:raise ValueError('Known diffusivity outside checkpoint training range')


def grid_queries(grid: GridSpec):
    """[T,*space,d+1], flattened physical (x,tau) in C order."""
    mesh=torch.meshgrid(torch.tensor(grid.times,dtype=torch.float32),*[x.float() for x in grid.coordinates()],indexing='ij')
    return torch.stack([*mesh[1:],mesh[0]],-1).reshape(-1,grid.dimension+1)


def validate_queries(config,queries):
    q=torch.as_tensor(queries,dtype=torch.float64)
    if q.ndim!=2 or q.shape[1]!=config.dimension+1 or not 1<=len(q)<=2_000_000 or not torch.isfinite(q).all():raise ValueError('Queries require finite [count,spatial_dimension+physical_time] coordinates')
    lo=q.new_tensor([*config.origins,config.interval[0]]);hi=q.new_tensor([*(o+l for o,l in zip(config.origins,config.lengths)),config.interval[-1]])
    from pdeflow.heat_continuous import bounded_time
    bounded_time(q[:,-1].detach().cpu().double().numpy(),config.interval)
    if torch.any(q[:,:-1]<lo[:-1]) or torch.any(q[:,:-1]>hi[:-1]):raise ValueError('Query outside physical domain; no extrapolation')
    q=q.clone();q[:,-1]=q[:,-1].clamp(*config.interval)
    return q.float()


def validate_rows(config,rows,problem=None):
    if len(rows)>20000:raise ValueError('At most20000 scalar observations')
    coordinates=[];values=[];sigma=[];keys=set()
    names=('x','y','z')[:config.dimension]
    for row in rows:
        if set(row)-{'field','time','value','sigma',*names} or not {'field','time','value',*names}<=set(row) or row['field']!='u':raise ValueError('Each coordinate observation needs field=u, physical coordinates, time, value, optional absolute sigma')
        try:point=[float(row[n]) for n in names]+[float(row['time'])];v=float(row['value']);s=float(row.get('sigma',0.))
        except (ValueError,TypeError,OverflowError) as error:raise ValueError('Observation coordinates, physical time, value and sigma must be numbers') from error
        if not math.isfinite(v) or not math.isfinite(s) or s<0:raise ValueError('Observed value and absolute sigma must be finite; sigma>=0')
        if any(point[j]<config.origins[j] or point[j]>config.origins[j]+config.lengths[j] for j in range(config.dimension)):raise ValueError('Observation outside physical domain; no extrapolation')
        if any(point[j]>=config.origins[j]+config.lengths[j] for j in range(config.dimension)):raise ValueError('Periodic observation coordinates exclude the duplicated upper endpoint; use its origin instead')
        key=tuple(point)
        if key in keys:raise ValueError('Duplicate scalar observation coordinates/time')
        keys.add(key);coordinates.append(point);values.append(v);sigma.append(s)
    q=validate_queries(config,coordinates) if rows else torch.empty(0,config.dimension+1)
    if problem is not None:config.compatible(problem)
    y=torch.tensor(values);noise=torch.tensor(sigma)
    if not torch.isfinite(y).all() or not torch.isfinite(noise).all():raise ValueError('Observation values/sigma must be representable in checkpoint float32 precision')
    return q,y,noise


def padded_observations(items,normalization,device):
    """Masked unordered scalar set; index0 is a learned null/known-physics token."""
    size=max(1,max(len(i[1]) for i in items));d=items[0][0].shape[1]
    points=torch.zeros(len(items),size,d,device=device);features=torch.zeros(len(items),size,3,device=device);mask=torch.zeros(len(items),size,dtype=torch.bool,device=device)
    for b,(q,y,s) in enumerate(items):
        n=len(y)
        if n:
            points[b,:n]=q.to(device);features[b,:n,0]=(y.to(device)-normalization['mean'])/normalization['scale'];features[b,:n,1]=s.to(device)/normalization['scale'];features[b,:n,2]=1.;mask[b,:n]=True
    return points,features,mask
