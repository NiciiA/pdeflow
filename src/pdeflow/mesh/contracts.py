"""Versioned periodic heat/wave mesh contracts. No legacy weight conversion."""
from dataclasses import dataclass, field, asdict, replace
import math
from pdeflow.runtime import digest

CANDIDATES={
 'mesh_m1_v1':{'name':'M1 · Deterministic initial-state mesh','stages':('reconstruction',),'generative':False,'trajectory':False},
 'mesh_m2_v1':{'name':'M2 · Deterministic mesh + evolution','stages':('reconstruction',),'generative':False,'trajectory':True},
 'mesh_m3_fgn_v1':{'name':'M3 · FGN-style mesh + evolution','stages':('central','stochastic'),'generative':True,'trajectory':True},
 'mesh_m4_flow_v1':{'name':'M4 · Conditional physical-state flow','stages':('dynamics','flow'),'generative':True,'trajectory':True},
 'mesh_m5_diffusion_v1':{'name':'M5 · Conditional physical-state diffusion','stages':('dynamics','diffusion'),'generative':True,'trajectory':True},
}
LADDER={
 'pipeline-check':(32,2,4,(4,4),(2,2),2),
 'xsmall':(96,4,4,(8,8),(4,4),4),
 'small':(192,6,6,(12,12),(4,4),6),
 'medium':(256,8,8,(16,16),(4,4),8),
 'large':(384,12,8,(24,24),(6,6),12),
 'xlarge':(512,16,8,(32,32),(8,8),16),
}
SCHEMA='mesh_physical_state_v1'

@dataclass(frozen=True)
class MeshConfig:
    architecture:str='mesh_m1_v1'
    capacity:str='pipeline-check'
    width:int=32
    depth:int=2
    heads:int=4
    node_shape:tuple=(4,4)
    coarse_shape:tuple=(2,2)
    global_tokens:int=2
    neighbors:int=4
    noise_dim:int=16
    local_noise_dim:int=4
    sensor_chunk:int=128
    query_chunk:int=128
    evolution_batch:int=8
    domain:dict=field(default_factory=lambda:{'lengths':[1.,1.],'origins':[0.,0.],'boundary':'periodic','dimension':2})
    interval:tuple=(0.,1.)
    family_weights:dict=field(default_factory=lambda:{'heat':.5,'wave':.5})
    data:dict=field(default_factory=dict)
    field_loss:dict=field(default_factory=lambda:{'type':'mse'})
    budgets:dict=field(default_factory=lambda:{'reconstruction':20,'central':20,'stochastic':20,'dynamics':20,'flow':20,'diffusion':20})
    batch_size:int=2
    microbatch:int=2
    learning_rate:float=.001
    optimizer:dict=field(default_factory=lambda:{'schedule':'constant','warmup':0,'minimum_ratio':.1})
    checkpoint_every:int=10
    validation_every:int=10
    validation_cases:int=2
    gate_ratio:float=1.
    seed:int=73
    runtime:dict=field(default_factory=lambda:{'device':'cpu','dtype':'float32','threads':2})
    precision:str='fp32'
    objective:dict=field(default_factory=dict)
    corruption:dict=field(default_factory=dict)
    sampler:dict=field(default_factory=lambda:{'steps':16,'solver':'heun','diffusion_steps':128,'eta':0.})
    warm_start:dict=field(default_factory=dict)
    freeze_dynamics:bool=True
    pipeline_check:bool=True

    @property
    def kind(self):return CANDIDATES[self.architecture]
    @property
    def nodes(self):return math.prod(self.node_shape)
    def to_dict(self):return asdict(self)
    def reference_config(self):
        from pdeflow.coordinate.contracts import CoordinateConfig
        values={'reference_shape':(16,16),'encoder_points':64,'query_points':64,'train_count':24,'validation_count':12,'test_count':12,'observation_counts':(4,24),'noise_range':(0.,.05),'unconditional_probability':0.,'distribution':{'max_wave':1,'width':[.25,.35]},'observation_time':{'policy':'continuous_uniform','stratified':True},'observation_placement':{'random':.5,'even':.25,'heat':.25},'joint':{'known_probability':.5},**self.data}
        return CoordinateConfig(**{**values,'family_weights':self.family_weights,'time_interval':self.interval,'lengths':tuple(self.domain['lengths']),'origins':tuple(self.domain['origins']),'runtime':self.runtime,'seed':self.seed,'field_loss':self.field_loss}).validate()
    def validate(self):
        if self.architecture not in CANDIDATES:raise ValueError('Unsupported mesh architecture')
        if self.domain.get('dimension')!=2 or self.domain.get('boundary')!='periodic' or set(self.domain)!={'dimension','boundary','lengths','origins'}:raise ValueError('Mesh v1 supports only periodic Cartesian2D heat/wave, not arbitrary domains/boundaries')
        for name in ('node_shape','coarse_shape','interval'):object.__setattr__(self,name,tuple(getattr(self,name)))
        if any(len(getattr(self,n))!=2 for n in ('node_shape','coarse_shape')) or any(type(n)!=int or n<2 for n in (*self.node_shape,*self.coarse_shape)):raise ValueError('Two regular node dimensions, each >=2, required')
        if any(a%b for a,b in zip(self.node_shape,self.coarse_shape)):raise ValueError('Coarse cells must partition fine periodic cells exactly')
        if self.nodes>4096 or math.prod(self.coarse_shape)>256:raise ValueError('Declared bounded node budget exceeded')
        for name in ('width','depth','heads','global_tokens','neighbors','noise_dim','local_noise_dim','sensor_chunk','query_chunk','evolution_batch','batch_size','microbatch','checkpoint_every','validation_every','validation_cases'):
            if type(getattr(self,name))!=int or getattr(self,name)<1:raise ValueError('Positive integer '+name+' required')
        if self.width%self.heads or self.neighbors>=self.nodes or self.microbatch>self.batch_size:raise ValueError('Invalid head, neighbor or microbatch sizes')
        if not math.isfinite(self.learning_rate) or self.learning_rate<=0 or not 0<self.gate_ratio<=1:raise ValueError('Invalid learning rate/gate')
        if set(self.budgets)!={'reconstruction','central','stochastic','dynamics','flow','diffusion'} or any(type(x)!=int or x<1 for x in self.budgets.values()):raise ValueError('All explicit positive stage budgets required')
        if set(self.optimizer)-{'schedule','warmup','minimum_ratio','clip','max_consecutive_overflows','initial_scale'} or self.optimizer.get('schedule','constant') not in ('constant','cosine'):raise ValueError('Invalid mesh optimizer settings')
        if self.optimizer.get('warmup',0)<0 or any(self.optimizer.get('warmup',0)>=self.budgets[s] for s in self.kind['stages']):raise ValueError('Optimizer warmup must be below every active stage budget')
        if set(self.objective)-{'initial','trajectory','parameter','anchor','central','crps','energy','observation','observation_minimum_s','observation_sensor_budget','observation_tolerance_relative','observation_discrepancy_relative','kinematic','acceleration','heat','heat_mean','heat_dissipation','physics_queries'}:raise ValueError('Unknown mesh objective setting')
        if any(not isinstance(v,(float,int)) or not math.isfinite(v) or v<0 for v in self.objective.values()):raise ValueError('Objective weights/settings must be finite and nonnegative')
        for key in ('physics_queries','observation_sensor_budget'):
            if key in self.objective and (type(self.objective[key])!=int or not 1<=self.objective[key]<=128):raise ValueError(key+' must be an integer in1..128')
        if self.objective.get('observation_minimum_s',.5)>1:raise ValueError('Observation minimum flow time must lie in[0,1]')
        if self.warm_start and (set(self.warm_start)!={'checkpoint'} or not isinstance(self.warm_start['checkpoint'],str) or self.architecture not in ('mesh_m3_fgn_v1','mesh_m4_flow_v1','mesh_m5_diffusion_v1')):raise ValueError('Warm start is an explicit M2 checkpoint for NEW M3–M5 experiments')
        if set(self.corruption)-{'missing_probability','quantity_dropout','time_dropout'} or any(not 0<=v<1 for v in self.corruption.values()):raise ValueError('Corruption probabilities must be in[0,1)')
        if self.sampler.get('solver') not in ('euler','heun') or set(self.sampler)!={'steps','solver','diffusion_steps','eta'} or not 1<=self.sampler['steps']<=512 or not 8<=self.sampler['diffusion_steps']<=4096 or self.sampler['eta']!=0:raise ValueError('Mesh v1 supports Euler/Heun FM and deterministic eta=0 DDIM; explicit bounded steps required')
        if not self.kind['trajectory'] and any(self.objective.get(k,0)>0 for k in ('heat','heat_mean','heat_dissipation','kinematic','acceleration','observation')):raise ValueError('M1 is initial-only; later-time likelihood/physics requires M2–M5')
        self.reference_config()
        return self


def preset(capacity='pipeline-check',architecture='mesh_m1_v1',device='cpu',precision='auto',overrides=None):
    if capacity=='smoke':capacity='pipeline-check'
    if capacity not in LADDER:raise ValueError('Unknown mesh capacity')
    w,d,h,n,c,g=LADDER[capacity];small=capacity=='pipeline-check'
    data={} if small else {'reference_shape':(32,32),'encoder_points':512,'query_points':256,'train_count':8192,'validation_count':256,'test_count':256,'observation_counts':(1,64),'distribution':{'max_wave':4,'width':[.1,.2]}}
    updates=2 if small else 10000 if capacity in ('xsmall','small','medium') else 100000
    cfg=MeshConfig(architecture=architecture,capacity=capacity,width=w,depth=d,heads=h,node_shape=n,coarse_shape=c,global_tokens=g,data=data,budgets={k:updates for k in MeshConfig().budgets},runtime={'device':device,'dtype':'float32','threads':2},precision=precision,checkpoint_every=1 if small else 500,validation_every=1 if small else 500,pipeline_check=small,batch_size=2 if small else 8,microbatch=2 if small else 8)
    raw=cfg.to_dict()
    for k,v in (overrides or {}).items():
        if k not in raw:raise ValueError('Unknown mesh configuration '+k)
        raw[k]={**raw[k],**v} if isinstance(v,dict) and isinstance(raw[k],dict) else v
    return MeshConfig(**raw).validate()


def model_contract(c):
    # JSON persists tuples as lists; compare the serialized scientific contract.
    import json
    return json.loads(json.dumps({k:v for k,v in c.to_dict().items() if k not in ('runtime','budgets','checkpoint_every','validation_every','validation_cases','pipeline_check')},sort_keys=True))


def capabilities(c):
    return {'architecture':c.architecture,'schema':SCHEMA,'families':c.family_weights,'state':{'heat':['u0','log_D'],'wave':['u0','v0','log_c']},'domain':c.domain,'interval':list(c.interval) if c.kind['trajectory'] else [0.,0.],'initial_only':not c.kind['trajectory'],'generative':c.kind['generative'],'observations':'Known continuous measurement timestamps across configured interval','reference_at_inference':False,'accuracy':'Unestablished; completion and smoke training are not scientific accuracy','representation':'Physical initial nodal values, not an autoencoder code'}
