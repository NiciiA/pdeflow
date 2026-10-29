"""Fixed-kernel Feynman–Kac SMC with explicit finite-step terminal target."""
from dataclasses import dataclass,field,asdict
import base64
import hashlib
import math
import time
import torch


@dataclass(frozen=True)
class FKConfig:
    dynamics: str = 'linear_cfm_score_sde'
    stepper: str = 'euler_maruyama'
    noise_schedule: str = 'linear_vanish'
    kappa: float = .2
    potential: str = 'annealed_difference'
    beta_start: float = 0.
    resampler: str = 'systematic'
    resample_every: int = 2
    ess_fraction: float = .5
    terminal_resample: bool = False
    chunk_size: int = 16
    observation_weight: float = .01
    physics_weight: float = 0.
    observation_convention: str = 'gaussian_sum'
    physics_scales: dict[str,float] = field(default_factory=dict)
    max_particle_evaluations: int = 131072
    diagnostic_every: int = 1
    record_particle_history: bool = True

    def validate(self):
        if (self.dynamics,self.stepper,self.noise_schedule,self.potential,self.resampler)!=('linear_cfm_score_sde','euler_maruyama','linear_vanish','annealed_difference','systematic'):raise ValueError('Unsupported FK dynamics/stepper/schedule/potential/resampler')
        if not math.isfinite(self.kappa) or not 0<=self.kappa<=2 or not 0<=self.beta_start<1:raise ValueError('FK kappa must be in [0,2], beta_start in [0,1)')
        if not 0<self.ess_fraction<=1:raise ValueError('ESS fraction must be in (0,1]')
        for n in ['resample_every','chunk_size','max_particle_evaluations','diagnostic_every']:
            if type(getattr(self,n)) is not int or getattr(self,n)<1:raise ValueError(f'{n} must be positive integer')
        if self.chunk_size>256 or self.max_particle_evaluations>1048576:raise ValueError('FK exceeds bounded local chunk/evaluation limits')
        if type(self.terminal_resample) is not bool:raise ValueError('terminal_resample must be boolean')
        if self.observation_convention not in ('gaussian_sum','tempered_mean'):raise ValueError('Unsupported FK observation likelihood convention')
        if any(not math.isfinite(v) or v<0 for v in [self.observation_weight,self.physics_weight]):raise ValueError('FK energy weights must be finite nonnegative')
        if not isinstance(self.physics_scales,dict) or any(not isinstance(k,str) or not math.isfinite(v) or v<=0 for k,v in self.physics_scales.items()):raise ValueError('PDE scales must be positive finite component scales')
        return self


class InvalidPopulation(FloatingPointError):pass


def normalize(log_weights):
    x=log_weights.detach().cpu().double()
    if x.ndim!=2 or torch.isnan(x).any() or torch.isposinf(x).any() or not torch.isfinite(x).any(1).all():raise InvalidPopulation('Invalid/no finite log-weight population; no uniform fallback')
    maximum=x.max(1,keepdim=True).values
    shifted=x-maximum
    shifted_normalizer=torch.logsumexp(shifted,1,keepdim=True)
    return shifted-shifted_normalizer,(maximum+shifted_normalizer)[:,0]


def weight_summary(log_weights):
    w=log_weights.exp();ess=1/w.square().sum(1)
    return {'ess':ess.tolist(),'ess_fraction':(ess/w.shape[1]).tolist(),'max_weight':w.max(1).values.tolist(),'entropy':(-(w*log_weights.masked_fill(w==0,0)).sum(1)).tolist()}


def systematic(log_weights,generators):
    weights=log_weights.detach().cpu().double().exp()
    if weights.ndim!=2 or not torch.isfinite(weights).all() or not torch.allclose(weights.sum(1),torch.ones(weights.shape[0],dtype=torch.float64),atol=1e-10):raise InvalidPopulation('Systematic resampling requires normalized finite weights')
    rows=[];p=weights.shape[1]
    for b,w in enumerate(weights):
        cdf=w.cumsum(0);cdf[-1]=1.
        positions=(torch.arange(p,dtype=torch.float64)+torch.rand((),generator=generators[b],dtype=torch.float64))/p
        rows.append(torch.searchsorted(cdf,positions).clamp_max(p-1))
    return torch.stack(rows)


def gather(tensor,parents):
    if tensor.shape[:2]!=parents.shape:raise ValueError('Joint particle component axes mismatch')
    ix=parents.to(tensor.device).reshape(*parents.shape,*([1]*(tensor.ndim-2))).expand_as(tensor)
    return torch.gather(tensor,1,ix).clone()


@dataclass
class ParticleState:
    components: dict[str,torch.Tensor]
    log_weights: torch.Tensor
    previous_potential: torch.Tensor
    ancestors: torch.Tensor
    coordinate: float = 0.
    history: list = field(default_factory=list)
    @property
    def model_state(self):return self.components["model"]
    def resample(self,parents):
        self.components={k:gather(v,parents) for k,v in self.components.items()}
        self.previous_potential=gather(self.previous_potential,parents)
        self.ancestors=gather(self.ancestors,parents)
        self.log_weights=torch.full_like(self.log_weights,-math.log(self.log_weights.shape[1]))


def score_from_velocity(x,s,v):
    if not 0<=s<1:raise ValueError('Score identity is defined only for s<1; use nonsingular corrected drift at endpoint')
    return (s*v-x)/(1-s)


def corrected_drift(x,s,v,kappa):
    return v+kappa*(1-s)*(s*v-x)


def stream(seed,b,name):
    n=int.from_bytes(hashlib.sha256(f'fk-v1:{seed}:{b}:{name}'.encode()).digest()[:8],'big')%(2**63)
    return torch.Generator(device='cpu').manual_seed(n)


@dataclass
class ParticleRun:
    state: ParticleState
    diagnostics: dict
    complete: bool
    stop_reason: str
    rng_states: dict


def simulate(velocity,energy,shape,steps,seed,config=None,*,device='cpu',dtype=torch.float64,schedule='uniform',method='fk',cancelled=None,budget=None,event=None,components=None,observer=None):
    """Callbacks preserve [B,P,...]; energy returns named [B,P] contributions."""
    from pdeflow.sampling import generation_schedule
    from pdeflow.process import BudgetReached
    start=time.perf_counter()
    cfg=(config or FKConfig()).validate()
    if method not in ('fk','importance','base','no_resampling'):raise ValueError('Unknown particle method')
    if len(shape)<3 or shape[0]<1 or not 1<=shape[1]<=65536:raise ValueError('Expected [problem,particle,...] population')
    b,p=shape[:2];nodes=generation_schedule(schedule,steps)
    rngs={n:[stream(seed,k,n) for k in range(b)] for n in ['source','noise','resample']}
    random=lambda name:torch.stack([torch.randn(shape[1:],generator=g,dtype=dtype) for g in rngs[name]]).to(device)
    if components and 'model' in components:raise ValueError('model component is owned by the base source')
    x=random('source');parts={'model':x,**(components or {})}
    if any(v.shape[:2]!=(b,p) for v in parts.values()):raise ValueError('Associated particle component axes mismatch')
    state=ParticleState(parts,torch.full((b,p),-math.log(p),dtype=torch.float64),torch.zeros(b,p,dtype=torch.float64),torch.arange(p).expand(b,p).clone())
    costs={'model_evaluations':0,'model_particle_evaluations':0,'endpoint_evaluations':0,'energy_evaluations':0,'energy_particle_evaluations':0,'steps':0,'resampling_events':0,'gradient_evaluations':0,'weight_transfer_seconds':0.,'reference_solver_calls':0}
    increments=[];log_z=torch.zeros(b,dtype=torch.float64);last_components={};stop='completed_schedule'
    def chunked(callback,tensor,s=None,kind='model'):
        output=[]
        for lo in range(0,p,cfg.chunk_size):
            hi=min(p,lo+cfg.chunk_size);size=b*(hi-lo)
            if kind=='model':
                if costs['model_particle_evaluations']+size>cfg.max_particle_evaluations:raise BudgetReached('particle evaluations')
                if budget:budget('network_evaluations',1)
                costs['model_evaluations']+=1;costs['model_particle_evaluations']+=size
            else:costs['energy_evaluations']+=1;costs['energy_particle_evaluations']+=size
            out=callback(tensor[:,lo:hi],s) if kind=='model' else callback(tensor[:,lo:hi])
            output.append(out)
        if kind=='model':return torch.cat(output,1)
        return {k:torch.cat([v[k] for v in output],1) for k in output[0]}
    def potential(x,s,v=None):
        nonlocal last_components
        beta=0. if method in ('base','importance') and s<1 else (0. if method=='base' else cfg.beta_start+(1-cfg.beta_start)*s)
        if beta==0:return torch.zeros(b,p,dtype=torch.float64)
        estimate=x if s==1 else x+(1-s)*v
        costs['endpoint_evaluations']+=b*p
        values=chunked(energy,estimate,kind='energy')
        transfer=time.perf_counter();last_components={k:t.detach().cpu().double() for k,t in values.items()};costs['weight_transfer_seconds']+=time.perf_counter()-transfer
        total=sum(last_components.values())
        if total.shape!=(b,p) or not torch.isfinite(total).all():raise InvalidPopulation(f'Nonfinite energy population ({int((~torch.isfinite(total)).sum())} entries); no weight fallback')
        return beta*total
    def reweight(phi):
        nonlocal log_z
        increment=-phi+state.previous_potential
        state.log_weights,normalizer=normalize(state.log_weights+increment)
        log_z+=normalizer
        if not torch.isfinite(log_z).all():raise InvalidPopulation('Nonfinite accumulated log-normalizer; rescale the stated energy')
        state.previous_potential=phi;increments.append(increment.tolist()) if cfg.record_particle_history else None
    with torch.no_grad():
        try:
            v=chunked(velocity,x,0.);reweight(potential(x,0.,v))
            if observer:observer(x,0.,0,x+v,None)
            for i,(s,t) in enumerate(zip(nodes,nodes[1:]),1):
                if cancelled and cancelled():stop='STOP_CANCELLED';break
                if budget:budget('sampling_steps',1)
                h=t-s
                proposal=x+h*corrected_drift(x,s,v,cfg.kappa)+math.sqrt(2*cfg.kappa*h)*(1-s)*random('noise')
                if not torch.isfinite(proposal).all():raise InvalidPopulation('Nonfinite stochastic proposal')
                next_v=None if t==1 else chunked(velocity,proposal,t)
                phi=potential(proposal,t,next_v);reweight(phi)
                x=proposal;state.components['model']=x;state.coordinate=t;costs['steps']+=1
                before=weight_summary(state.log_weights)
                ess=torch.tensor(before['ess_fraction'])
                eligible=i%cfg.resample_every==0 and t<1 and method=='fk'
                selected=(ess<cfg.ess_fraction) if eligible else torch.zeros(b,dtype=torch.bool)
                terminal=t==1 and cfg.terminal_resample and method!='base'
                if terminal:selected[:]=True
                parents=torch.arange(p).expand(b,p).clone()
                if selected.any():
                    sampled=systematic(state.log_weights,rngs['resample']);parents[selected]=sampled[selected]
                    # Nonselected problems retain their normalized weights.
                    old_weights=state.log_weights.clone();state.resample(parents);state.log_weights[~selected]=old_weights[~selected]
                    x=state.components['model'];next_v=gather(next_v,parents) if next_v is not None else None
                    last_components={k:gather(val,parents) for k,val in last_components.items()}
                    costs['resampling_events']+=int(selected.sum())
                if observer:observer(x,t,i,x if t==1 else x+(1-t)*next_v,parents[0].tolist())
                after=weight_summary(state.log_weights)
                trace={'step':i,'generation_position':t,'before':before,'after':after,'resampled_problems':selected.tolist(),'parents':parents.tolist() if selected.any() and cfg.record_particle_history else None,'unique_ancestors':[len(torch.unique(row)) for row in state.ancestors],'trigger':'terminal_systematic' if terminal else 'scheduled_ESS' if selected.any() else 'none','energy_components_weighted_mean':{k:(val*state.log_weights.exp()).sum(1).tolist() for k,val in last_components.items()},'cost':dict(costs)}
                state.history.append(trace)
                # Retain every numerical diagnostic; throttle only live notifications.
                if event and (i%cfg.diagnostic_every==0 or selected.any() or t==1):event(trace)
                v=next_v
        except BudgetReached as error:stop='FINISH_BUDGET_EXHAUSTED';costs['budget_reason']=str(error)
    if device!='cpu':
        if str(device).startswith('mps'):torch.mps.synchronize()
        elif str(device).startswith('cuda'):torch.cuda.synchronize()
    costs.update(seconds=time.perf_counter()-start,log_normalizer_estimate=log_z.tolist(),incremental_log_potentials=increments,trace=state.history,config=asdict(cfg),method=method,nonfinite_energy_count=0,nonfinite_weight_count=0,weight_device='cpu_float64',random_device='cpu_explicit',inference_resume=False)
    rng_states={k:[base64.b64encode(g.get_state().numpy().tobytes()).decode() for g in bank] for k,bank in rngs.items()}
    complete=state.coordinate==1. and stop=='completed_schedule'
    return ParticleRun(state,costs,complete,stop,rng_states)
