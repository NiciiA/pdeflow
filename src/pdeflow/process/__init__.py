"""Bounded deterministic feedback, distinct from generative integration and FK."""
from dataclasses import asdict,dataclass,field,replace
import math
import time
from pdeflow.process.diagnostics import report


@dataclass(frozen=True)
class ProcessConfig:
    check_interval: int = 2
    max_network_evaluations: int = 128
    max_sampling_steps: int = 32
    max_guidance_steps: int = 32
    max_attempts: int = 1
    max_refinement_levels: int = 1
    max_grid_points: int = 32768
    max_particles: int = 16
    wall_seconds: float | None = None
    patience: int = 2
    stagnation_relative: float = .001
    thresholds: dict[str,float] = field(default_factory=dict)
    refine: bool = False
    fine_model_id: str | None = None
    request_resampling: bool = False

    def validate(self):
        for name in ['check_interval','max_network_evaluations','max_sampling_steps','max_guidance_steps','max_attempts','max_grid_points','max_particles','patience']:
            if type(getattr(self,name)) is not int or getattr(self,name)<1:raise ValueError(f'{name} must be positive integer')
        if type(self.max_refinement_levels) is not int or not 0<=self.max_refinement_levels<=1:raise ValueError('Only zero/one global refinement level implemented')
        if self.wall_seconds is not None and (not math.isfinite(self.wall_seconds) or self.wall_seconds<=0):raise ValueError('wall_seconds must be positive')
        if not 0<=self.stagnation_relative<1:raise ValueError('Invalid stagnation relative tolerance')
        if not isinstance(self.thresholds,dict):raise ValueError('thresholds must map component names to upper bounds')
        if any(not isinstance(k,str) or not math.isfinite(v) or v<0 for k,v in self.thresholds.items()):raise ValueError('Thresholds must be finite nonnegative component upper bounds')
        if type(self.refine) is not bool or type(self.request_resampling) is not bool:raise ValueError('Action flags must be boolean')
        # Bound even trusted local configurations; a policy must not create an unbounded action loop.
        if self.max_attempts>16 or self.max_network_evaluations>16384 or self.max_sampling_steps>4096 or self.max_particles>256:raise ValueError('Process exceeds bounded local limits')
        return self


class BudgetReached(Exception):pass


@dataclass
class ProcessState:
    config: ProcessConfig
    consumed: dict = field(default_factory=lambda:{'network_evaluations':0,'sampling_steps':0,'guidance_steps':0,'attempts':0,'refinement_levels':0})
    diagnostics: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    started: float = field(default_factory=time.monotonic)
    guidance_enabled: bool = True
    passing_streak: int = 0

    def budget(self,kind,amount=1):
        if self.config.wall_seconds is not None and time.monotonic()-self.started>=self.config.wall_seconds:raise BudgetReached('wall_time')
        limit=getattr(self.config,'max_'+kind)
        if self.consumed[kind]+amount>limit:raise BudgetReached(kind)

    def action(self,action,reason,**context):
        event={'index':len(self.actions),'action':action,'reason':reason,'budget':dict(self.consumed),**context}
        self.actions.append(event);return event

    def intermediate(self,diagnostic,step,s,mode):
        diagnostic['observation']={k:v for k,v in diagnostic['observation'].items() if k not in ('raw_residuals','noise_standardized_residuals')}
        self.diagnostics.append(diagnostic)
        if not diagnostic['finite']:return self.action('FAIL_INVALID_NUMERICS','Nonfinite physical diagnostic',step=step,generation_position=s)
        # Estimated endpoints never finish or trigger a new model/refinement.
        self.passing_streak=self.passing_streak+1 if diagnostic['criteria_met'] else 0
        self.guidance_enabled=mode!='unguided' and self.passing_streak<self.config.patience
        return self.action('CONTINUE_WITH_CONFIGURED_GUIDANCE' if self.guidance_enabled else 'CONTINUE_SAMPLING','Estimated endpoint constraints unresolved' if not diagnostic['criteria_met'] else 'Endpoint estimate meets thresholds; complete the trained generation path before final verification',step=step,generation_position=s,failures=diagnostic['mandatory_failures'])

    def stagnant(self):
        n=self.config.patience
        if len(self.diagnostics)<n+1:return False
        def score(r):
            e=r['entries'];return max((v['value']/max(v['threshold'],1e-30) for v in e.values() if v['applicable'] and v['finite'] and v['threshold'] is not None),default=float('inf'))
        scores=[score(r) for r in self.diagnostics[-n-1:]]
        return all(math.isfinite(x) for x in scores) and all(b>=a*(1-self.config.stagnation_relative) for a,b in zip(scores,scores[1:]))


def run(bundle,problem,observations,inference,mode,progress=None,fine_bundle=None,cancelled=None,process_event=None):
    from pdeflow.sampling import _sample_once
    from pdeflow.refinement import refine_request
    policy=ProcessConfig(**inference.controller).validate();state=ProcessState(policy)
    if inference.particles>policy.max_particles or math.prod(problem.grid.shape)>policy.max_grid_points:raise ValueError('Initial physical grid/particles exceed controller limits')
    if policy.request_resampling:state.action('RESAMPLE_UNAVAILABLE','Resampling requires explicit FK mode; independent gradient candidates are not FK')
    current=bundle;p=problem;obs=observations;seed=inference.seed
    attempt_costs=[];guidance_history=[]
    while True:
        state.guidance_enabled=True;state.passing_streak=0
        state.consumed['attempts']+=1
        result=_sample_once(current,p,obs,replace(inference,seed=seed,controller=None),mode,progress,process_state=state,cancelled=cancelled,process_event=process_event)
        attempt_costs.append({'attempt':state.consumed['attempts'],'grid':asdict(p.grid),'checkpoint':current.metadata.get('checkpoint_sha256'),'seconds':result.diagnostics['seconds'],**{k:result.diagnostics[k] for k in ['model_evaluations','ode_evaluations','gradient_evaluations','steps']}})
        guidance_history.extend([{**event,'attempt':state.consumed['attempts'],'physical_grid':list(p.grid.shape)} for event in result.diagnostics['guidance_trace']])
        final=result.diagnostics['final_verification'];state.diagnostics.append(final)
        if result.stop_reason in ('FAIL_INVALID_NUMERICS','STOP_CANCELLED','FINISH_BUDGET_EXHAUSTED'):
            state.action(result.stop_reason,result.diagnostics.get('termination_detail','Sampling interrupted before full schedule'));break
        if not final['finite']:
            state.action('FAIL_INVALID_NUMERICS','Nonfinite final component evaluation');result.stop_reason='FAIL_INVALID_NUMERICS';break
        if final['criteria_met']:
            state.action('FINISH_CRITERIA_MET','Actual returned fields passed all configured mandatory component checks');result.stop_reason='FINISH_CRITERIA_MET';break
        stagnant=state.stagnant()
        if policy.refine and state.consumed['refinement_levels']<policy.max_refinement_levels:
            try:
                if state.consumed['attempts']>=policy.max_attempts:raise ValueError('No attempt budget remains for a fine reconstruction')
                if fine_bundle is None:raise ValueError('No compatible fine checkpoint supplied; interpolation alone is not refinement')
                p,obs=refine_request(p,obs,fine_bundle,inference,policy.max_grid_points)
            except ValueError as error:state.action('REFINE_UNAVAILABLE',str(error))
            else:
                state.consumed['refinement_levels']+=1
                state.action('REFINE','Fresh Gaussian fine-grid reconstruction, same actual measurements; coarse prediction is not a condition',coarse_grid=asdict(current.config.grid),fine_grid=asdict(p.grid),compatibility_rechecked=True,coarse_checkpoint=current.metadata.get("checkpoint_sha256"),fine_checkpoint=fine_bundle.metadata.get("checkpoint_sha256"))
                current=fine_bundle;fine_bundle=None;seed=(seed+1)%(2**63);continue
        if policy.refine and policy.max_refinement_levels==0:
            state.action('REFINE_UNAVAILABLE','Physical refinement level budget is zero')
        if state.consumed['attempts']<policy.max_attempts:
            state.action('RETRY_WITH_DEFINED_POLICY','Diagnostic stagnation at candidate boundary' if stagnant else 'Final mandatory mismatch; fresh independent source, seed incremented by one; last attempt returned, no best-sample selection')
            seed=(seed+1)%(2**63);continue
        state.action('FINISH_BUDGET_EXHAUSTED','Final constraints unresolved; no legal remaining attempt/refinement; execution completion is not verification success',failures=final['mandatory_failures']);result.stop_reason='FINISH_BUDGET_EXHAUSTED';break
    result.diagnostics.update(attempt_costs=attempt_costs,process_seconds_total=time.monotonic()-state.started,guidance_trace=guidance_history)
    for key in ['model_evaluations','ode_evaluations','gradient_evaluations','steps','seconds']:
        result.diagnostics[key]=sum(item[key] for item in attempt_costs)
    result.diagnostics['cost_scope']='All attempts including coarse/fine sampling; process_seconds_total also includes policy/transfer overhead. Preloaded model I/O and conventional baseline cost are separate.'
    result.diagnostics['process']={'config':asdict(policy),'actions':state.actions,'diagnostic_history':state.diagnostics,'consumed':state.consumed,'final_grid':asdict(p.grid),'stop_reason':result.stop_reason,'reference_used':False,'selection':'last attempted candidate population; no reference/best-score selection','resampling':'unavailable; FK absent','criteria_met':result.stop_reason=='FINISH_CRITERIA_MET'}
    result.provenance['final_problem']=p.to_dict();result.provenance['final_observations']=obs.to_dict()
    if process_event:process_event({'stage':'final_verification','stop_reason':result.stop_reason,'actions':state.actions[-1:]})
    return result
