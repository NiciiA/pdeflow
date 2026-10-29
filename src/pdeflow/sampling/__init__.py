"""Replaceable ODE steppers/schedules and independent-candidate sampling."""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Protocol

import torch

from pdeflow.core import Capabilities, Conditioning, ProblemSpec, ReconstructionResult, SamplingState
from pdeflow.specialists import get_specialist
from pdeflow.generative import make_formulation
from pdeflow.observations import ObservationSet
from pdeflow.runtime import InferenceConfig, process_peak_memory_bytes, synchronize
from pdeflow.steering import ReconstructionEnergy, SplitGradientGuidance
from pdeflow.training import InferenceBundle


@dataclass
class StepResult:
    state: torch.Tensor
    evaluations: int


class ODEStepper(Protocol):
    def advance(self, dynamics: Callable, state: torch.Tensor, s: float, h: float) -> StepResult: ...


class Euler:
    def advance(self, dynamics, state, s, h):
        return StepResult(state + h * dynamics(state, s), 1)


class Heun:
    def advance(self, dynamics, state, s, h):
        k1 = dynamics(state, s)
        k2 = dynamics(state + h * k1, s + h)
        return StepResult(state + .5 * h * (k1 + k2), 2)


def make_stepper(name: str) -> ODEStepper:
    if name not in ("euler", "heun"):
        raise ValueError("Only Euler/Heun ODE steppers implemented; no SDE compatibility")
    return {"euler": Euler, "heun": Heun}[name]()


def generation_schedule(name: str, steps: int) -> list[float]:
    if steps < 1 or name not in ("uniform", "quadratic"):
        raise ValueError("Expected a positive step count and uniform/quadratic schedule")
    s = torch.linspace(0, 1, steps + 1, dtype=torch.float64)
    return (s if name == "uniform" else s.square()).tolist()


def _sample_once(bundle: InferenceBundle, problem: ProblemSpec, observations: ObservationSet,
           config: InferenceConfig, mode: str = "unguided", progress=None, process_state=None, cancelled=None, process_event=None) -> ReconstructionResult:
    config.validate()
    if mode not in ("unguided", "observation", "observation_pde"):
        raise ValueError("Sampling mode must be unguided, observation or observation_pde")
    specialist = get_specialist(bundle.config.specialist)
    specialist.validate(problem, bundle.config.formulation, config.stepper, config.schedule)
    if problem.grid != bundle.layout.grid or observations.operator.grid != problem.grid:
        raise ValueError("Inference requires checkpoint's exact grid and physical-time layout")
    if observations.values.shape[0] != 1 or config.particles < 1:
        raise ValueError("This entry point reconstructs one problem with >=1 independent candidate particles")
    specialist.validate_observations(problem, observations)
    # Physics factory validates the operator even for unguided sampling.
    parameter = next(bundle.model.parameters())
    device, dtype = parameter.device, parameter.dtype
    n = config.particles
    condition = Conditioning.from_problem(problem, n, device=device, dtype=dtype)
    physics = specialist.physics(problem, condition)
    formulation = make_formulation(bundle.config.formulation)
    stepper, schedule = make_stepper(config.stepper), generation_schedule(config.schedule, config.steps)
    source_rng = torch.Generator(device="cpu").manual_seed(config.seed)
    source = torch.randn((1, n, bundle.layout.channels * len(problem.grid.times), *problem.grid.shape), generator=source_rng, dtype=dtype).to(device)
    state = SamplingState(source, 0., 0, source_rng)
    repeated_obs = ObservationSet(observations.operator, observations.values.repeat_interleave(n, dim=0),
                                  observations.mask.repeat_interleave(n, dim=0), observations.noise_std, observations.noiseless_tolerance, observations.field_noise_std, observations.field_tolerance)
    decode = lambda x: bundle.transform.decode(bundle.layout.unpack(x))
    guidance = None
    if mode != "unguided":
        energy = ReconstructionEnergy(repeated_obs, physics, decode, config.observation_strength,
                                       config.physics_strength if mode == "observation_pde" else 0.)
        guidance = SplitGradientGuidance(energy, config.correction_strength, config.gradient_max_norm)
    model_evaluations, ode_evaluations, gradient_evaluations = 0, 0, 0

    def velocity(x, s):
        nonlocal model_evaluations
        if process_state:
            process_state.budget('network_evaluations')
            process_state.consumed['network_evaluations']+=1
        model_evaluations += 1
        return bundle.model(x, x.new_full((len(x),), s), condition)

    def endpoint(x, s):
        # At the data endpoint the estimator is the state itself; no singular conversion.
        return x if s == 1. else formulation.endpoint(x, s, velocity(x, s))

    collector=None
    if config.snapshots is not None:
        from .snapshots import SnapshotCollector
        collector=SnapshotCollector(bundle,problem,config)
    def capture(x,s,step,force=False):
        if collector is None or (step not in collector.steps and not force):return
        with torch.no_grad():
            if s==1.:estimate=x
            else:
                # Passive evaluation, excluded from scientific controller budget; cost separately recorded.
                estimate=formulation.endpoint(x,s,bundle.model(x,x.new_full((len(x),),s),condition));collector.calls+=1
        collector.record(x,s,step,estimate,force=force)
    capture(state.model_state.flatten(0,1),0.,0)
    trace = []
    synchronize(device)
    start = time.perf_counter()
    from pdeflow.process import BudgetReached
    from pdeflow.process.diagnostics import report
    stop_reason='completed_schedule';termination_detail='Full generation path completed'
    for j, (s, snext) in enumerate(zip(schedule, schedule[1:])):
        try:
            if cancelled and cancelled():
                stop_reason='STOP_CANCELLED';termination_detail='Cancellation observed at generation boundary';break
            if process_state:process_state.budget('sampling_steps')
            h = snext - s
            x = state.model_state.flatten(0, 1)
            with torch.no_grad():
                result = stepper.advance(velocity, x, s, h)
            ode_evaluations += result.evaluations
            x = result.state
            if guidance is not None and (process_state is None or process_state.guidance_enabled):
                if process_state:process_state.budget('guidance_steps')
                x, diagnostic = guidance.correct(x, snext, h, endpoint)
                gradient_evaluations += 1
                if process_state:process_state.consumed['guidance_steps']+=1
                trace.append({'step':j+1,'generation_time':snext,'label':'estimated_endpoint_before_split_correction','placement':'after full ODE proposal; once per interval, strength × actual interval h',**diagnostic})
            if not torch.isfinite(x).all():raise FloatingPointError('Nonfinite generated state')
            state.model_state = x.reshape(1, n, *x.shape[1:])
            state.generation_time, state.step_index = snext, j + 1
            capture(x,snext,j+1)
            if process_state:
                process_state.consumed['sampling_steps']+=1
                if (j+1)%process_state.config.check_interval==0 and snext<1.:
                    with torch.no_grad():
                        state.estimated_physical_endpoint=decode(endpoint(x,snext)).detach()
                        diagnostic=report(state.estimated_physical_endpoint,problem,observations,'estimated_endpoint',process_state.config.thresholds)
                    event=process_state.intermediate(diagnostic,j+1,snext,mode)
                    if process_event:process_event({'stage':'controller_decision','report':diagnostic,'decision':event})
                    if event['action']=='FAIL_INVALID_NUMERICS':
                        stop_reason='FAIL_INVALID_NUMERICS';termination_detail=event['reason'];break
            if progress is not None:progress(j + 1, config.steps)
        except BudgetReached as error:
            stop_reason='FINISH_BUDGET_EXHAUSTED';termination_detail=str(error);break
        except FloatingPointError as error:
            stop_reason='FAIL_INVALID_NUMERICS';termination_detail=str(error)+'; returning last finite completed state';break
    capture(state.model_state.flatten(0,1),state.generation_time,state.step_index,True)
    synchronize(device)
    decode_start = time.perf_counter()
    with torch.no_grad():
        physical = decode(state.model_state.flatten(0, 1))
        output_diagnostics = specialist.output_diagnostics(state.model_state.flatten(0, 1), bundle.layout, bundle.transform)
    synchronize(device)
    decode_seconds = time.perf_counter() - decode_start
    if not torch.isfinite(physical).all(): raise FloatingPointError("Nonfinite decoded physical fields")
    samples = physical.reshape(1, n, *physical.shape[1:]).detach().cpu()
    synchronize(device)
    final_report=report(physical,problem,observations,'incomplete_generation_field' if state.generation_time<1. else 'final_corrected_field' if guidance is not None or problem.grid.temporal_kind=='static' else 'final_field',process_state.config.thresholds if process_state else {})
    import base64,hashlib
    state_snapshot={'generation_position':state.generation_time,'step':state.step_index,'physical_grid':problem.grid.to_dict() if hasattr(problem.grid,'to_dict') else __import__('dataclasses').asdict(problem.grid),'specialist':specialist.id,'checkpoint':bundle.metadata.get('checkpoint_sha256'),'seed':config.seed,'rng_state_base64':base64.b64encode(source_rng.get_state().numpy().tobytes()).decode(),'particle_ids':[f'{config.seed}:{k}' for k in range(n)],'model_state_shape':list(state.model_state.shape),'model_state_sha256':hashlib.sha256(state.model_state.detach().cpu().numpy().tobytes()).hexdigest(),'endpoint_estimate':'transient diagnostic only; actual returned decode verified separately','reference_used':False}
    diagnostics = {'final_verification':final_report,'inference_state':state_snapshot,'termination_detail':termination_detail,**output_diagnostics, "final_decode_projection_seconds": decode_seconds, "decode_cost_note": "Includes field transform and gauge diagnostics; guidance decodes included in total seconds", "seconds": time.perf_counter() - start, "model_evaluations": model_evaluations,
                   "ode_evaluations": ode_evaluations, "gradient_evaluations": gradient_evaluations,
                   "snapshot_model_evaluations":collector.calls if collector else 0,"particles": n, "steps": state.step_index, "requested_steps":config.steps, "stepper": config.stepper, "schedule": config.schedule,
                   "guidance_trace": trace, "process_peak_rss_bytes": process_peak_memory_bytes(),
                   "memory_scope": "process_lifetime_high_water_mark"}
    return ReconstructionResult(samples, diagnostics, {**bundle.metadata, "mode": mode, "sampling_seed": config.seed,
                                "population_interaction": False, "posterior_claim": False, "specialist": specialist.id, "representation": specialist.fields, "output_gauge_projection": problem.grid.temporal_kind == "static"}, stop_reason=stop_reason, corrected=guidance is not None or problem.grid.temporal_kind == "static", inference_state=state,generation_snapshots=collector.finish() if collector else None)


def sample(bundle,problem,observations,config,mode='unguided',progress=None,*,fine_bundle=None,cancelled=None,process_event=None):
    if mode in ('fk','importance'):
        from pdeflow.fk.pde import run as run_fk
        return run_fk(bundle,problem,observations,config,mode,progress,fine_bundle=fine_bundle,cancelled=cancelled,process_event=process_event)
    if config.controller is not None:
        from pdeflow.process import run
        return run(bundle,problem,observations,config,mode,progress,fine_bundle,cancelled,process_event)
    return _sample_once(bundle,problem,observations,config,mode,progress,cancelled=cancelled,process_event=process_event)
