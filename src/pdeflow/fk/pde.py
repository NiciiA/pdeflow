"""Physical PDE energy adapter and immutable-target outer-controller integration."""
from dataclasses import asdict
import math,time
import torch
from pdeflow.core import Conditioning,ReconstructionResult
from pdeflow.fk import FKConfig,simulate,weight_summary
from pdeflow.inference import validate_compatibility
from pdeflow.process import ProcessConfig,ProcessState
from pdeflow.process.diagnostics import report
from pdeflow.specialists import get_specialist
from pdeflow.runtime import process_peak_memory_bytes


class PDEEnergy:
    def __init__(self,bundle,problem,observations,config):
        self.bundle,self.problem,self.observations,self.config=bundle,problem,observations,config
    def __call__(self,x):
        if x.shape[0]!=1:raise ValueError('PDE adapter reconstructs one request; generic FK engine supports independent problem batches')
        model=x[0];physical=self.bundle.transform.decode(self.bundle.layout.unpack(model));n=len(physical);obs=self.observations;cfg=self.config
        if not torch.isfinite(physical).all():raise FloatingPointError('Nonfinite decoded FK physical fields')
        predicted=obs.operator.apply(physical);values=obs.values.to(predicted);mask=obs.mask.to(predicted.device)
        noise=obs.field_noise_std or (obs.noise_std,)*obs.operator.channels
        tolerance=obs.field_tolerance or (obs.noiseless_tolerance,)*obs.operator.channels
        scales=predicted.new_tensor([max(a,b) if cfg.observation_convention=='tempered_mean' else a if a>0 else b for a,b in zip(noise,tolerance)])[obs.operator.indices[:,0].to(predicted.device)]
        error=(predicted-values)/scales
        energy=.5*(error.square()*mask).sum(1)
        if cfg.observation_convention=='tempered_mean':energy=energy/mask.sum(1)
        components={'observation':(cfg.observation_weight*energy)[None]}
        physics=get_specialist(self.problem.family).physics(self.problem,Conditioning.from_problem(self.problem,n,device=physical.device,dtype=physical.dtype)).evaluate(physical)
        r=physics.pointwise['pde'];w=physics.time_weights.reshape(1,-1,*([1]*self.problem.grid.dimension))
        names=['u'] if r.shape[1]==1 else [f.name for f in self.problem.fields]
        if set(cfg.physics_scales)-set(names):raise ValueError('Unknown PDE energy component scales')
        for i,name in enumerate(names):
            weights=w.expand_as(r[:,i]);axes=tuple(range(1,weights.ndim))
            value=.5*((r[:,i]/cfg.physics_scales.get(name,1.)).square()*weights).sum(axes)/weights.sum(axes)
            components['pde.'+name]=(cfg.physics_weight*value)[None]
        return components


def run(bundle,problem,observations,inference,mode,progress=None,*,cancelled=None,process_event=None,fine_bundle=None):
    if inference.fk is None:raise ValueError('FK/importance modes require explicit inference.fk settings')
    cfg=FKConfig(**inference.fk).validate();policy=ProcessConfig(**(inference.controller or {})).validate()
    if policy.refine or policy.fine_model_id or fine_bundle is not None or policy.max_attempts!=1:raise ValueError('FK plus refinement/retry is unsupported; use one immutable target/grid per run')
    if bundle.config.formulation!='independent_gaussian_linear_cfm':raise ValueError('FK requires independent Gaussian-source linear CFM score capability')
    validate_compatibility(problem,bundle.config,inference)
    specialist=get_specialist(problem.family);specialist.validate_observations(problem,observations)
    if math.prod(problem.grid.shape)>policy.max_grid_points or inference.particles>policy.max_particles:raise ValueError('FK grid/particle count exceeds explicit controller limit')
    state=ProcessState(policy);state.consumed['attempts']=1
    def budget(kind,amount):
        state.budget(kind,amount);state.consumed[kind]+=amount
    parameter=next(bundle.model.parameters());device,dtype=parameter.device,parameter.dtype
    def velocity(x,s):
        flat=x.flatten(0,1);condition=Conditioning.from_problem(problem,len(flat),device=device,dtype=dtype)
        return bundle.model(flat,flat.new_full((len(flat),),s),condition).reshape_as(x)
    energy=PDEEnergy(bundle,problem,observations,cfg)
    def event(trace):
        action='RESAMPLE' if any(trace['resampled_problems']) else 'CONTINUE_SAMPLING'
        evt=state.action(action,'Prescribed systematic FK resampling' if action=='RESAMPLE' else 'Continue fixed base transition/potential schedule; no gradient proposal',generation_position=trace['generation_position'],ess_before=trace['before'],ess_after=trace['after'],trigger=trace['trigger'],parents=trace['parents'],unique_ancestors=trace['unique_ancestors'])
        if process_event:process_event({'stage':'fk_resampling' if action=='RESAMPLE' else 'fk_sampling','decision':evt,'fk':trace})
        if progress:progress(trace['step'],inference.steps)
    collector=None
    if inference.snapshots is not None:
        from pdeflow.sampling.snapshots import SnapshotCollector
        collector=SnapshotCollector(bundle,problem,inference)
    def observe(x,s,step,estimate,parents):
        collector.record(x[0],s,step,estimate[0],parents,force=s==1.)
    run=simulate(velocity,energy,(1,inference.particles,bundle.layout.channels*len(problem.grid.times),*problem.grid.shape),inference.steps,inference.seed,cfg,device=device,dtype=dtype,schedule=inference.schedule,method=mode,cancelled=cancelled,budget=budget,event=event,observer=observe if collector else None)
    with torch.no_grad():physical=bundle.transform.decode(bundle.layout.unpack(run.state.components['model'][0]))
    final=report(physical,problem,observations,'terminal_fk_field' if run.complete else 'incomplete_fk_state',policy.thresholds)
    reason=run.stop_reason
    if run.complete:reason='FINISH_CRITERIA_MET' if final['criteria_met'] else 'FINISH_BUDGET_EXHAUSTED'
    if not final['finite']:reason='FAIL_INVALID_NUMERICS'
    state.action(reason,'Actual terminal verification completed' if run.complete else 'Path incomplete; not terminal-target samples')
    process={'config':asdict(policy),'actions':state.actions,'diagnostic_history':[final],'consumed':state.consumed,'final_grid':asdict(problem.grid),'stop_reason':reason,'reference_used':False,'selection':'weighted FK population or explicitly terminal-systematic-resampled population','resampling':'systematic at fixed eligible times with ESS trigger','criteria_met':reason=='FINISH_CRITERIA_MET','terminal_target_complete':run.complete}
    diagnostics={**run.diagnostics,'fk':{**run.diagnostics,'terminal_target_complete':run.complete,'terminal_weights':run.state.log_weights.exp().tolist(),'unique_ancestors':[len(torch.unique(row)) for row in run.state.ancestors],'weight_summary':weight_summary(run.state.log_weights)},'final_verification':final,'process':process,'particles':inference.particles,'process_peak_rss_bytes':process_peak_memory_bytes(),'memory_scope':'process_lifetime_high_water_mark','ode_evaluations':0,'stepper':cfg.stepper,'guidance_trace':[],'inference_state':{'generation_position':run.state.coordinate,'specialist':specialist.id,'checkpoint':bundle.metadata['checkpoint_sha256'],'seed':inference.seed,'physical_grid':asdict(problem.grid),'rng_states':run.rng_states,'inference_resume':False},'energy_conventions':{'observation':cfg.observation_convention,'observation_weight':cfg.observation_weight,'physics_weight':cfg.physics_weight,'physics_scales':cfg.physics_scales,'boundary_initial_terms':'not applicable; periodic representation and unknown initial state','reference_used':False}}
    provenance={**bundle.metadata,'mode':mode,'sampling_seed':inference.seed,'population_interaction':mode=='fk' or cfg.terminal_resample,'posterior_claim':False,'specialist':specialist.id,'terminal_target_complete':run.complete,'target':'Euler-Maruyama base terminal law tilted by stated actual terminal energy; learned-flow equivalence approximate','representation':specialist.fields,'weights':'terminal_systematic_unweighted' if cfg.terminal_resample else 'normalized_nonuniform','inference_resume':False}
    if process_event:process_event({'stage':'final_verification','stop_reason':reason})
    return ReconstructionResult(physical[None].detach().cpu(),diagnostics,provenance,stop_reason=reason,corrected=True,inference_state=run.state,weights=run.state.log_weights.exp(),generation_snapshots=collector.finish() if collector else None)
