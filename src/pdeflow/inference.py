"""Checkpoint compatibility is delegated to physical specialist declarations."""
from pdeflow.specialists import get_specialist


def validate_inference_budget(problem,inference):
    inference.validate()
    if inference.controller is not None:
        from pdeflow.process import ProcessConfig
        import math
        policy=ProcessConfig(**inference.controller).validate()
        nodes=math.prod(problem.grid.shape)
        if nodes>policy.max_grid_points or inference.particles>policy.max_particles:
            raise ValueError(f'Feedback controller budget is below the requested native grid/population ({nodes} nodes, {inference.particles} particles). Review max_grid_points/max_particles in Feedback controller and budgets, or disable feedback. The physical grid is never reduced automatically.')


def validate_compatibility(problem, checkpoint_config, inference):
    validate_inference_budget(problem,inference)
    get_specialist(checkpoint_config.specialist).validate(problem, checkpoint_config.formulation, inference.stepper, inference.schedule)
    if problem.grid != checkpoint_config.grid:
        raise ValueError("Checkpoint requires the exact spatial dimension, grid, domain, boundary and physical-time/static layout")


def compatibility_report(problem,checkpoint_config,inference=None,candidate=None):
    """All exact layout differences plus physical-law validation; no relaxed physics."""
    from dataclasses import asdict
    inference=inference or __import__('pdeflow.runtime',fromlist=['InferenceConfig']).InferenceConfig()
    expected=get_specialist(checkpoint_config.specialist).default_problem(checkpoint_config.grid)
    mismatches=[]
    def compare(path,want,actual,remedy='Review and adopt the model-compatible layout, or train a matching target'):
        if want!=actual:mismatches.append({'path':path,'expected':want,'actual':actual,'reason':f'{path} differs','remedy':remedy})
    compare('family',checkpoint_config.specialist,problem.family,'Choose a specialist for the current equation; do not change equations implicitly')
    compare('operator_spec',expected.operator_spec,problem.operator_spec,'Use supported equivalent kinetics; different laws require a new specialist')
    compare('fields',[asdict(f) for f in expected.fields],[asdict(f) for f in problem.fields],'Use declared supported species names/roles; arbitrary species renaming is not implemented')
    compare('dimension',checkpoint_config.grid.dimension,problem.grid.dimension)
    for name in ('shape','domain','convention','times','temporal_kind'):
        compare('grid.'+name,asdict(checkpoint_config.grid)[name],asdict(problem.grid)[name])
    error=None
    try:validate_compatibility(problem,checkpoint_config,inference)
    except ValueError as exc:error=str(exc)
    if error and not mismatches:mismatches.append({'path':'physical_contract','expected':'supported specialist conditions','actual':problem.to_dict(),'reason':error,'remedy':'Correct declarations, known conditions or unsupported algorithm choices'})
    return {'candidate':candidate,'requested_capability':{'family':problem.family,'grid':asdict(problem.grid),'operator_spec':problem.operator_spec},'compatible':error is None,'mismatches':mismatches,'summary':'; '.join(m['reason'] for m in mismatches) or 'Compatible'}


def selection_key(identifier,entry):
    """Effort/provenance priority within compatible exports, never evaluation truth."""
    cp=entry['checkpoint'];config=entry['config'];updates=cp.get('progress',0)
    completed=entry.get('run_state')=='completed' and updates>=config.training.steps
    return (-updates,-int(completed),-int('validation-selected' in entry.get('export_kinds',[])),identifier)
