"""Stateless physical sampling, distinct from reference solving and CFM pairs."""
from dataclasses import asdict,dataclass
import time
import json
import numpy as np
import torch
from pdeflow.core import Conditioning,ProblemSpec
from pdeflow.runtime import digest

NAMESPACES=('train','calibration','validation','test')

@dataclass
class PhysicalSample:
    specification: dict
    problem: ProblemSpec
    values: torch.Tensor  # [C,T,*space], clean physical fields
    diagnostics: dict
    timings: dict


def distribution(config):
    d=config.data
    value={'version':config.provider.distribution_version,'family':config.specialist,'grid':asdict(config.grid),
        'root_seed':d.seed,'bandwidth':d.bandwidth,'bandwidth_range':d.bandwidth_range,
        'amplitude_range':d.amplitude_range,'offset_range':d.offset_range,
        'diffusivity_range':d.diffusivity_range if config.specialist=='heat' else None,
        'chemistry':asdict(config.chemistry) if config.chemistry else None,
        'solver':{'version':get_solver_version(config.specialist),'rtol':config.provider.solver_rtol,'max_iterations':config.provider.solver_max_iterations}}
    return json.loads(json.dumps(value))


def get_solver_version(family):
    return {'heat':'numpy_fft_heat_v1','darcy':'harmonic_fv_projected_cg_v1','chemistry':'positive_strang_exact_reaction_v1'}[family]


def seed_for(config,namespace,index,attempt=0,stream='physical'):
    if namespace not in NAMESPACES or index<0 or attempt<0:raise ValueError('Invalid physical ID allocation')
    return int(digest({'distribution':distribution(config),'namespace':namespace,'index':index,'attempt':attempt,'stream':stream})[:32],16)


def specification(config,namespace,index,attempt=0):
    start=time.perf_counter();seed=seed_for(config,namespace,index,attempt);rng=np.random.default_rng(seed);d=config.data
    spec={'id':f'{namespace}-'+digest({'distribution':distribution(config),'namespace':namespace,'index':index,'attempt':attempt})[:24],
        'namespace':namespace,'index':index,'attempt':attempt,'seed':str(seed),'distribution':distribution(config),
        'initial_seed':str(seed_for(config,namespace,index,attempt,'initial')),'forcing_seed':str(seed_for(config,namespace,index,attempt,'forcing')),
        'solver_seed':str(seed_for(config,namespace,index,attempt,'solver')),'parameters':{}}
    from pdeflow.specialists import get_specialist
    get_specialist(config.specialist).sample_spec(config,spec,rng)
    return spec,time.perf_counter()-start


def solve_spec(config,spec,sampling_seconds=0.):
    if spec['distribution'] != distribution(config):
        # JSON roundtrips turn tuples into lists; compare canonical serialization.
        if digest(spec['distribution'])!=digest(distribution(config)):raise ValueError('Physical specification/solver/distribution mismatch')
    from pdeflow.specialists import get_specialist
    start=time.perf_counter();problem,values,diagnostics=get_specialist(config.specialist).solve_spec(config,spec)
    values=torch.as_tensor(values,dtype=torch.float64)
    if tuple(values.shape)!=(len(problem.fields),len(config.grid.times),*config.grid.shape) or not torch.isfinite(values).all():raise ValueError('Invalid clean reference layout/values')
    return PhysicalSample(spec,problem,values,diagnostics,{'sampling_seconds':sampling_seconds,'reference_seconds':time.perf_counter()-start})


def combine(samples):
    if not samples:raise ValueError('Empty physical batch')
    family=samples[0].problem.family
    if any(s.problem.family!=family for s in samples):raise ValueError('Mixed specialist batch')
    params={k:torch.tensor([s.problem.known_parameters[k] for s in samples],dtype=torch.float64) for k in samples[0].problem.known_parameters}
    maps={k:torch.tensor(np.stack([s.problem.known_fields[k] for s in samples]),dtype=torch.float64) for k in samples[0].problem.known_fields}
    return torch.stack([s.values for s in samples]),Conditioning(family,params,maps)
