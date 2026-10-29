"""Reproducible shared on-grid observation sampling and explicit disclosure."""
from dataclasses import asdict,dataclass,field
from pathlib import Path
import numpy as np
import torch
from pdeflow.core import ProblemSpec
from pdeflow.observations import PointOperator,ObservationSet
from pdeflow.workbench.observations import observation_rows,parse_rows
from pdeflow.runtime import write_json,atomic_write,digest
from . import load

@dataclass(frozen=True)
class ObservationSamplingSpec:
    pattern: str='fixed_sensors'
    count: int=10
    fields: tuple[str,...]=()
    time_indices: tuple[int,...]=()
    seed: int=54
    noise_std: float=.03
    noise_mode: str="absolute"
    positions: tuple[dict,...]=()
    reuse_indices: tuple[tuple[int,...],...]=()
    disclosure: dict=field(default_factory=dict)

def disclose(problem: ProblemSpec,spec: ObservationSamplingSpec)->dict:
    names={f.name for f in problem.fields}
    choices={n:spec.disclosure.get(n,'partially_measured' if not spec.fields or n in spec.fields else 'unknown') for n in names}
    for n in [*problem.known_parameters,*problem.known_fields]:
        choices[n]=spec.disclosure.get(n,'fully_known')
        if choices[n]!='fully_known':raise ValueError(f'Current {problem.family} checkpoints require fully known {n}; unknown parameter/forcing conditioning is unsupported')
    if set(spec.disclosure)-set(choices):raise ValueError('Disclosure names must match actual physical fields/parameters/forcing')
    for n in names:
        if choices[n] not in ('partially_measured','unknown'):raise ValueError(f'Current joint {problem.family} specialist cannot treat full {n} as a known conditioning field. Choose partially measured or unknown, or train a compatible conditional specialist.')
    if not any(v=='partially_measured' for n,v in choices.items() if n in names):raise ValueError('At least one physical field must be partially measured')
    return choices

def sample(reference: str | Path,spec: ObservationSamplingSpec,output: str | Path)->dict:
    problem,values,manifest=load(reference);grid=problem.grid
    if spec.pattern not in ['random_spatial','regular','space_time','fixed_sensors','positions','reuse']:raise ValueError('Unknown sampling pattern')
    if type(spec.count)!=int or spec.count<1 or spec.count>20000 or not 0<=spec.seed<2**63 or not np.isfinite(spec.noise_std) or spec.noise_std<0:raise ValueError('Sampling needs count1..20000, nonnegative bounded seed and finite nonnegative noise')
    if spec.noise_mode not in ('absolute','relative_rms'):raise ValueError('Noise mode must be absolute or relative_rms')
    disclosure=disclose(problem,spec)
    fields=spec.fields or tuple(f.name for f in problem.fields if disclosure[f.name]=='partially_measured')
    if not fields or len(set(fields))!=len(fields) or any(n not in disclosure or disclosure[n]!='partially_measured' for n in fields):raise ValueError('Only explicitly partially measured physical fields may be sampled')
    field_ids=[next(i for i,f in enumerate(problem.fields) if f.name==n) for n in fields]
    times=spec.time_indices or tuple(range(len(grid.times)))
    if len(set(times))!=len(times) or any(type(t)!=int or not 0<=t<len(grid.times) for t in times):raise ValueError('Choose unique saved physical-time indices; no temporal interpolation')
    spatial_rng,noise_rng=[np.random.default_rng(s) for s in np.random.SeedSequence(spec.seed).spawn(2)]
    size=int(np.prod(grid.shape));indices=[]
    if spec.pattern=='positions':
        rows=[]
        for position in spec.positions:
            if position.get('field') not in fields:raise ValueError('Explicit position field is not disclosed for measurement')
            row={**position,'value':0.}
            rows.append(row)
        # Shared strict physical-coordinate parser rejects off-grid positions, never snaps.
        op=parse_rows(problem,rows,spec.noise_std).operator
        if any(int(t) not in times for t in op.indices[:,1]):raise ValueError('Explicit positions must use the selected saved times')
        indices=op.indices.tolist()
    elif spec.pattern=='reuse':
        indices=[list(i) for i in spec.reuse_indices]
        if any(i[0] not in field_ids or i[1] not in times for i in indices):raise ValueError('Reused pattern contains undisclosed fields/times')
    elif spec.pattern=='space_time':
        total=len(field_ids)*len(times)*size
        if spec.count>total:raise ValueError('Count exceeds unique field-space-time nodes')
        for flat in spatial_rng.choice(total,spec.count,replace=False):
            c,t,*space=np.unravel_index(flat,(len(field_ids),len(times),*grid.shape))
            indices.append((field_ids[c],times[t],*space))
    else:
        if spec.count>size:raise ValueError('Spatial count exceeds unique nodes; reduce count')
        spatial=np.floor(np.arange(spec.count)*size/spec.count).astype(int) if spec.pattern=='regular' else spatial_rng.choice(size,spec.count,replace=False)
        # random_spatial chooses separate locations at each selected physical time;
        # fixed_sensors reuses one population of locations across all selected times.
        for c in field_ids:
            for t in times:
                selected=spatial_rng.choice(size,spec.count,replace=False) if spec.pattern=='random_spatial' else spatial
                indices.extend((c,t,*np.unravel_index(i,grid.shape)) for i in selected)
    if len(indices)>20000:raise ValueError('Observation set exceeds20000 points; reduce fields/count/times')
    op=PointOperator(grid,torch.tensor(indices,dtype=torch.long),len(problem.fields))
    clean=op.apply(torch.from_numpy(values))
    # Relative sigma uses the complete selected field/time reference RMS, independent of drawn noise.
    sigmas=[]
    for ix in op.indices.tolist():
        rms=float(np.sqrt(np.mean(values[0,ix[0],ix[1]]**2)))
        sigmas.append(spec.noise_std*(rms if spec.noise_mode=='relative_rms' else 1.))
    noisy=clean+torch.from_numpy(noise_rng.normal(size=tuple(clean.shape)))*torch.tensor(sigmas)
    obs=ObservationSet(op,noisy,torch.ones_like(noisy,dtype=torch.bool),max(sigmas,default=0.),measurement_noise_std=tuple(sigmas))
    # Only the existing inverse ProblemSpec and noisy ObservationSet cross inference.
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    write_json(output/'problem.json',problem.to_dict());write_json(output/'observations.json',obs.to_dict())
    write_json(output/'rows.json',observation_rows(obs,problem))
    atomic_write(output/'clean-samples-evaluation-only.npz',lambda f:np.savez_compressed(f,values=clean.numpy()))
    record={'schema_version':1,'kind':'sampled_observations','reference_id':manifest['id'],'reference_path':str(Path(reference).resolve()),'sampling':asdict(spec),'disclosure':disclosure,
        'inverse_identity':digest({'problem':problem.to_dict(),'observations':obs.to_dict()}),'count':len(indices),'count_meaning':'total field-space-time points' if spec.pattern in ['space_time','positions','reuse'] else 'spatial locations per selected field and saved time',
        'noise_definition':'independent Gaussian; absolute sigma or fraction of full reference field RMS at that physical time; zero RMS gives sigma0', 'actual_sigma':sigmas, 'replacement':False, 'restriction':'exact saved grid nodes and times; no interpolation or implicit snapping','clean_values':'evaluation-only separate file; never included in observations.json'}
    from copy import deepcopy
    from pdeflow.runtime import read_json
    raw=read_json(Path(reference)/'reference-problem.json')['request']
    if 'problem' in raw:
        from .examples import from_problem
        mathematical=deepcopy(from_problem(problem,53).request)
    else:mathematical=deepcopy(raw)
    mathematical['known_fields']=problem.known_fields
    mathematical['measurements']={**mathematical['measurements'],'rows':observation_rows(obs,problem),'noise_std':obs.noise_std,'status':'known'}
    inverse={'problem':problem.to_dict(),'observations':obs.to_dict(),'mathematical_request':mathematical,'disclosure':disclosure}
    write_json(output/'inverse-request.json',inverse)
    write_json(output/'sampling.json',record)
    return {**record,'path':str(output.resolve()),'problem':problem.to_dict(),'rows':observation_rows(obs,problem),'mathematical_request':mathematical}
