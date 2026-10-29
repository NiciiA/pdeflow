"""Immutable reference artifacts. Reference arrays never belong to inference inputs."""
from dataclasses import asdict
from pathlib import Path
import time
import uuid
import numpy as np
from pdeflow.core import ProblemSpec
from pdeflow.runtime import atomic_write,write_json,read_json,digest,code_identity,file_digest
from pdeflow.local import now
from .solvers import ReferenceProblem,solve,compile_problem,capabilities
from .fields import field_array

def create(spec: ReferenceProblem,output: str | Path,identifier: str | None=None) -> dict:
    output=Path(output)
    if output.exists():raise FileExistsError('Reference destination exists; create a new immutable artifact')
    started=time.perf_counter();problem,values,diagnostics=solve(spec)
    output.mkdir(parents=True)
    identifier=identifier or uuid.uuid4().hex
    atomic_write(output/'reference.npz',lambda f:np.savez_compressed(f,values=values[None]))
    write_json(output/'problem.json',problem.to_dict())
    write_json(output/'reference-problem.json',asdict(spec))
    manifest={'schema_version':1,'kind':'reference_solution','id':identifier,'created_at':now(),'source':code_identity(),'specification_identity':digest(asdict(spec)),
        'reference_problem':'reference-problem.json','problem':problem.to_dict(),'coordinates':{a:c.tolist() for a,c in zip(problem.grid.axis_names,problem.grid.coordinates())},
        'physical_times':list(problem.grid.times) if problem.grid.temporal_kind=='transient' else None,'fields':[asdict(f) for f in problem.fields],
        'array_sha256':file_digest(output/'reference.npz'),'diagnostics':diagnostics,'seed':spec.seed,'seconds':time.perf_counter()-started,'meaning':'Reference-only full inputs. Numerical references are approximations; no neural checkpoint was loaded.'}
    write_json(output/'reference.json',manifest)
    return {**manifest,'path':str(output.resolve())}

def load(path: str | Path):
    path=Path(path)
    if path.is_file():path=path.parent
    manifest=read_json(path/'reference.json')
    if manifest.get('kind')!='reference_solution' or manifest.get('schema_version')!=1:raise ValueError('Unsupported reference artifact')
    if digest(read_json(path/'reference-problem.json'))!=manifest['specification_identity']:raise ValueError('Reference specification checksum mismatch')
    if read_json(path/'problem.json')!=manifest['problem']:raise ValueError('Reference physical metadata mismatch')
    if file_digest(path/'reference.npz')!=manifest['array_sha256']:raise ValueError('Reference array checksum mismatch')
    problem=ProblemSpec.from_dict(manifest['problem'])
    with np.load(path/'reference.npz',allow_pickle=False) as arrays:values=arrays['values'].copy()
    if values.shape!=(1,len(problem.fields),len(problem.grid.times),*problem.grid.shape) or not np.isfinite(values).all():raise ValueError('Reference shape/values invalid')
    return problem,values,manifest

def preview_inputs(spec: ReferenceProblem,field: str,axis=2,index=0):
    p,_=compile_problem(spec)
    if field not in spec.inputs and field not in p.known_fields:raise ValueError('Choose one explicitly supplied reference input')
    value=field_array(spec.inputs[field] if field in spec.inputs else p.known_fields[field],p.grid)
    return slice_payload(p,value,field,axis,index)

def slice_payload(problem: ProblemSpec,value: np.ndarray,field: str,axis=2,index=0,time_index=None):
    grid=problem.grid;axes=list(grid.axis_names)
    if grid.dimension==3:
        if not 0<=axis<3 or not 0<=index<grid.shape[axis]:raise ValueError('Reference slice out of bounds')
        value=np.take(value,index,axis=axis);axes.pop(axis)
    native_shape=list(grid.shape);strides=[max(1,__import__('math').ceil(n/128)) for n in value.shape]
    value=value[tuple(slice(None,None,k) for k in strides)]
    return {'values':value.tolist(),'reference':None,'error':None,'shape':list(value.shape),'native_shape':native_shape,'display_downsampled':any(k>1 for k in strides),'axes':axes,'coordinates':{a:grid.coordinates()[grid.axis_names.index(a)][::k].tolist() for a,k in zip(axes,strides)},
        'field':field,'units':'nondimensional','physical_time':grid.times[time_index] if time_index is not None and grid.temporal_kind=='transient' else None,
        'slice_coordinate':float(grid.coordinates()[axis][index]) if grid.dimension==3 else None,'vmin':float(value.min()),'vmax':float(value.max()),'observations':[],'statistic':'reference-only supplied input' if time_index is None else 'computed reference; evaluation only'}

def preview(path: str | Path,field_index=0,time_index=0,axis=2,index=0):
    p,values,m=load(path)
    if not 0<=field_index<len(p.fields) or not 0<=time_index<len(p.grid.times):raise ValueError('Reference field/time index out of bounds')
    return {**slice_payload(p,values[0,field_index,time_index],p.fields[field_index].name,axis,index,time_index),'reference_id':m['id'],'diagnostics':m['diagnostics']}
