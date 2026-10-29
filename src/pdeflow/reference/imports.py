"""Explicit uniform-grid JSON reference import; no silent interpolation/extrapolation."""
from pathlib import Path
from dataclasses import asdict
import numpy as np
from pdeflow.core import ProblemSpec
from pdeflow.runtime import atomic_write,write_json,digest,file_digest,code_identity
from pdeflow.local import now

def import_reference(payload,output,identifier):
    required={'problem','coordinates','physical_times','fields','values','units','provenance'}
    if set(payload)!=required:raise ValueError('Reference JSON requires exactly problem, coordinates, physical_times, fields, values, units, provenance')
    problem=ProblemSpec.from_dict(payload['problem']);g=problem.grid
    if payload['units']!='nondimensional' or payload['fields']!=[f.name for f in problem.fields]:raise ValueError('Imported field order/units must match the physical problem; only nondimensional fields supported')
    if g.dimension>3 or np.prod(g.shape)*len(g.times)*len(problem.fields)>2_000_000:raise ValueError('Imported field budget exceeded')
    from pdeflow.specialists import get_specialist
    get_specialist(problem.family).validate(problem,'independent_gaussian_linear_cfm','heun','uniform')
    expected=dict(zip(g.axis_names,[c.numpy() for c in g.coordinates()]))
    if set(payload['coordinates'])!=set(expected):raise ValueError('Coordinate axis names/order must match the problem')
    for name,want in expected.items():
        actual=np.asarray(payload['coordinates'][name],dtype=float)
        if actual.shape!=want.shape or not np.isfinite(actual).all() or not np.all(np.diff(actual)>0) or not np.allclose(actual,want,rtol=1e-10,atol=1e-10):raise ValueError(f'{name}: require unique increasing exact uniform-grid coordinates; resample explicitly before import')
    times=payload['physical_times']
    if times!=(list(g.times) if g.temporal_kind=='transient' else None):raise ValueError('Imported physical times must exactly match the problem; static has null physical_times')
    values=np.asarray(payload['values'],dtype=np.float64)
    if values.shape!=(len(problem.fields),len(g.times),*g.shape) or not np.isfinite(values).all():raise ValueError('Values require finite [field,physical-time,*spatial] layout; no axis inference/transposition')
    directory=Path(output);directory.mkdir(parents=True,exist_ok=False)
    spec={'request':{'problem':problem.to_dict()},'inputs':{'imported':True},'solver':'imported','provenance':payload['provenance']}
    write_json(directory/'problem.json',problem.to_dict());write_json(directory/'reference-problem.json',spec)
    atomic_write(directory/'reference.npz',lambda f:np.savez_compressed(f,values=values[None]))
    manifest={'schema_version':1,'kind':'reference_solution','id':identifier,'created_at':now(),'source':code_identity(),'specification_identity':digest(spec),'problem':problem.to_dict(),'coordinates':payload['coordinates'],'physical_times':times,'fields':[asdict(f) for f in problem.fields],'array_sha256':file_digest(directory/'reference.npz'),'diagnostics':{'classification':'imported numerical/reference field; PDE validity not established by import','solver':'external','provenance':payload['provenance'],'interpolation':'none; exact provided uniform grid only','reference_only':True},'seed':None}
    write_json(directory/'reference.json',manifest)
    return {**manifest,'path':str(directory.resolve())}

def export_reference(path):
    from pdeflow.reference import load
    p,y,m=load(path)
    return {'problem':p.to_dict(),'coordinates':m['coordinates'],'physical_times':m['physical_times'],'fields':[f.name for f in p.fields],'values':y[0].tolist(),'units':'nondimensional','provenance':{'reference_id':m['id'],'diagnostics':m['diagnostics'],'source':m['source']}}
