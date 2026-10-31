"""Immutable registered spectral dynamic reference artifacts for the playground."""
from pathlib import Path
import uuid
import numpy as np
from pdeflow.runtime import atomic_write,write_json,digest,file_digest,code_identity
from pdeflow.local import now

def save(problem,solver,output,spec):
    output=Path(output)
    if output.exists():raise FileExistsError('Reference artifact already exists')
    output.mkdir(parents=True);key=next(iter(problem.known_parameters));parameter=problem.known_parameters[key];names=[f.name for f in problem.fields]
    values=np.stack([np.stack([solver.frame(t,parameter,(0.,problem.grid.times[-1]),quantity=q) for t in problem.grid.times]) for q in names])
    specification={'request':problem.to_dict(),'inputs':{'initial_u':solver.initial.tolist(),**({'initial_v':solver.velocity.tolist()} if problem.family=='wave' else {})},'solver':'periodic_spectral_'+problem.family+'_v2','configuration':spec}
    write_json(output/'problem.json',problem.to_dict());write_json(output/'reference-problem.json',specification);atomic_write(output/'reference.npz',lambda f:np.savez_compressed(f,values=values[None]))
    invariant=[{'time':float(t),**solver.invariants(t,parameter)} for t in problem.grid.times]
    manifest={'schema_version':1,'kind':'reference_solution','id':output.name,'created_at':now(),'source':code_identity(),'specification_identity':digest(specification),'problem':problem.to_dict(),'physical_times':list(problem.grid.times),'array_sha256':file_digest(output/'reference.npz'),'diagnostics':{'solver':'periodic_spectral_'+problem.family+'_v2','classification':'finite periodic Fourier numerical reference; analytic mode evolution, not exact arbitrary drawing','reference_only':True,'invariants':invariant,'zero_mode':'wave mean displacement is linear in t for nonzero mean v0','nyquist':'symmetric split cosine interpolant','temporal_display_only':True}}
    write_json(output/'reference.json',manifest);return values,manifest
