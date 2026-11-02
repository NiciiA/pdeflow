"""Audited frozen-physics reuse and explicit NEW-experiment B initialization.

This is deliberately narrower than allowing an arbitrary training source change.
It does not authorize resuming an old optimizer under changed scientific source.
"""
import ast
from pathlib import Path
from zipfile import ZipFile
import hashlib
import torch
from pdeflow.runtime import digest, file_digest


def frozen_source(checkpoint, cp):
    from pdeflow.source_bundle import verify
    candidates=[p/folder/'source.zip' for p in Path(checkpoint).resolve().parents for folder in ('frozen-source','source')]
    archive=None
    for candidate in candidates:
        if not candidate.is_file():continue
        with ZipFile(candidate) as z:
            hits=[n for n in z.namelist() if n.endswith('/src/pdeflow/coordinate/model.py')]
            if len(hits)==1 and hashlib.sha256(z.read(hits[0])).hexdigest()==cp['scientific_source']['files'].get('coordinate/model.py'):
                archive=candidate;break
    if archive is None:raise ValueError('Verified frozen reuse requires the dependency original source/source.zip')
    verify(archive)
    base=Path(__file__).parents[1]
    protected={
        'coordinate/model.py':['derivative_attention','Coordinates','Space','Attention','Block','Geometry','StateEncoder','Coefficient','SpatialDecoder','Evolution','Autoencoder'],
        'coordinate/joint.py':['Status','Group','layout','active_mask'],
    }
    whole=['coordinate/graph.py','coordinate/physics.py','numerics/dynamic.py','coordinate/dataset.py','coordinate/temporal.py','heat_continuous.py','field_losses.py']
    with ZipFile(archive) as z:
        def original(name):
            hits=[n for n in z.namelist() if n.endswith('/src/pdeflow/'+name)]
            if len(hits)!=1:raise ValueError('Ambiguous/missing scientific source '+name)
            data=z.read(hits[0]);expected=cp['scientific_source']['files'].get(name)
            if expected is None or hashlib.sha256(data).hexdigest()!=expected:raise ValueError('Dependency/source hash mismatch: '+name)
            return data
        for name in whole:
            if original(name)!=(base/name).read_bytes():raise ValueError('Frozen dependency implementation changed: '+name)
        for name,nodes in protected.items():
            def signatures(data):return {x.name:ast.dump(x,include_attributes=False) for x in ast.parse(data).body if isinstance(x,(ast.FunctionDef,ast.ClassDef))}
            old=signatures(original(name));new=signatures((base/name).read_bytes())
            for node in nodes:
                if old.get(node)!=new.get(node):raise ValueError('Frozen dependency implementation changed: '+name+':'+node)
    return {'policy':'verified_frozen_v2','archive_sha256':file_digest(archive),'original_scientific_source':cp['scientific_source']['sha256'],'verified_whole_files':whole,'verified_ast_nodes':protected,'scope':'Frozen physical encoder/decoder/evolution and unchanged reference recipe only; not exact old-source resume'}


def initialize_flow(model,c,path,dependency,dependency_cp):
    from .training import read_checkpoint
    from .optimization import representation_contract
    old,prior=read_checkpoint(path,'flow')
    if not old['completed'] or representation_contract(c)!=representation_contract(prior):raise ValueError('Warm start requires completed compatible B representation/data/observation distribution')
    if old['dependency']['sha256']!=file_digest(dependency):raise ValueError('Warm start must retain the exact original A2 dependency')
    if old['dataset_identity']!=dependency_cp['dataset_identity'] or digest(old['normalizations'])!=digest(dependency_cp['normalizations']):raise ValueError('Warm start normalization/dataset mismatch')
    current=model.state_dict();previous=old['velocity'];copied=[];initialized=[]
    if set(previous)-set(current):raise ValueError('Warm start would discard trained modules')
    for name,value in current.items():
        if name in previous:
            if value.shape!=previous[name].shape:raise ValueError('Warm start tensor shape mismatch: '+name)
            current[name]=previous[name];copied.append(name)
        elif name.startswith('sensor_attention.'):
            # A new residual branch starts with zero effect, never an alleged
            # trained architectural upgrade. Its first output gradient is live.
            if name in ('sensor_attention.out.weight','sensor_attention.out.bias'):current[name]=torch.zeros_like(value)
            initialized.append(name)
        else:raise ValueError('Unapproved new warm-start module: '+name)
    model.load_state_dict(current,strict=True)
    return {'kind':'new_experiment_not_resume','parent_sha256':file_digest(path),'parent_updates':old['progress'],'parent_architecture':old['architecture'],'copied_keys':copied,'new_keys':initialized,'new_reader_output':'zero initialization preserves parent initial prediction','optimizer':'fresh; parent optimizer/scheduler/RNG are not restored'}
