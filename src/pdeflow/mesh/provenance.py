"""Scientific fingerprint and strict mesh checkpoint validation."""
from pathlib import Path
import hashlib
import torch
from pdeflow.runtime import digest,file_digest
from .contracts import MeshConfig,SCHEMA,CANDIDATES


def scientific_source():
    base=Path(__file__).parents[1];names=[p.relative_to(base).as_posix() for p in (base/'mesh').glob('*.py')]+['coordinate/contracts.py','coordinate/physics.py','coordinate/conditioning.py','coordinate/training.py','coordinate/evolution_objective.py','coordinate/evolution_monitor.py','coordinate/dataset.py','coordinate/temporal.py','coordinate/engine.py','coordinate/graph.py','coordinate/model.py','coordinate/optimization.py','field_losses.py','numerics/dynamic.py','observation_times.py','heat_placement.py']
    files={n:hashlib.sha256((base/n).read_bytes()).hexdigest() for n in sorted(names)}
    return {'sha256':digest(files),'files':files,'scope':'Mesh implementation plus reused reference, sensor, numerical/device and graph definitions'}


def read_checkpoint(path):
    cp=torch.load(path,map_location='cpu',weights_only=True)
    if cp.get('schema')!=SCHEMA or cp.get('architecture') not in CANDIDATES:raise ValueError('Not a supported mesh physical-state checkpoint; legacy uses its original loader')
    c=MeshConfig(**cp['config']).validate()
    if cp['architecture']!=c.architecture or cp['config_identity']!=digest(cp['config']):raise ValueError('Mesh checkpoint configuration/architecture mismatch')
    if cp.get('state_contract')!='normalized_physical_nodes_u0_v0_log_parameter_v1':raise ValueError('Unsupported physical-state tensor contract')
    training=set(cp['training_ids'])
    if not set(sum(cp['normalization_fitted_ids'].values(),[]))<=training:raise ValueError('Normalization leaks nontraining identities')
    if cp['attempted_updates']!=cp['successful_updates']+cp['skipped_updates']:raise ValueError('Mesh optimizer counter semantics mismatch')
    if cp.get('model_partial_frozen_modules'):
        dependency=cp.get('dependency')
        if not dependency:raise ValueError('Partial model is missing its frozen prerequisite')
        target=(Path(path).parent/dependency['path']).resolve()
        if file_digest(target)!=dependency['sha256']:raise ValueError('Frozen mesh prerequisite changed')
        parent,_=read_checkpoint(target);frozen=set(cp['model_partial_frozen_modules']);restored={k:v for k,v in parent['model'].items() if k.split('.')[0] in frozen};restored.update(cp['model']);cp={**cp,'model':restored}
    return cp,c


_COUNTS={}

def counts(c):
    key=digest({k:c.to_dict()[k] for k in ("architecture","width","depth","heads","node_shape","coarse_shape","global_tokens","neighbors","noise_dim","local_noise_dim")})
    if key in _COUNTS:return dict(_COUNTS[key])
    from .model import MeshModel
    with torch.device('meta'):m=MeshModel(c)
    result={**{name:sum(p.numel() for p in module.parameters()) for name,module in m.named_children()},'total':sum(p.numel() for p in m.parameters()),'nodes':c.nodes,'method':'Exact parameter numel; activations and fit require measurement'}
    if len(_COUNTS)>=64:_COUNTS.pop(next(iter(_COUNTS)))
    _COUNTS[key]=result;return dict(result)
