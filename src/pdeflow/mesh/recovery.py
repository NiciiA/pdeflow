"""Explicit audited M1 corrected-source continuation, never an exact-resume override."""
from pathlib import Path
from dataclasses import replace
import shutil,zipfile,hashlib
import torch
from pdeflow.runtime import read_json,write_json,atomic_write,file_digest,digest
from pdeflow.source_bundle import verify as verify_source
from .provenance import read_checkpoint,scientific_source
from .diagnostics import checkpoint_health

# The supported patch changes arithmetic/safety/monitoring, not references, datasets,
# state layout, sampling, objective definitions, optimizer or architecture dimensions.
PATCH_FILES={'mesh/benchmark.py','mesh/state.py','mesh/backbone.py','mesh/model.py','mesh/objectives.py','mesh/training.py','mesh/session.py','mesh/suite.py','mesh/provenance.py','mesh/artifacts.py','mesh/cli.py','mesh/numerical.py','mesh/diagnostics.py','mesh/monitoring.py','mesh/recovery.py'}


def corrected_continuation(parent_root,local_root,persistent_root=None,device=None):
    from .session import MeshSession
    parent=Path(parent_root).resolve();path=parent/'stages/reconstruction/resume.pt';health=checkpoint_health(path)
    if not health['healthy']:raise ValueError('Parent checkpoint is unhealthy; restore an earlier complete healthy snapshot first')
    cp,c=read_checkpoint(path)
    if c.architecture!='mesh_m1_v1':raise ValueError('Audited corrected-source continuation currently supports the affected M1 reconstruction stage only')
    source=parent/'source/source.zip';verify_source(source);identity=read_json(parent/'source/identity.json')
    if file_digest(source)!=identity['sha256'] or cp['scientific_source']['sha256']!=identity['scientific_source']['sha256']:raise ValueError('Parent checkpoint/source identity mismatch')
    current=scientific_source();prior=cp['scientific_source']
    patch=read_json(Path(__file__).parent/'patches/stability_v1.json')
    if prior['sha256']!=patch['parent_scientific_source'] or current['sha256']!=patch['corrected_scientific_source']:raise ValueError('This source transition is not the audited stability_v1 release; do not override hashes')
    with zipfile.ZipFile(source) as archive:
        for name,sha in prior['files'].items():
            if hashlib.sha256(archive.read('pdeflow/src/pdeflow/'+name)).hexdigest()!=sha:raise ValueError('Parent scientific member changed: '+name)
    changes=[name for name in set(prior['files'])|set(current['files']) if prior['files'].get(name)!=current['files'].get(name)]
    if set(changes)-PATCH_FILES:raise ValueError('Unsupported scientific source transition: '+str(sorted(set(changes)-PATCH_FILES)))
    if not changes:raise ValueError('Source is unchanged: use exact resume, not corrected continuation')
    if cp['successful_updates']>=c.budgets['reconstruction']:raise ValueError('Parent budget is complete; continuation requires an explicit separate extension protocol')
    if device:c=replace(c,runtime={**c.runtime,'device':device}).validate()
    # Validate every actual tensor against the corrected implementation before allocation.
    from .model import MeshModel
    with torch.device('meta'):model=MeshModel(c)
    model.load_state_dict(cp['model'],strict=True,assign=True)
    result=MeshSession.new(c,'corrected-source continuation',local_root,persistent_root)
    original=result.root/'continuation';original.mkdir();shutil.copy2(path,original/'parent.pt');shutil.copy2(source,original/'parent-source.zip');shutil.copy2(parent/'source/identity.json',original/'parent-source-identity.json')
    (result.root/'data').mkdir();shutil.copy2(parent/'data/dataset.json',result.root/'data/dataset.json')
    lineage={'kind':'corrected_source_continuation_v1','not_exact_resume':True,'patch_manifest_sha256':file_digest(Path(__file__).parent/'patches/stability_v1.json'),'parent_root':str(parent),'parent_checkpoint_sha256':health['sha256'],'parent_successful_updates':cp['successful_updates'],'parent_scientific_source':prior,'new_scientific_source':current,'changed_files':sorted(changes),'inherited':['model','optimizer','scheduler','scaler','generator','ambient_rng','torch_rng','cuda_rng','normalizations','progress'],'config_policy':'Identical scientific settings; optional explicit runtime device remap only','arithmetic_change':'Explicit FP32 mesh attention/kernel products and loss reductions; scientific objective unchanged; exact cross-source trajectory equality not promised','parent_health':health}
    write_json(original/'lineage.json',lineage);result._source()
    # This is a NEW checkpoint with full source lineage. Original bytes stay immutable.
    child={**cp,'config':c.to_dict(),'config_identity':digest(c.to_dict()),'scientific_source':current,'source_transition':lineage,'completed':False}
    stage=result.root/'stages/reconstruction';stage.mkdir(parents=True);atomic_write(stage/'resume.pt',lambda f:torch.save(child,f))
    for name in ('history.json','history-live.jsonl'):
        if (path.parent/name).exists():shutil.copy2(path.parent/name,stage/name)
    write_json(result.root/'resolved-config.json',c.to_dict());write_json(result.root/'session.json',{'schema':'mesh_session_v1','config':c.to_dict(),'initial_config':c.to_dict(),'scientific_source':current,'state':'corrected continuation ready','progress':cp['successful_updates'],'stage':'reconstruction','stages':{},'persistent':str(result.persistent) if result.persistent else None,'source_transition':lineage})
    assert checkpoint_health(stage/'resume.pt')['healthy'];result.sync('corrected-continuation-ready');return result
