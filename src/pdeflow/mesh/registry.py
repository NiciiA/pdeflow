"""Explicit mesh imports/discovery; manifests list capabilities without loading weights."""
from pathlib import Path
from pdeflow.runtime import read_json,write_json,file_digest
from pdeflow.local import workspace_root
from .contracts import MeshConfig,capabilities
from .provenance import read_checkpoint


def register_checkpoint(path,workspace=None):
    path=Path(path).expanduser().resolve();cp,c=read_checkpoint(path)
    if not cp.get('inference_ready'):raise ValueError('Choose an explicitly exported trained final mesh stage, not a dynamics/bootstrap checkpoint')
    from .inference import MeshBundle
    MeshBundle(path,'cpu')
    sha=file_digest(path);record={'id':sha[:32],'model_id':sha[:32],'architecture':c.architecture,'name':c.kind['name'],'export':str(path),'sha256':sha,'config':c.to_dict(),'capabilities':capabilities(c),'progress':cp['successful_updates'],'state':'completed' if cp['completed'] else 'partial','available':True,'pipeline_check':c.pipeline_check,'training_provenance':cp.get('training_provenance',{}),'accuracy':'Scientifically unevaluated'}
    target=workspace_root(workspace)/'mesh-models'/(sha[:32]+'.json');target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() and read_json(target)['sha256']!=sha:raise ValueError('Registered identity conflict')
    write_json(target,record);return record


def runs(workspace=None):
    from pdeflow.training_workflow import known_workspaces
    found={}
    for root in known_workspaces(workspace):
        for path in (root/'mesh-models').glob('*.json'):
            row=read_json(path);target=Path(row['export']);row={**row,'available':target.is_file()}
            if row['available'] and file_digest(target)!=row['sha256']:row.update(available=False,error='Registered export changed; reimport explicitly')
            found[row['id']]=row
    return list(found.values())


def model(workspace,identifier):
    for row in runs(workspace):
        if row['id']==identifier and row['available']:return row
    raise ValueError('Mesh checkpoint unavailable; explicitly import its standalone export')
