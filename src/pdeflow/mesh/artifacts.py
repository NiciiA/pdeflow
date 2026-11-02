"""Standalone mesh inference, complete diagnostics and resumable manifests."""
from pathlib import Path
import zipfile
import torch
from pdeflow.runtime import write_json,read_json,file_digest,atomic_write,digest
from .contracts import SCHEMA
from .provenance import read_checkpoint


def export(root):
    root=Path(root);session=read_json(root/'session.json');c=__import__('pdeflow.mesh.contracts',fromlist=['MeshConfig']).MeshConfig(**session['config']).validate();path=root/'stages'/c.kind['stages'][-1]/'resume.pt'
    if not path.exists():return None
    cp,_=read_checkpoint(path)
    if not cp['inference_ready']:return None
    provenance={}
    for stage in c.kind['stages']:
        prior=root/'stages'/stage/'resume.pt'
        if prior.exists():
            p,_=read_checkpoint(prior);provenance[stage]={'checkpoint_sha256':file_digest(prior),'successful_updates':p['successful_updates'],'attempted_updates':p['attempted_updates'],'skipped_updates':p['skipped_updates'],'effective_examples':p['effective_examples'],'training_seconds':p['training_seconds'],'scope':p['training_only_information']}
    out={k:v for k,v in cp.items() if k not in ('optimizer','scaler','scheduler','generator','ambient_rng','torch_rng','cuda_rng')};out.update(model_partial_frozen_modules=[],dependency=None,dataset_root=None,standalone=True,training_provenance=provenance,training_status='completed budget; scientifically unevaluated' if cp['completed'] else 'explicitly exported partial training')
    target=root/'export/model.pt';target.parent.mkdir(exist_ok=True);atomic_write(target,lambda f:torch.save(out,f));write_json(target.with_suffix('.json'),{'architecture':c.architecture,'sha256':file_digest(target),'capabilities':out['capabilities'],'training_provenance':provenance,'accuracy':'Unestablished'})
    return target


def verify(root,kind):
    root=Path(root).resolve();record=read_json(root/'bundles'/(kind+'.json'))
    if record['schema']!=SCHEMA or record['kind']!=kind:raise ValueError('Mesh artifact schema mismatch')
    for name,entry in record['files'].items():
        path=root/name
        if Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink() or not path.resolve().is_relative_to(root):raise ValueError('Unsafe artifact path')
        if not path.is_file() or path.stat().st_size!=entry['bytes'] or file_digest(path)!=entry['sha256']:raise ValueError('Mesh artifact checksum mismatch: '+name)
    return record


def build(root):
    root=Path(root);source=root/'source/source.zip'
    from pdeflow.source_bundle import verify as verify_source
    verify_source(source)
    if file_digest(source)!=read_json(root/'source/identity.json')['sha256']:raise ValueError('Source identity mismatch')
    files={root/'session.json',root/'resolved-config.json',source,root/'source/identity.json',root/'data/dataset.json'}
    stages={}
    for path in (root/'stages').glob('*/resume.pt'):
        cp,c=read_checkpoint(path)
        for key in ('optimizer','scheduler','scaler','generator','ambient_rng','torch_rng','cuda_rng'):
            if key not in cp:raise ValueError('Incomplete mesh resume state: '+key)
        from .model import MeshModel
        with torch.device('meta'):m=MeshModel(c)
        m.load_state_dict(cp['model'],strict=True,assign=True)
        if cp.get('dependency'):
            target=(path.parent/cp['dependency']['path']).resolve()
            if not target.is_relative_to(root.resolve()) or file_digest(target)!=cp['dependency']['sha256']:raise ValueError('Frozen stage outside complete run or changed')
        stages[path.parent.name]={'successful_updates':cp['successful_updates'],'attempted_updates':cp['attempted_updates'],'completed':cp['completed'],'sha256':file_digest(path)}
        files.update(p for p in path.parent.iterdir() if p.is_file() and p.suffix in ('.pt','.json','.npz','.jsonl'))
    for p in (root/'stages').rglob('*'):
        if p.is_file() and any(part in ('failure','healthy') for part in p.relative_to(root).parts) and p.suffix in ('.pt','.json'):files.add(p)
    for p in (root/'continuation').glob('*'):
        if p.is_file() and p.suffix in ('.pt','.json','.zip'):files.add(p)
    for p in (root/'bootstrap').glob('*'):
        if p.is_file() and p.suffix in ('.pt','.json'):files.add(p)
    inference={root/'export/model.pt',root/'export/model.json'} if (root/'export/model.pt').exists() else set()
    if inference:
        from .inference import MeshBundle
        MeshBundle(root/'export/model.pt','cpu')
    result={}
    for kind,paths in [('inference',inference),('diagnostic',files|inference),('resumable',files|inference)]:
        record={'schema':SCHEMA,'kind':kind,'available':bool(inference) if kind=='inference' else True,'stages':stages,'architecture':read_json(root/'session.json')['config']['architecture'],'files':{p.relative_to(root).as_posix():{'bytes':p.stat().st_size,'sha256':file_digest(p)} for p in sorted(paths)},'diagnostic_completeness':'All modules this mesh model actually uses; no A1 encoder exists or is required','accuracy':'Not established by smoke execution'}
        write_json(root/'bundles'/(kind+'.json'),record);result[kind]=verify(root,kind)
    return result


def archive(root,kind):
    root=Path(root);record=verify(root,kind)
    if not record['available']:raise ValueError('Artifact unavailable before learned final stage')
    target=root/'downloads'/(kind+'-'+digest(record)[:16]+'.zip');target.parent.mkdir(exist_ok=True)
    if not target.exists():
        with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_STORED) as z:
            for name in record['files']:z.write(root/name,name)
            z.write(root/'bundles'/(kind+'.json'),'bundles/'+kind+'.json')
    with zipfile.ZipFile(target) as z:
        if z.testzip():raise ValueError('Artifact ZIP CRC failure')
    return target
