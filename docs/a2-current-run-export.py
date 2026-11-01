"""Read-only diagnostic export; this function also runs in an OLD active kernel.

Only stdlib + the already-installed torch are used. No installation, source edit,
model forward, synchronization request, optimizer operation or Drive write.
"""
from pathlib import Path
import json,hashlib,tempfile,time,zipfile,platform
import torch


def export_a2_diagnostics(run_root,destination,include_weights=True,max_weight_mib=512,persistent_root=None):
    run_root=Path(run_root);destination=Path(destination)
    if destination.exists():raise FileExistsError('Choose a new diagnostic ZIP; no run file is overwritten')
    destination.parent.mkdir(parents=True,exist_ok=True)
    files={};notes=[];cp_path=run_root/'evolution/resume.pt'
    with tempfile.TemporaryDirectory(prefix='pdeflow-a2-export-') as d:
        temporary=Path(d)
        for relative in ('session.json','resolved-config.json','source-input.json','source/identity.json','source/source.zip','representation-quality.json','representation-validation.json','representation-validation.npz','evolution-validation.json','evolution-validation.npz','sync-status.json','autoencoder/run.json','evolution/run.json','evolution/config.json','evolution/history.json','evolution/history-live.jsonl','evolution/evolution-preview.json','evolution/evolution-preview.npz'):
            p=run_root/relative
            if p.is_file():
                if p.stat().st_size>100*1024**2:notes.append('Metadata file exceeds explicit100MiB export cap: '+relative);continue
                data=p.read_bytes();target=temporary/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data);files[relative]=target
        checkpoint_info=None
        if cp_path.exists():
            # Trainer replaces checkpoints atomically. mmap keeps the opened file
            # inode stable while only selected state pages are serialized.
            cp=torch.load(cp_path,map_location='cpu',weights_only=True,mmap=True)
            checkpoint_info={k:cp.get(k) for k in ('stage','progress','attempted_updates','skipped_updates','actual_precision','field_loss','resolved_field_loss','dependency','config','scientific_source','dataset_identity')}
            checkpoint_info['scheduler']=cp.get('scheduler');checkpoint_info['history']=cp.get('history',[])
            weights=cp.get('autoencoder')
            if include_weights:
                if not weights or not any(k.startswith('encoder.') for k in weights):raise ValueError('A2 snapshot lacks full matching state encoder; no substitute was made')
                size=sum(v.numel()*v.element_size() for v in weights.values())
                if size>max_weight_mib*1024**2:raise ValueError(f'A2 weights use {size/1024**2:.1f}MiB, exceeding the explicit cap; set include_weights=False or deliberately increase max_weight_mib')
                diagnostic={k:v for k,v in cp.items() if k not in ('optimizer','scheduler','scaler','rng','torch_rng','cuda_rng','ambient_rng')}
                diagnostic['artifact_role']='A2 diagnostic-only snapshot; not resumable training state'
                target=temporary/'a2-diagnostic.pt';torch.save(diagnostic,target);files[target.name]=target
                checkpoint_info['exported_weight_bytes']=size
            checkpoint_info['snapshot_semantics']='Existing complete local checkpoint, possibly older than live run.json; progress here identifies the exported weights. No new checkpoint was requested.'
        else:notes.append('No existing A2 checkpoint yet; metadata only')
        persistent=[]
        if persistent_root:
            for p in (Path(persistent_root)/'snapshots').glob('*.json'):
                try:
                    r=json.loads(p.read_text());persistent.append({'id':r['id'],'stage':r.get('stage'),'successful_update':r.get('successful_update'),'created_unix':r['created_unix'],'manifest':str(p),'verification':'Candidate metadata only; objects were NOT scanned during active training'})
                except (OSError,ValueError,KeyError):continue
        report={'run_id':run_root.name,'created_unix':time.time(),'checkpoint':checkpoint_info,'notes':notes,'persistent_snapshot_candidates':persistent,'environment':{'torch':str(torch.__version__),'python':platform.python_version(),'cuda_runtime':torch.version.cuda,'cuda_available':torch.cuda.is_available()},'safety':'Read-only old-runtime export; no model evaluation/training/source modification. Diagnostic weights omit optimizer/RNG and cannot resume training.'}
        target=temporary/'diagnostic-summary.json';target.write_text(json.dumps(report,indent=2)+'\n');files[target.name]=target
        manifest={'format':1,'files':{n:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for n,p in files.items()}}
        with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_STORED) as archive:
            for name,p in files.items():archive.write(p,name)
            archive.writestr('DIAGNOSTIC_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:raise ValueError('Diagnostic ZIP failed CRC verification')
        for name,e in manifest['files'].items():
            if hashlib.sha256(archive.read(name)).hexdigest()!=e['sha256']:raise ValueError('Diagnostic checksum mismatch: '+name)
    return {'path':str(destination),'sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),'files':len(files),'checkpoint_update':checkpoint_info.get('progress') if checkpoint_info else None,'contains_diagnostic_weights':include_weights and 'a2-diagnostic.pt' in files,'not_a_resume_bundle':True}

# Copy this WHOLE cell into the existing active runtime: no pip/reload needed.
EXPORT_A2_DIAGNOSTICS = False
INCLUDE_A2_WEIGHTS = True  # matching E+T+D, capped512MiB; no optimizer
DOWNLOAD_DIAGNOSTIC_ZIP = False
if EXPORT_A2_DIAGNOSTICS:
    import uuid
    diagnostic_zip = Path(LOCAL_ROOT) / ("a2-diagnostics-" + session.name + "-" + uuid.uuid4().hex[:8] + ".zip")
    diagnostic_export = export_a2_diagnostics(session.root, diagnostic_zip, INCLUDE_A2_WEIGHTS, persistent_root=session.persistent)
    print(json.dumps(diagnostic_export,indent=2))
    if DOWNLOAD_DIAGNOSTIC_ZIP:
        from google.colab import files
        files.download(str(diagnostic_zip))
