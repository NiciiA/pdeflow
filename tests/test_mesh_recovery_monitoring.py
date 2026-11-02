from pathlib import Path
import json,os,sys,subprocess,zipfile
import torch
import pytest
from pdeflow.mesh.session import MeshSession
from pdeflow.mesh.recovery import corrected_continuation
from pdeflow.runtime import file_digest,read_json


def test_original_source_continuation_preserves_full_training_state(tmp_path):
    archive=Path('outputs/mesh-family-20261102/pdeflow-mesh-family-20261102.zip').resolve()
    if not archive.exists():pytest.skip('Original matching source archive is a preserved local fixture')
    with zipfile.ZipFile(archive) as z:z.extractall(tmp_path/'original-source')
    source=(tmp_path/'original-source/pdeflow').resolve();parent=(tmp_path/'parent').resolve();baseline=(tmp_path/'baseline').resolve();env={**os.environ,'PYTHONPATH':str(source/'src')}
    script="""from pdeflow.mesh.contracts import preset
from pdeflow.mesh.session import MeshSession
from pdeflow.mesh.provenance import scientific_source
assert scientific_source()['sha256']=='4439ad180a5653dcf5af057c94c2d78112c282e6575a94f745ed1176dd6713cd'
c=preset('pipeline-check','mesh_m1_v1','cpu','fp32',{'data':{'query_points':8},'budgets':{s:3 for s in preset().budgets}})
MeshSession(c,PARENT).run(stop_after=1)
MeshSession(c,BASELINE).run()
""".replace('PARENT',repr(str(parent))).replace('BASELINE',repr(str(baseline)))
    subprocess.run([sys.executable,'-c',script],env=env,cwd=source,check=True,capture_output=True,text=True,timeout=60)
    from pdeflow import persistence
    from pdeflow.mesh.monitoring import status
    persistence.publish(parent,tmp_path/'old-layout-store');old_status=status(tmp_path/'old-layout-store');assert old_status['successful_updates']==1 and old_status['resolved_precision']=='fp32' and old_status['failed_attempts'] is None
    cp=parent/'stages/reconstruction/resume.pt';sha=file_digest(cp);child=corrected_continuation(parent,tmp_path/'children',tmp_path/'persistent',device='cpu');inherited=torch.load(child.root/'stages/reconstruction/resume.pt',weights_only=True);original=torch.load(cp,weights_only=True)
    assert inherited['successful_updates']==1 and inherited['optimizer']['state'] and inherited['source_transition']['not_exact_resume'];assert torch.equal(inherited['generator'],original['generator'])
    for name,value in original['model'].items():torch.testing.assert_close(inherited['model'][name],value,rtol=0,atol=0)
    assert child.run()['state']=='completed' and file_digest(cp)==sha
    result=torch.load(child.root/'stages/reconstruction/resume.pt',weights_only=True);expected=torch.load(baseline/'stages/reconstruction/resume.pt',weights_only=True)
    for name,value in expected['model'].items():torch.testing.assert_close(result['model'][name],value,rtol=0,atol=0)
    # Read-only collector exercises real object-store manifests; does not alter parent/child.
    collector=Path('scripts/collect_mesh_failure.py').resolve();out=tmp_path/'collection.zip';subprocess.run([sys.executable,str(collector),'--local-run',str(child.root),'--store',str(child.persistent),'--output',str(out)],check=True,capture_output=True,text=True,timeout=60)
    with zipfile.ZipFile(out) as z:r=json.loads(z.read('collection.json'));assert r['highest_healthy_observed']['successful_updates']==3 and any(s['verified'] for s in r['snapshots'])
    assert file_digest(cp)==sha
    replay=Path('scripts/replay_mesh_checkpoint.py').resolve();subprocess.run([sys.executable,str(replay),'--source-root',str(source),'--checkpoint',str(cp),'--dataset',str(parent/'data/dataset.json'),'--output',str(tmp_path/'replay'),'--device','cpu'],env=env,cwd=source,check=True,capture_output=True,text=True,timeout=60)
    r=read_json(tmp_path/'replay/report.json');assert r['source_sha256']=='4439ad180a5653dcf5af057c94c2d78112c282e6575a94f745ed1176dd6713cd';assert all(v['loss']['nonfinite']==0 and not v['invalid_gradients'] for v in r['comparisons'].values())
