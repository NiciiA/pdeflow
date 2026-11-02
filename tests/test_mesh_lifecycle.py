from pathlib import Path
from dataclasses import replace
import pytest
import torch
from pdeflow.mesh.contracts import preset,CANDIDATES
from pdeflow.mesh.session import MeshSession
from pdeflow.mesh.provenance import read_checkpoint
from pdeflow.mesh.inference import MeshBundle
from pdeflow.mesh.artifacts import verify,archive

@pytest.mark.parametrize('name',CANDIDATES)
def test_real_stage_train_export_snapshot_and_resume(tmp_path,name):
    torch.set_num_threads(2);c=preset('pipeline-check',name,'cpu','fp32',{'data':{'query_points':12},'checkpoint_every':2,'validation_every':2});s=MeshSession.new(c,name,tmp_path/'local',tmp_path/'persistent',save_updates=2)
    first=s.run(stop_after=1);assert first['state']=='interrupted'
    record=s.run();assert record['state']=='completed'
    result=s.finalize();assert result['export'];bundle=MeshBundle(result['export'],'cpu');assert bundle.cp['inference_ready'] and all(p.grad is None for p in bundle.model.parameters())
    for kind in ('inference','diagnostic','resumable'):assert verify(s.root,kind)['available'] and archive(s.root,kind).is_file()
    restored=MeshSession.from_store(tmp_path/'persistent',s.root.name,tmp_path/'restored',device='cpu');assert restored.run()['state']=='completed'
    for stage in c.kind['stages']:
        cp,_=read_checkpoint(s.root/'stages'/stage/'resume.pt');assert cp['successful_updates']==2 and cp['optimizer'] and cp['scheduler'] and cp['ambient_rng'] and cp['torch_rng'] is not None
    changed=replace(c,field_loss={'type':'mae'}).validate()
    with pytest.raises(ValueError,match='Incompatible'):MeshSession(changed,s.root).run()

def test_json_roundtrip_tuple_data_resume(tmp_path):
    c=preset('pipeline-check','mesh_m1_v1','cpu','fp32',{'data':{'reference_shape':(16,16),'observation_counts':(1,8),'query_points':8}});s=MeshSession(c,tmp_path/'tuple-run');assert s.run(stop_after=1)['state']=='interrupted';assert s.run()['state']=='completed'
