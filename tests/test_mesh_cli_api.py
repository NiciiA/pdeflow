import json,subprocess,sys
from fastapi.testclient import TestClient
from pdeflow.workbench.api import create_app

def test_public_cli_mesh_and_managed_defaults(tmp_path):
    for args in (['mesh','plan','--candidate','M2','--capacity','pipeline-check','--device','cpu'],['training','start','--budget','pipeline-check','--updates','2','--device','cpu','--plan']):
        result=subprocess.run([sys.executable,'-m','pdeflow.cli',*args],text=True,capture_output=True,check=True);record=json.loads(result.stdout)
        assert record.get('architecture',record.get('model')) in ('mesh_m2_v1','mesh_m1_v1')

def test_api_defaults_and_gate_override_rejection(tmp_path):
    with TestClient(create_app(workspace=tmp_path)) as client:
        r=client.post('/api/training/plan',json={'device':'cpu'});assert r.status_code==200,r.text;assert r.json()['model']=='mesh_m1_v1' and r.json()['config']['field_loss']['type']=='mse'
        r=client.post('/api/training/plan',json={'device':'cpu','allow_poor_representation':True});assert r.status_code==422 and 'bypass' in r.text

def test_production_inference_does_not_read_reference_or_dataset(tmp_path,monkeypatch):
    import numpy as np
    from pdeflow.mesh.contracts import preset
    from pdeflow.mesh.session import MeshSession
    from pdeflow.mesh.evaluation import create_cases,load_case
    from pdeflow.mesh.inference import MeshBundle
    from pdeflow.coordinate.dataset import Dataset
    from pdeflow.numerics.dynamic import SpectralState
    c=preset('pipeline-check','mesh_m2_v1','cpu','fp32',{'family_weights':{'heat':1.},'data':{'query_points':8}});s=MeshSession(c,tmp_path/'run');s.run();create_cases(c,tmp_path/'cases',count=1);case,p,*_=load_case(tmp_path/'cases/case-000')
    def forbidden(*a,**kw):raise AssertionError('Hidden reference was read by production inference')
    monkeypatch.setattr(Dataset,'get',forbidden);monkeypatch.setattr(SpectralState,'evaluate',forbidden)
    bundle=MeshBundle(s.root/'export/model.pt');a,m=bundle.sample(p,case['observations'],[[.13,.29,.237],[.81,.63,.83]])
    b,_=bundle.sample(p,list(reversed(case['observations'])),[[.13,.29,.237],[.81,.63,.83]])
    np.testing.assert_allclose(a,b,atol=3e-6);assert m['no_reference_input']
