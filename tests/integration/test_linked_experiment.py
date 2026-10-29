from dataclasses import asdict
from pathlib import Path
import uuid
import pytest,numpy as np
from fastapi.testclient import TestClient
from pdeflow.workbench.api import create_app
from pdeflow.reference.examples import example
from pdeflow.runtime import file_digest,read_json
from test_workbench import wait

@pytest.mark.parametrize('family,dimension,root',[('heat',2,'outputs/accepted_cpu_2d'),('darcy',3,'outputs/multispecialist-20261021/cpu-3d'),('chemistry',3,'outputs/online-chemistry-20261021/chemistry-3d')])
def test_real_auto_selection_reference_context_snapshots_local_terms(tmp_path,family,dimension,root):
 with TestClient(create_app(model_roots=[Path(root)/'training'],workspace=tmp_path)) as c:
  ref=c.post('/api/references',json=asdict(example(f'{family}-{dimension}d')));assert ref.status_code==201,ref.text
  rid=ref.json()['id'];v=c.post(f'/api/references/{rid}/sample',json={'sampling':{'count':3}}).json()
  payload={'model_selection':{'mode':'auto'},'problem':v['problem'],'rows':v['rows'],'noise_std':.03,'mathematical_request':v['mathematical_request'],'inference':{'steps':3,'particles':2,'snapshots':{'field_indices':list(range(len(v['problem']['fields']))),'time_indices':[0]}}}
  j=c.post('/api/jobs',json=payload);assert j.status_code==202,j.text
  jid=j.json()['id'];context=c.post(f'/api/jobs/{jid}/evaluation-context',json={'reference_id':rid,'sampling_id':v['sampling_id']});assert context.status_code==200,context.text
  assert wait(c,jid)['status']=='succeeded'
  request=read_json(tmp_path/'workbench/jobs'/jid/'request.json');assert request['model_id'] and 'reference_id' not in request and 'inputs' not in request and 'reference' not in request and request['selection']['mode']=='auto'
  r=c.get(f'/api/jobs/{jid}/result');assert r.status_code==200 and r.json()['reference_available'],r.text
  assert 'evaluation' in r.json()['metrics']
  h=file_digest(tmp_path/'workbench/jobs'/jid/'result/result.npz');assert c.get(f'/api/jobs/{jid}/result').json()['reference_available'];assert file_digest(tmp_path/'workbench/jobs'/jid/'result/result.npz')==h
  inspect=c.get(f'/api/jobs/{jid}/inspection');assert inspect.status_code==200,inspect.text
  assert len(inspect.json()['views'])==(3 if dimension==3 else 1) and inspect.json()['reference_value'] is not None
  frames=c.get(f'/api/jobs/{jid}/generation');assert frames.status_code==200,frames.text
  assert frames.json()['frames'][0]['generation_position']==0 and frames.json()['frames'][-1]['generation_position']==1
  endpoint=c.get(f'/api/jobs/{jid}/generation?frame=3&kind=endpoint_physical');assert endpoint.status_code==200,endpoint.text
  assert c.post(f'/api/jobs/{jid}/reference',json={'reference_id':None}).status_code==200
  assert not c.get(f'/api/jobs/{jid}/result').json()['reference_available']
  assert c.get(f'/api/jobs/{jid}/generation').status_code==200 # re-export preserves actual recorded history
