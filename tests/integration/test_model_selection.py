from dataclasses import asdict
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from pdeflow.workbench.api import create_app
from pdeflow.reference.examples import example
from pdeflow.reference.solvers import solve
from pdeflow.core import ProblemSpec

def request():
 p,_,_=solve(example('heat-2d'));return {'model_selection':{'mode':'auto'},'problem':p.to_dict(),'rows':[{'field':'u','x':0,'y':0,'time':0,'value':.8}],'inference':{'steps':2,'particles':1}}

def test_empty_registry_selection_contract_and_human_error(tmp_path):
 with TestClient(create_app(model_roots=[],workspace=tmp_path)) as c:
  v=request();r=c.post('/api/reconstruction/validate',json=v);assert r.status_code==422 and 'No compatible trained model' in r.json()['error']['message']
  assert not c.get('/api/jobs').json()
  v.pop('model_selection');v['model_id']='';r=c.post('/api/jobs',json=v)
  assert r.status_code==422 and 'cannot be empty' in r.json()['error']['message']
  assert r.json()['error']['details'][0]['code']=='string_pattern_mismatch'


def test_authoritative_auto_explicit_stale_and_wrong_layout(tmp_path):
 root=Path('outputs/accepted_cpu_2d/training')
 with TestClient(create_app(model_roots=[root],workspace=tmp_path)) as c:
  models=c.get('/api/models').json();assert models
  v=request();r=c.post('/api/reconstruction/validate',json=v);assert r.status_code==200,r.text
  chosen=r.json()['resolved']['model_id'];assert chosen in [m['id'] for m in models]
  assert r.json()['resolved']['selection']['mode']=='auto'
  v['model_selection']={'mode':'explicit','id':chosen};assert c.post('/api/reconstruction/validate',json=v).status_code==200
  v['model_selection']['id']='0'*32;r=c.post('/api/jobs',json=v);assert r.status_code==422 and 'unavailable' in r.json()['error']['message']
  v['model_selection']={'mode':'auto'};v['problem']['grid']['shape']=[16,16];r=c.post('/api/jobs',json=v);assert r.status_code==422 and 'No compatible' in r.json()['error']['message']
