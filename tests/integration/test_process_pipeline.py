from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from pdeflow.core import ProblemSpec
from pdeflow.observations import ObservationSet
from pdeflow.runtime import read_json,RuntimeSpec,write_json
from pdeflow.training import load_inference
from pdeflow.sampling import sample
from pdeflow.evaluation import export_result
from pdeflow.workbench.observations import observation_rows
from pdeflow.workbench.api import create_app

ROOT=Path('outputs')
CASES=[('heat',ROOT/'accepted_cpu_2d','data/test_problem.json','data/test_observations.json'),('darcy',ROOT/'multispecialist-20261021/cpu-3d','data/test_problem.json','data/test_observations.json'),('chemistry',ROOT/'online-chemistry-20261021/chemistry-3d','evaluation/test/0/problem.json','evaluation/test/0/observations.json')]

@pytest.mark.parametrize('family,root,problem_path,obs_path',CASES)
def test_real_process_export_and_api_agreement_reference_isolation(family,root,problem_path,obs_path,tmp_path):
 if not (root/problem_path).exists():
  from pdeflow.runtime import Config
  from pdeflow.data.providers import prepare
  from pdeflow.training import train
  dimension=2 if family=='heat' else 3
  c=Config.load(f'configs/experiments/online_{family}_{dimension}d.json')
  c=replace(c,provider=replace(c.provider,calibration_count=2,validation_count=1,test_count=1),model=replace(c.model,width=4,blocks=1),training=replace(c.training,steps=2))
  root=tmp_path/'portable-run';prepare(c,root/'evaluation');train(c,None,root/'training',evaluation_path=root/'evaluation')
  problem_path='evaluation/test/0/problem.json';obs_path='evaluation/test/0/observations.json'
 b=load_inference(root/'training/inference.pt',RuntimeSpec())
 p=ProblemSpec.from_dict(read_json(root/problem_path));o=ObservationSet.from_dict(read_json(root/obs_path),p.grid)
 inf=replace(b.config.inference,steps=3,particles=1,controller={'check_interval':1,'max_attempts':1})
 result=sample(b,p,o,inf,'observation_pde')
 assert result.stop_reason=='FINISH_BUDGET_EXHAUSTED' and result.diagnostics['process']['consumed']['sampling_steps']==3
 before=result.samples.clone();trace=json.dumps(result.diagnostics['process'],sort_keys=True)
 export_result(tmp_path/'none',result,p,o)
 export_result(tmp_path/'truth',result,p,o,(torch.cat((torch.ones_like(result.samples[:,0,:1]),torch.zeros_like(result.samples[:,0,1:])),dim=1) if family=="darcy" else torch.ones_like(result.samples[:,0])))
 a=read_json(tmp_path/'none/metrics.json');z=read_json(tmp_path/'truth/metrics.json')
 assert a['process']==z['process'] and torch.equal(before,result.samples) and trace==json.dumps(z['process'],sort_keys=True)
 assert 'fields' not in a and 'fields' in z
 with np.load(tmp_path/'none/result.npz') as array:assert 'final_model_state' in array
 # Execute the API's exact resolved request through the same worker function.
 app=create_app(model_roots=[root/'training'],output_root=tmp_path/'jobs',data_root=tmp_path/'data')
 with TestClient(app) as client:
  model=client.get('/api/models').json()[0]
  payload={'model_id':model['id'],'problem':p.to_dict(),'rows':observation_rows(o,p),'noise_std':o.noise_std,'tolerance':o.noiseless_tolerance,'inference':__import__('dataclasses').asdict(inf),'mode':'observation_pde'}
  resolved=client.post('/api/reconstruction/validate',json=payload)
  assert resolved.status_code==200,resolved.text
  request=resolved.json()['resolved'];resolved_obs=ObservationSet.from_dict(request['observations'],p.grid)
  # CLI-equivalent serialized inputs and API equivalent request agree numerically.
  other=sample(b,ProblemSpec.from_dict(request['problem']),resolved_obs,replace(inf),'observation_pde')
  assert torch.equal(other.samples,result.samples) and other.diagnostics['process']['actions']==result.diagnostics['process']['actions']


def test_actual_cli_and_api_worker_have_identical_controller_fields_and_decisions(tmp_path):
 from pdeflow.cli import reconstruct
 from test_workbench import wait
 root=ROOT/'accepted_cpu_2d'
 if not (root/'data/test_problem.json').exists():
  from pdeflow.runtime import Config
  from pdeflow.data import generate
  from pdeflow.training import train
  c=Config.load('configs/experiments/cpu_2d.json');c=replace(c,data=replace(c.data,count=6),model=replace(c.model,width=4,blocks=1),training=replace(c.training,steps=2))
  root=tmp_path/'portable';generate(c,root/'data');train(c,root/'data',root/'training')
 b=load_inference(root/'training/inference.pt',RuntimeSpec());p=ProblemSpec.from_dict(read_json(root/'data/test_problem.json'));o=ObservationSet.from_dict(read_json(root/'data/test_observations.json'),p.grid)
 inf=replace(b.config.inference,steps=3,particles=1,controller={'check_interval':1})
 c=replace(b.config,runtime=RuntimeSpec(),inference=inf)
 # Reference files are read only after the entire controller and all sample modes.
 np.savez(tmp_path/'unrelated.npz',values=np.ones((1,1,len(p.grid.times),*p.grid.shape)))
 reconstruct(c,root/'training/inference.pt',root/'data/test_problem.json',root/'data/test_observations.json',tmp_path/'cli',tmp_path/'unrelated.npz')
 with TestClient(create_app([root/'training'],tmp_path/'jobs',tmp_path/'references')) as client:
  model=client.get('/api/models').json()[0]
  payload={'model_id':model['id'],'problem':p.to_dict(),'rows':observation_rows(o,p),'noise_std':o.noise_std,'tolerance':o.noiseless_tolerance,'inference':__import__('dataclasses').asdict(inf),'mode':'observation_pde'}
  response=client.post('/api/jobs',json=payload);assert response.status_code==202,response.text
  identifier=response.json()['id'];assert wait(client,identifier)['status']=='succeeded'
  api_result=client.get(f'/api/jobs/{identifier}/result').json()
  from io import BytesIO
  data=client.get(f'/api/jobs/{identifier}/artifacts/samples.npz').content
  with np.load(BytesIO(data)) as a,np.load(tmp_path/'cli/observation_pde/result.npz') as z:assert np.array_equal(a['samples'],z['samples'])
  cli=read_json(tmp_path/'cli/observation_pde/metrics.json')
  assert cli['process']==api_result['metrics']['process']
  assert 'fields' in cli and 'fields' not in api_result['metrics']
