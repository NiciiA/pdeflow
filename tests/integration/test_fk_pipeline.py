from dataclasses import replace,asdict
from pathlib import Path
import json,io
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from pdeflow.runtime import Config,RuntimeSpec,read_json
from pdeflow.core import ProblemSpec
from pdeflow.observations import ObservationSet
from pdeflow.training import load_inference
from pdeflow.sampling import sample
from pdeflow.evaluation import export_result
from pdeflow.workbench.api import create_app
from pdeflow.workbench.observations import observation_rows

CASES=[('heat',2,'outputs/accepted_cpu_2d','data/test_problem.json','data/test_observations.json'),('heat',3,'outputs/accepted_cpu_3d','data/test_problem.json','data/test_observations.json'),('darcy',2,'outputs/multispecialist-20261021/cpu-2d','data/test_problem.json','data/test_observations.json'),('chemistry',3,'outputs/online-chemistry-20261021/chemistry-3d','evaluation/test/0/problem.json','evaluation/test/0/observations.json')]

@pytest.mark.parametrize('family,dim,path,pp,op',CASES)
def test_real_fk_family_reference_isolation_weights_and_serialization(family,dim,path,pp,op,tmp_path):
 path=Path(path)
 if not (path/pp).exists():
  from pdeflow.data.providers import prepare
  from pdeflow.training import train
  c=Config.load(f'configs/experiments/online_{family}_{dim}d.json');c=replace(c,provider=replace(c.provider,calibration_count=2,validation_count=1,test_count=1),model=replace(c.model,width=4,blocks=1),training=replace(c.training,steps=2))
  path=tmp_path/'model';prepare(c,path/'evaluation');train(c,None,path/'training',evaluation_path=path/'evaluation');pp='evaluation/test/0/problem.json';op='evaluation/test/0/observations.json'
 b=load_inference(path/'training/inference.pt',RuntimeSpec());p=ProblemSpec.from_dict(read_json(path/pp));o=ObservationSet.from_dict(read_json(path/op),p.grid)
 inf=replace(b.config.inference,steps=4,particles=8,controller={'max_particles':8},fk={'chunk_size':4,'ess_fraction':1.,'resample_every':1,'observation_weight':.01,'physics_weight':.0001,'observation_convention':'tempered_mean'})
 r=sample(b,p,o,inf,'fk');repeat=sample(b,p,o,inf,'fk')
 assert torch.equal(r.samples,repeat.samples) and torch.equal(r.weights,repeat.weights)
 assert r.provenance['terminal_target_complete'] and torch.allclose(r.weights.sum(1),torch.ones(1,dtype=torch.float64))
 assert any(e['action']=='RESAMPLE' for e in r.diagnostics['process']['actions'])
 assert r.diagnostics['model_particle_evaluations']==4*8
 assert r.diagnostics['model_evaluations']==8
 if family=='darcy':assert r.samples[:,:,0].min()>0 and r.samples[:,:,1].mean().abs()<1e-6
 if family=='chemistry':assert r.samples.min()>=0
 a=export_result(tmp_path/'no-reference',r,p,o);z=export_result(tmp_path/'truth',r,p,o,(torch.cat((torch.ones_like(r.samples[:,0,:1]),torch.zeros_like(r.samples[:,0,1:])),dim=1) if family=="darcy" else torch.ones_like(r.samples[:,0])))
 assert a['process']==z['process'] and a['population_weights']==z['population_weights'] and 'fields' not in a
 with np.load(tmp_path/'truth/result.npz') as arr:assert np.array_equal(arr['weights'],r.weights.numpy()) and 'fk_ancestry' in arr
 with pytest.raises(ValueError,match='refinement/retry'):sample(b,p,o,replace(inf,controller={'refine':True}),'fk')
 with pytest.raises(ValueError,match='require explicit'):sample(b,p,o,replace(inf,fk=None),'fk')
 # Exact final systematic resampling convention is explicit.
 unweighted=sample(b,p,o,replace(inf,fk={**inf.fk,'terminal_resample':True}),'fk')
 assert torch.allclose(unweighted.weights,torch.full((1,8),1/8,dtype=torch.float64))


def test_actual_fk_api_weights_slice_reference_and_invalid_config(tmp_path):
 from test_workbench import wait
 root=Path('outputs/accepted_cpu_2d')
 if not (root/'training/inference.pt').exists():
  from pdeflow.data import generate
  from pdeflow.training import train
  c=Config.load('configs/experiments/cpu_2d.json');c=replace(c,data=replace(c.data,count=6),model=replace(c.model,width=4,blocks=1),training=replace(c.training,steps=2))
  root=tmp_path/'portable';generate(c,root/'data');train(c,root/'data',root/'training')
 with TestClient(create_app([root/'training'],tmp_path/'jobs',tmp_path/'data')) as client:
  model=client.get('/api/models').json()[0];assert model['steering']['fk']
  demo=client.post('/api/demos',json={'model_id':model['id'],'problem':model['problem'],'sensors':3}).json()
  request={'model_id':model['id'],'problem':model['problem'],'rows':demo['rows'],'noise_std':.03,'mode':'fk','inference':{'steps':4,'particles':8,'fk':{'resample_every':1,'ess_fraction':1.,'chunk_size':4},'controller':{}}}
  bad={**request,'inference':{**request['inference'],'controller':{'refine':True}}}
  assert client.post('/api/reconstruction/validate',json=bad).status_code==422
  assert client.post('/api/reconstruction/validate',json={**request,'inference':{**request['inference'],'fk':{'ess_fraction':0}}}).status_code==422
  response=client.post('/api/jobs',json=request);assert response.status_code==202,response.text
  identifier=response.json()['id'];assert wait(client,identifier)['status']=='succeeded'
  url='/api/jobs/'+identifier
  def arrays():
   data=client.get(url+'/artifacts/samples.npz').content
   with np.load(io.BytesIO(data)) as a:return {k:a[k].copy() for k in a.files}
  before=arrays();m=client.get(url+'/result').json()['metrics']
  assert 'fields' not in m and m['process']['terminal_target_complete']
  field=client.get(url+'/slice?time_index=0').json()['values']
  expected=np.einsum('p,pxy->xy',before['weights'][0],before['samples'][0,:,0,0])
  assert np.allclose(field,expected)
  for reference_id in [demo['reference_id'],None]:
   assert client.post(url+'/reference',json={'reference_id':reference_id}).status_code==200
   after=arrays()
   for key in before:assert np.array_equal(before[key],after[key])
   assert client.get(url+'/result').json()['metrics']['process']==m['process']
