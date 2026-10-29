from copy import deepcopy
from dataclasses import replace
import io
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from pdeflow.runtime import Config,read_json,RuntimeSpec
from pdeflow.core import ProblemSpec
from pdeflow.data.providers import prepare
from pdeflow.training import train,load_inference
from pdeflow.sampling import sample
from pdeflow.observations import ObservationSet
from pdeflow.symbolic import presets,analyze
from pdeflow.workbench.api import create_app
from pdeflow.workbench.observations import observation_rows
from test_workbench import wait


def mathematical(p,rows=()):
 preset=next(x for x in presets() if x['id']==p.family)
 return {'schema_version':1,'equations':preset['equations'],'declarations':[{**d,'value':p.known_parameters.get(d['name'],d['value'])} for d in preset['declarations']],
 'domain':p.to_dict()['grid'],'boundary':{'type':'periodic','status':'known'},'initial':{'status':'unknown'},'measurements':{'rows':list(rows),'noise_model':'independent_gaussian','noise_std':.005,'status':'known'},'observation_operator':{'kind':'point_on_grid','status':'known'},'units':'nondimensional','preset':'misleading-heat-label'}

@pytest.fixture(scope='module')
def models(tmp_path_factory):
 root=tmp_path_factory.mktemp('chemistry-pipeline');configs={}
 for family,dim in [('heat',2),('darcy',2),('chemistry',2),('chemistry',3)]:
  c=Config.load(f'configs/experiments/online_{family}_{dim}d.json');c=replace(c,provider=replace(c.provider,calibration_count=2,validation_count=1,test_count=1),training=replace(c.training,steps=2,batch_size=2,checkpoint_every=1),model=replace(c.model,width=4,blocks=1))
  path=root/f'{family}{dim}';prepare(c,path/'evaluation');train(c,None,path/'training',stop_after=1,evaluation_path=path/'evaluation');train(c,None,path/'training',resume=path/'training/checkpoint.pt',evaluation_path=path/'evaluation');configs[family,dim]=(c,path)
 return root,configs

@pytest.mark.parametrize('dim',[2,3])
def test_chemical_common_training_reconstruction_species_and_reference_isolation(models,dim,tmp_path):
 root,configs=models;c,path=configs['chemistry',dim];inputs=path/'evaluation/test/0'
 p=ProblemSpec.from_dict(read_json(inputs/'problem.json'));obs=ObservationSet.from_dict(read_json(inputs/'observations.json'),p.grid);bundle=load_inference(path/'training/inference.pt',RuntimeSpec(device='cpu'))
 inf=replace(c.inference,steps=2,particles=1)
 with np.load(inputs/'reference.npz') as a:truth=torch.from_numpy(a['values'].copy())
 from pdeflow.evaluation import metrics
 for mode in ['unguided','observation','observation_pde']:
  first=sample(bundle,p,obs,inf,mode);assert first.samples.shape==(1,1,3,4,*p.grid.shape) and first.samples.min()>=0
  report=metrics(first,p,obs);assert 'fields' not in report and 'relative_l2_unobserved_mean' not in report
  correct=metrics(first,p,obs,truth);unrelated=metrics(first,p,obs,truth+1)
  assert set(correct['fields'])=={'a','b','c'} and 'field_u_mae' not in correct
  assert correct['fields']['a']['mae']!=unrelated['fields']['a']['mae']
  assert torch.equal(first.samples,sample(bundle,p,obs,inf,mode).samples)
 from pdeflow.cli import reconstruct
 for label,reference in [('correct',truth),('unrelated',truth+1)]:
  file=tmp_path/f'{label}.npz';np.savez(file,values=reference.numpy())
  reconstruct(replace(c,inference=inf),path/'training/inference.pt',inputs/'problem.json',inputs/'observations.json',tmp_path/label,file)
 for mode in ['unguided','observation','observation_pde']:
  with np.load(tmp_path/'correct'/mode/'result.npz') as a,np.load(tmp_path/'unrelated'/mode/'result.npz') as b:assert np.array_equal(a['samples'],b['samples'])


def test_chemical_equation_reordering_residuals_roles_and_kinetics(models):
 c,path=models[1]['chemistry',2];p=ProblemSpec.from_dict(read_json(path/'evaluation/test/0/problem.json'));draft=mathematical(p)
 assert analyze(draft)['recognized_family']=='chemistry' and analyze(draft)['problem'] is not None
 draft['equations']=list(reversed(draft['equations']));assert analyze(draft)['problem'] is not None
 draft['equations'][-1]='a_τ-(D_A lap(a)-r)=0';assert analyze(draft)['problem'] is not None
 bad=deepcopy(draft);bad['equations'][0]='r=k_f a b-k_r c+a';assert analyze(bad)['problem'] is None
 bad=deepcopy(draft);next(d for d in bad['declarations'] if d['name']=='k_f')['value']=-1;assert analyze(bad)['problem'] is None
 bad=deepcopy(draft);next(d for d in bad['declarations'] if d['name']=='a')['status']='known';assert analyze(bad)['problem'] is None

@pytest.mark.parametrize('dim',[2,3])
def test_api_actual_chemistry_jobs_routing_wrong_checkpoint_and_truth(models,tmp_path,dim):
 root,configs=models
 with TestClient(create_app([root],tmp_path/'jobs',tmp_path/'references')) as client:
  all_models=client.get('/api/models').json();assert {m['specialist'] for m in all_models}=={'heat','darcy','chemistry'}
  model=next(m for m in all_models if m['specialist']=='chemistry' and len(m['problem']['grid']['shape'])==dim)
  p=ProblemSpec.from_dict(model['problem']);draft=mathematical(p)
  routed=client.post('/api/mathematics/parse',json={'request':draft}).json();assert routed['routing']['selected_checkpoint']==model['id']
  assert model['training']['data_mode']=='online' and model['training']['evaluation_identity']
  demo=client.post('/api/demos',json={'model_id':model['id'],'problem':model['problem'],'sensors':3,'noise_std':.005}).json();draft=mathematical(p,demo['rows'])
  request={'model_id':model['id'],'problem':model['problem'],'rows':demo['rows'],'noise_std':.005,'mathematical_request':draft,'inference':{'steps':2,'particles':1}}
  wrong=next(m for m in all_models if m['specialist']=='heat');assert client.post('/api/jobs',json={**request,'model_id':wrong['id']}).status_code==422
  draft['equations'][-1]='r=k_f a b-k_r c+a';assert client.post('/api/jobs',json=request).status_code==422
  draft['equations'][-1]='r=k_f a b-k_r c'
  rejected=deepcopy(request);rejected['problem']['fields'][0]['name']='u';assert client.post('/api/jobs',json=rejected).status_code==422
  response=client.post('/api/jobs',json=request);assert response.status_code==202,response.text
  identifier=response.json()['id'];assert wait(client,identifier)['status']=='succeeded'
  url=f'/api/jobs/{identifier}';report=client.get(url+'/result').json();assert 'fields' not in report['metrics'] and report['baselines']=={}
  before=np.load(io.BytesIO(client.get(url+'/artifacts/samples.npz').content))['samples'].copy()
  assert client.post(url+'/reference',json={'reference_id':demo['reference_id']}).status_code==200
  assert set(client.get(url+'/result').json()['metrics']['fields'])=={'a','b','c'}
  for field in (0,1,2):
   value=client.get(url+f'/slice?field_index={field}&time_index=2&axis=0&index=3').json()
   assert value['field']==['a','b','c'][field] and value['physical_time']==.1
  assert client.post(url+'/reference',json={'reference_id':None}).status_code==200
  after=np.load(io.BytesIO(client.get(url+'/artifacts/samples.npz').content))['samples'];assert np.array_equal(before,after)
