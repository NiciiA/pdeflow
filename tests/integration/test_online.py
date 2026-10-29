from dataclasses import replace
import signal,os
import pytest
import torch
from pdeflow.runtime import Config,read_json,write_json
from pdeflow.data.physical import specification,solve_spec,seed_for
from pdeflow.data.providers import PhysicalProvider,prepare,load_sample
from pdeflow.training import train,load_inference


def config(family='heat',mode='online',dim=2):
 c=Config.load(f'configs/experiments/online_{family}_{dim}d.json')
 return replace(c,provider=replace(c.provider,mode=mode,calibration_count=2,validation_count=1,test_count=1,capacity=2,offline_count=4),training=replace(c.training,steps=3,batch_size=2,checkpoint_every=3),model=replace(c.model,width=4,blocks=1))

@pytest.mark.parametrize('family',['heat','darcy'])
def test_stable_specs_fixed_evaluation_modes_and_calibration(family,tmp_path):
 c=config(family);spec,_=specification(c,'train',0);a=solve_spec(c,spec);b=solve_spec(c,spec)
 assert torch.equal(a.values,b.values)
 seeds={seed_for(c,ns,0) for ns in ['train','calibration','validation','test']};assert len(seeds)==4
 evaluation=prepare(c,tmp_path/'eval');before=(tmp_path/'eval/manifest.json').read_bytes()
 offline=replace(c,provider=replace(c.provider,mode='offline'));other=prepare(offline,tmp_path/'eval-other');assert other['identity']==evaluation['identity'];prepare(offline,tmp_path/'offline',True)
 cached=load_sample(tmp_path/'offline/train/0',c);assert cached.specification==spec and torch.equal(cached.values,a.values)
 p=PhysicalProvider(c,tmp_path/'eval',tmp_path/'run');assert all(x.startswith('calibration-') for x in p.transform.fitted_ids)
 q=PhysicalProvider(offline,tmp_path/'eval',tmp_path/'other',tmp_path/'offline');assert p.transform==q.transform
 assert before==(tmp_path/'eval/manifest.json').read_bytes();p.close();q.close()

@pytest.mark.parametrize('mode',['online','hybrid'])
def test_exact_sequence_resume_and_bounded_cache(mode,tmp_path):
 c=config('darcy',mode);prepare(c,tmp_path/'eval')
 train(c,None,tmp_path/'full',evaluation_path=tmp_path/'eval')
 train(c,None,tmp_path/'resumed',stop_after=1,evaluation_path=tmp_path/'eval')
 train(c,None,tmp_path/'resumed',resume=tmp_path/'resumed/checkpoint.pt',evaluation_path=tmp_path/'eval')
 a=load_inference(tmp_path/'full/inference.pt');b=load_inference(tmp_path/'resumed/inference.pt')
 assert all(torch.equal(v,b.model.state_dict()[k]) for k,v in a.model.state_dict().items())
 ha=read_json(tmp_path/'full/history.json');hb=read_json(tmp_path/'resumed/history.json')
 assert [r['physical_ids'] for r in ha]==[r['physical_ids'] for r in hb]
 report=read_json(tmp_path/'resumed/data-accounting.json')
 assert report['presentations']==6 and len(report['pool'])<=2
 assert report['reused']==(2 if mode=='hybrid' else 0)
 assert report['unique_training_problems_consumed']==(4 if mode=='hybrid' else 6)


def test_bounded_failure_worker_rejection_and_cache_invalidation(tmp_path,monkeypatch):
 import pdeflow.data.providers as module
 c=config();prepare(c,tmp_path/'eval');p=PhysicalProvider(c,tmp_path/'eval',tmp_path/'run')
 def broken(*args):raise RuntimeError('controlled reference failure')
 monkeypatch.setattr(module,'solve_spec',broken)
 with pytest.raises(RuntimeError,match='exhausted 2 attempts'):p.training_batch(torch.Generator(),1)
 assert p.accounting['failed_attempts']==2;p.close()
 with pytest.raises(RuntimeError,match='closed'):p.training_batch(torch.Generator(),1)
 with pytest.raises(ValueError,match='workers=0'):replace(c,provider=replace(c.provider,workers=2)).validate()
 record=tmp_path/'eval/validation/0/sample.json';v=read_json(record);v['specification']['distribution']['version']='stale';write_json(record,v)
 with pytest.raises(ValueError,match='Immutable'):PhysicalProvider(c,tmp_path/'eval',tmp_path/'other')


def test_ctrl_c_deferred_safe_checkpoint_and_sequence(tmp_path,monkeypatch):
 c=config();prepare(c,tmp_path/'eval');train(c,None,tmp_path/'full',evaluation_path=tmp_path/'eval')
 original=torch.optim.Adam.step
 def interrupt_after_update(self,*args,**kwargs):
  result=original(self,*args,**kwargs);os.kill(os.getpid(),signal.SIGINT);return result
 with monkeypatch.context() as patch:
  patch.setattr(torch.optim.Adam,'step',interrupt_after_update)
  r=train(c,None,tmp_path/'resumed',evaluation_path=tmp_path/'eval')
 assert r['interrupted'] and r['progress']==1
 train(c,None,tmp_path/'resumed',resume=tmp_path/'resumed/checkpoint.pt',evaluation_path=tmp_path/'eval')
 a=load_inference(tmp_path/'full/inference.pt');b=load_inference(tmp_path/'resumed/inference.pt')
 assert all(torch.equal(v,b.model.state_dict()[k]) for k,v in a.model.state_dict().items())
