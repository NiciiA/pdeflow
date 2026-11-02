"""One-batch, no-step diagnostic against ORIGINAL matching source; never modifies run.

Use a separate process. A later generation index with earlier weights is explicitly
not historical trajectory replay. Exact reconstruction batch/noise seeds are restored.
"""
from pathlib import Path
import argparse,sys,json,tempfile,shutil,hashlib
import torch
from torch.utils._python_dispatch import TorchDispatchMode

class FirstInvalidOperation(TorchDispatchMode):
    """Targeted diagnostic only; deliberately ignores legitimate masked-fill -inf."""
    def __init__(self):super().__init__();self.first=None
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        result=func(*args,**(kwargs or {}));name=str(func)
        sensitive=any(key in name for key in ('mm.','bmm.','addmm.','_to_copy.','exp.','log.','pow.','sqrt.','div.','softmax','sum.','mean.','sub.','layer_norm'))
        if self.first is None and sensitive:
            outputs=result if isinstance(result,(tuple,list)) else (result,)
            for value in outputs:
                if torch.is_tensor(value) and value.is_floating_point() and not bool(torch.isfinite(value).all()):
                    self.first={'operation':name,'shape':list(value.shape),'dtype':str(value.dtype),'nonfinite_elements':int((~torch.isfinite(value)).sum())};break
        return result


def main():
 p=argparse.ArgumentParser();p.add_argument('--source-root',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--dataset',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda');p.add_argument('--generation-index',type=int);a=p.parse_args();source=Path(a.source_root).resolve()
 if not (source/'src/pdeflow').exists() and (source/'pdeflow/src/pdeflow').exists():source=source/'pdeflow'
 sys.path.insert(0,str(source/'src'))
 import pdeflow
 if not Path(pdeflow.__file__).resolve().is_relative_to(source):raise ValueError('A different pdeflow was already imported; use a fresh separate process')
 from pdeflow.mesh.provenance import read_checkpoint,scientific_source
 from pdeflow.mesh.model import MeshModel
 from pdeflow.mesh.data import Data
 from pdeflow.mesh.training import draw,slice_noise,ownership
 from pdeflow.mesh.objectives import loss
 from pdeflow.coordinate.optimization import precision
 from pdeflow.runtime import RuntimeSpec
 cp,c=read_checkpoint(a.checkpoint)
 if cp['scientific_source']['sha256']!=scientific_source()['sha256']:raise ValueError('Original source does not match checkpoint')
 if c.architecture!='mesh_m1_v1':raise ValueError('This original-run diagnostic is scoped to the affected M1 stage')
 device,_=RuntimeSpec(device=a.device).resolve();policy,amp=precision(c.precision,device);model=MeshModel(c).to(device);model.load_state_dict(cp['model'],strict=True);ownership(model,'reconstruction');output=Path(a.output);output.mkdir(parents=True,exist_ok=False)
 index=cp['successful_updates'] if a.generation_index is None else a.generation_index
 def summary(x):
  if torch.is_tensor(x):
   v=x.detach().float();f=torch.isfinite(v);z=v[f];return {'shape':list(x.shape),'dtype':str(x.dtype),'nonfinite':int((~f).sum()),'min':float(z.min()) if len(z.flatten()) else None,'max':float(z.max()) if len(z.flatten()) else None}
  if isinstance(x,(tuple,list)):return [summary(v) for v in x]
  if isinstance(x,dict):return {str(k):summary(v) for k,v in x.items()}
  return x
 def cpu(x):
  if torch.is_tensor(x):return x.detach().cpu()
  if isinstance(x,tuple):return tuple(cpu(v) for v in x)
  if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
  return x
 with tempfile.TemporaryDirectory(prefix='mesh-original-batch-') as tmp:
  path=Path(tmp);shutil.copy2(a.dataset,path/'dataset.json');data=Data(c,path,create=False);batch=data.prepare(index).to(device)
  generator=torch.Generator();generator.set_state(cp['generator']);draws=draw(c,c.batch_size,generator,device)
  torch.save({'batch':cpu(batch.__dict__),'draws':cpu(draws),'generation_index':index,'checkpoint_sha256':hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest()},output/'batch.pt');results={}
  for label,enabled,micro in [('resolved_mixed',policy!='fp32',c.microbatch),('explicit_fp32',False,c.microbatch),('resolved_micro1',policy!='fp32',1)]:
   if 'torch_rng' in cp:torch.set_rng_state(cp['torch_rng'])
   if device.type=='cuda' and cp.get('cuda_rng'):torch.cuda.set_rng_state_all(cp['cuda_rng'])
   model.zero_grad(set_to_none=True);optimizer=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=c.learning_rate);optimizer.load_state_dict(cp['optimizer']);scaler=torch.amp.GradScaler('cuda',enabled=enabled and policy=='fp16')
   if scaler.is_enabled():scaler.load_state_dict(cp['scaler'])
   first=[];handles=[]
   def hook(name):
    def inspect(module,args,value):
     if torch.is_tensor(value) and value.is_floating_point() and not torch.isfinite(value).all() and not first:first.append({'module':name,'output':summary(value)})
    return inspect
   for name,module in model.named_modules():
    if not list(module.children()):handles.append(module.register_forward_hook(hook(name)))
   op_trace=FirstInvalidOperation();op_trace.__enter__()
   try:
    values_all=[];components=[]
    for start in range(0,c.batch_size,micro):
     end=min(c.batch_size,start+micro);noise,s,k,q=draws
     with torch.autocast(device.type,dtype=amp,enabled=enabled):values,parts=loss(model,batch.slice(start,end),'reconstruction',data.norms,slice_noise(noise,start,end),s[start:end],k[start:end],q[start:end])
     values_all.append(values.detach().cpu());components.append(summary(parts))
     if not torch.isfinite(values).all():break
     scaler.scale(values.float().sum()/c.batch_size).backward()
    scaler.unscale_(optimizer)
    invalid=[name for name,v in model.named_parameters() if v.grad is not None and not torch.isfinite(v.grad).all()]
    results[label]={'loss':summary(torch.cat(values_all)),'components':components,'first_nonfinite_module':first,'first_nonfinite_operation':op_trace.first,'invalid_gradients':invalid,'scale':scaler.get_scale(),'precision':policy if enabled else 'fp32','microbatch':micro,'optimizer_stepped':False}
   except Exception as error:results[label]={'error':str(error),'first_nonfinite_module':first,'first_nonfinite_operation':op_trace.first}
   finally:
    op_trace.__exit__(None,None,None)
    for h in handles:h.remove()
  record={'checkpoint':str(Path(a.checkpoint).resolve()),'checkpoint_update':cp['successful_updates'],'batch_generation_index':index,'source_sha256':scientific_source()['sha256'],'actual_resolved_precision':policy,'data':summary(batch.__dict__),'comparisons':results,'limits':'No optimizer steps. Exact weights/batch diagnostic, not complete historical trajectory replay. Old failure checkpoint stored post-draw RNG; M1 forward is deterministic and uses no generative draws, but historical CUDA kernel determinism is not guaranteed.'};(output/'report.json').write_text(json.dumps(record,indent=2,allow_nan=False));print(json.dumps({'output':str(output),'precision':policy,'comparisons':{k:{x:v.get(x) for x in ('loss','invalid_gradients','first_nonfinite_module','error')} for k,v in results.items()}},indent=2))

if __name__=='__main__':main()
