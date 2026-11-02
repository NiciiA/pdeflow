"""Exceptional-path numerical capture and read-only checkpoint health checks."""
from pathlib import Path
from dataclasses import fields
import math,time,traceback
import torch
from pdeflow.runtime import write_json,atomic_write,file_digest


def tensor_summary(x):
    x=x.detach();finite=torch.isfinite(x) if x.is_floating_point() or x.is_complex() else torch.ones_like(x,dtype=torch.bool)
    values=x[finite].float()
    return {'shape':list(x.shape),'dtype':str(x.dtype),'elements':x.numel(),'nonfinite':int((~finite).sum()),'minimum':float(values.min()) if values.numel() else None,'maximum':float(values.max()) if values.numel() else None}


def tree_summary(value):
    if torch.is_tensor(value):return tensor_summary(value)
    if isinstance(value,dict):return {str(k):tree_summary(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [tree_summary(v) for v in value]
    return value


def cpu_tree(value):
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
    if isinstance(value,list):return [cpu_tree(v) for v in value]
    return value


def invalid_tensors(value,prefix=''):
    bad=[]
    if torch.is_tensor(value):
        if (value.is_floating_point() or value.is_complex()) and not bool(torch.isfinite(value).all()):bad.append({'path':prefix,**tensor_summary(value)})
    elif isinstance(value,float) and not math.isfinite(value):bad.append({'path':prefix,'nonfinite_scalar':str(value)})
    elif isinstance(value,dict):
        for k,v in value.items():bad.extend(invalid_tensors(v,prefix+'.'+str(k)))
    elif isinstance(value,(list,tuple)):
        for i,v in enumerate(value):bad.extend(invalid_tensors(v,prefix+f'[{i}]'))
    return bad


def checkpoint_health(path):
    """No model creation or training; check every model/optimizer tensor on CPU."""
    cp=torch.load(path,map_location='cpu',weights_only=True)
    if cp.get('model_partial_frozen_modules'):
        from .provenance import read_checkpoint
        cp,_=read_checkpoint(path)
    bad=invalid_tensors({'model':cp['model'],'optimizer':cp.get('optimizer',{}),'normalizations':cp.get('normalizations',{})})
    scales=[n.get('scale',0) for rows in cp.get('normalizations',{}).values() for n in rows]
    return {'path':str(Path(path).resolve()),'sha256':file_digest(path),'healthy':not bad and all(math.isfinite(s) and s>0 for s in scales),'invalid_tensors':bad[:30],'successful_updates':cp.get('successful_updates'),'attempted_updates':cp.get('attempted_updates'),'skipped_updates':cp.get('skipped_updates'),'failed_attempts':cp.get('failed_attempts',0),'completed_budget':cp.get('completed'),'precision':cp.get('precision'),'scaler':cp.get('scaler'),'scientific_source':cp.get('scientific_source',{}).get('sha256'),'health_scope':'Every saved model/optimizer floating tensor and positive normalization scale; not accuracy or checksum proof of an external Drive snapshot'}


class NonfiniteError(FloatingPointError):
    def __init__(self,location):self.location=location;super().__init__('Nonfinite '+location+'; optimizer not stepped')


def require_finite(value,location):
    bad=invalid_tensors(value)
    if bad:raise NonfiniteError(location+': '+bad[0]['path'])


class Trace:
    """Installed only on exceptional replay, never during ordinary updates."""
    def __init__(self,model):self.rows=[];self.handles=[];self.first=None;self.model=model
    def __enter__(self):
        from .numerical import trace_sink
        def functional(name,value):
            bad=invalid_tensors(value)
            if bad and self.first is None:self.first=name
            if len(self.rows)<300 or bad:self.rows.append({"operation":name,"output":tree_summary(value)})
        self.token=trace_sink.set(functional)
        def hook(name):
            def inspect(module,args,result):
                bad=invalid_tensors(result);row={'module':name,'type':type(module).__name__,'output':tree_summary(result)}
                if bad and self.first is None:self.first=name
                if len(self.rows)<300 or bad:self.rows.append(row)
            return inspect
        for name,module in self.model.named_modules():
            if not list(module.children()):self.handles.append(module.register_forward_hook(hook(name)))
        return self
    def __exit__(self,*args):
        from .numerical import trace_sink
        trace_sink.reset(self.token)
        for h in self.handles:h.remove()


def capture_failure(path,model,optimizer,batch,draws,rng,context,error,replay):
    """No writes to resume.pt. Full exact pre-update replay state is separate."""
    import uuid
    stage_path=Path(path);base=stage_path/'failure';base.mkdir(exist_ok=True);path=base/(str(context['attempted_updates'])+'-'+uuid.uuid4().hex[:12]);path.mkdir();report={**context,'unix':time.time(),'error':str(error),'traceback':traceback.format_exc(),'location':getattr(error,'location',context.get('phase')),'batch':tree_summary(batch.__dict__) if batch is not None else None,'parameter_ranges':{n:tensor_summary(p) for n,p in model.named_parameters()},'model_invalid':invalid_tensors(model.state_dict())[:20],'optimizer_invalid':invalid_tensors(optimizer.state_dict())[:20],'last_good_checkpoint':None}
    checkpoint=stage_path/'resume.pt'
    if checkpoint.exists():report['last_good_checkpoint']=checkpoint_health(checkpoint)
    if batch is not None and draws is not None:
        payload={'batch':cpu_tree(batch.__dict__),'draws':cpu_tree(draws),'rng_before_attempt':cpu_tree(rng),'context':context}
        atomic_write(path/'batch.pt',lambda f:torch.save(payload,f))
        # Needed to replay an uncheckpointed update exactly. Never advertised as healthy resume.
        state={**payload,'model':cpu_tree(model.state_dict()),'optimizer':cpu_tree(optimizer.state_dict())}
        atomic_write(path/'replay.pt',lambda f:torch.save(state,f))
        # Replay diagnostics use the same batch/weights and restore global RNG afterwards.
        report['comparisons']=replay()
    write_json(path/'report.json',report);write_json(base/'latest.json',{'directory':path.name,'report':'report.json'});return report


def compare_replay(model,batch,draws,norms,stage,policy,amp,rng,scaler_state=None):
    """Diagnostic forwards/backwards, same weights/batch/noise; never optimizer.step."""
    from .objectives import loss
    from pdeflow.coordinate.random_state import capture,restore
    before=capture();cpu=torch.get_rng_state();cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [];result={};noise,s,k,q=draws
    try:
        for label,enabled in [('resolved_precision',policy!='fp32'),('explicit_fp32',False)]:
            restore(rng['ambient']);torch.set_rng_state(rng['torch']);
            if cuda and rng.get('cuda'):torch.cuda.set_rng_state_all(rng['cuda'])
            model.zero_grad(set_to_none=True)
            try:
                parameters=[p for p in model.parameters() if p.requires_grad];diagnostic_optimizer=torch.optim.AdamW(parameters)
                diagnostic_scaler=torch.amp.GradScaler('cuda',enabled=enabled and policy=='fp16')
                if diagnostic_scaler.is_enabled() and scaler_state:diagnostic_scaler.load_state_dict(scaler_state)
                # Only exceptional replay traces backward origins. Normal training stays untouched.
                with Trace(model) as trace,torch.autograd.detect_anomaly(check_nan=True):
                    with torch.autocast(batch.parameter.device.type,dtype=amp,enabled=enabled):values,parts=loss(model,batch,stage,norms,noise,s,k,q)
                    valid=not invalid_tensors({'loss':values,'components':parts})
                    if valid:
                        diagnostic_scaler.scale(values.float().mean()).backward();diagnostic_scaler.unscale_(diagnostic_optimizer)
                result[label]={'finite_objective':valid,'first_nonfinite_module':trace.first,'modules':trace.rows,'loss':tree_summary(values),'components':tree_summary(parts),'gradient_invalid':invalid_tensors({n:p.grad for n,p in model.named_parameters() if p.grad is not None})[:20],'precision':str(amp) if enabled else 'torch.float32','backward_scaler_scale':diagnostic_scaler.get_scale(),'optimizer_stepped':False}
            except Exception as error:result[label]={'diagnostic_error':str(error),'traceback':traceback.format_exc(),'first_nonfinite_module':getattr(locals().get('trace'),'first',None)}
    finally:
        model.zero_grad(set_to_none=True);restore(before);torch.set_rng_state(cpu)
        if cuda:torch.cuda.set_rng_state_all(cuda)
    return result


def replay_failure(path,device='cpu'):
    from .contracts import MeshConfig
    from .data import Batch
    from .model import MeshModel
    from .training import ownership
    from .provenance import scientific_source
    from pdeflow.coordinate.optimization import precision
    root=Path(path)
    if (root/'latest.json').exists():root=root/__import__('pdeflow.runtime',fromlist=['read_json']).read_json(root/'latest.json')['directory']
    payload=torch.load(root/'replay.pt',map_location='cpu',weights_only=True);context=payload['context'];c=MeshConfig(**context['config']).validate()
    if context['source']['sha256']!=scientific_source()['sha256']:raise ValueError('Replay requires the captured scientific source; do not bypass source compatibility')
    from pdeflow.runtime import RuntimeSpec
    resolved,_=RuntimeSpec(device=device).resolve();policy,amp=precision(c.precision,resolved);model=MeshModel(c).to(resolved);model.load_state_dict(payload['model'],strict=True);ownership(model,context['stage']);batch=Batch(**payload['batch']).to(resolved)
    def to(x):return x.to(resolved) if torch.is_tensor(x) else tuple(to(v) for v in x) if isinstance(x,tuple) else x
    return compare_replay(model,batch,to(payload['draws']),context['normalizations'],context['stage'],policy,amp,payload['rng_before_attempt'],context.get('scaler'))
