"""Bounded B compute overhead, identical model/conditioning/workload; no training run.

Uses actual frozen T/D, flow and observation encoder; random latent fixtures are
explicitly NOT PDE predictions or accuracy evidence. No optimizer/checkpoint write.
"""
import argparse,json,time,statistics
from dataclasses import replace
from pathlib import Path
import torch
from pdeflow.colab import training_preset
from pdeflow.coordinate.contracts import CoordinateConfig
from pdeflow.coordinate.model import Autoencoder,Velocity
from pdeflow.coordinate.observation_objective import endpoint_loss
from pdeflow.coordinate.joint import path,flow_loss
from pdeflow.coordinate.optimization import precision,options
from pdeflow.runtime import RuntimeSpec,synchronize,write_json


def run(c,device,warmup=2,measured=5):
    if not 1<=warmup<=5 or not 5<=measured<=20:raise ValueError('Bounded benchmark requires 1..5 warmup and 5..20 measured steps')
    torch.manual_seed(c.seed);dev,_=RuntimeSpec(device=device,threads=2).resolve();policy,amp=precision(options(c)['precision'],dev)
    ae=Autoencoder(c).to(dev).eval().requires_grad_(False);flow=Velocity(c).to(dev).train();B=c.batch_size
    f=torch.arange(B,device=dev)%2;p=torch.full((B,),.5,device=dev);known=torch.arange(B,device=dev)%2==0;target=torch.randn(B,c.tokens,c.width,device=dev);noise=torch.randn_like(target);s=torch.full((B,),.75,device=dev);state,velocity,_=path(c,target,noise,s,known,f)
    N=c.observation_counts[1];q=torch.rand(B,N,3,device=dev);q[...,:2]=q[...,:2]*q.new_tensor(c.lengths)+q.new_tensor(c.origins);q[...,-1]*=c.interval[1]
    features=torch.randn(B,N,3,device=dev);features[...,1]=.05;features[...,2]=1;ids=torch.zeros(B,N,dtype=torch.long,device=dev);ids[f.eq(1),1::2]=1
    obs=(q,features,torch.ones(B,N,dtype=torch.bool,device=dev),ids);norms={'heat':[{'mean':0.,'scale':1.}],'wave':[{'mean':0.,'scale':1.},{'mean':0.,'scale':1.}]};graph=ae.graph_encoder(f).detach();mean=torch.zeros(1,c.tokens,c.width,device=dev);scale=torch.ones_like(mean)
    cfg={'beta':.01,'minimum_s':.5,'sensor_budget':8,'query_chunk':8,'discrepancy_relative':.05}
    results={}
    for name,aux in [('baseline',{}),('observation_aligned',cfg)]:
        cc=replace(c,joint={**c.joint,'observation_auxiliary':aux}).validate();times=[];aux_times=[]
        if dev.type=='cuda':torch.cuda.reset_peak_memory_stats(dev)
        for step in range(warmup+measured):
            flow.zero_grad(set_to_none=True);synchronize(dev);start=time.perf_counter()
            with torch.autocast(dev.type,dtype=amp,enabled=policy!='fp32'):
                pred=flow(state,s,p,obs,known,f,graph);base=flow_loss(cc,pred,velocity,known,f)
                synchronize(dev);at=time.perf_counter();extra,_=endpoint_loss(ae,cc,state,pred,s,mean,scale,p,known,f,obs,norms,100,123);synchronize(dev);ae_time=time.perf_counter()-at
                loss=base+extra.mean()
            # FP32/MPS backward; CUDA uses the same explicit scaler policy.
            scaler=torch.amp.GradScaler('cuda',enabled=policy=='fp16',init_scale=128.)
            scaler.scale(loss).backward();synchronize(dev);elapsed=time.perf_counter()-start
            if not torch.isfinite(loss) or any(x.grad is not None and not torch.isfinite(x.grad).all() for x in flow.parameters()):raise FloatingPointError('Nonfinite benchmark loss/gradient')
            if step>=warmup:times.append(elapsed);aux_times.append(ae_time)
        results[name]={'median_forward_backward_seconds':statistics.median(times),'min_seconds':min(times),'max_seconds':max(times),'measured_seconds':times,'examples_per_second':B/statistics.median(times),'median_endpoint_forward_seconds':statistics.median(aux_times),'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(dev) if dev.type=='cuda' else None,'mps_current_allocated_bytes':torch.mps.current_allocated_memory() if dev.type=='mps' else None}
    return {'scope':'Compute-only forward/backward with identical random fixtures; excludes reference/target prep, optimizer, validation and persistence. No accuracy result. CUDA peak only when available; MPS current allocation is not peak memory.', 'device':str(dev),'precision':policy,'config':c.to_dict(),'warmup':warmup,'measured':measured,'auxiliary':cfg,'results':results,'overhead_ratio':results['observation_aligned']['median_forward_backward_seconds']/results['baseline']['median_forward_backward_seconds']}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--device',default='cpu');p.add_argument('--output',required=True);a=p.parse_args()
    c=CoordinateConfig(**json.loads(Path(a.config).read_text())).validate() if a.config else training_preset('pipeline-check',a.device,'auto',{'encoder_points':16,'query_points':16,'field_loss':{'analysis_resolution':8,'region_queries':16,'time_groups':1}})
    output=Path(a.output)
    if output.exists():raise FileExistsError('Use a new benchmark output; existing evidence preserved')
    result=run(c,a.device);write_json(output,result);print(json.dumps(result,indent=2))
