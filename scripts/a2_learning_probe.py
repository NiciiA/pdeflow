"""Explicit bounded fixed-collection overfit; NOT production training or accuracy proof."""
import argparse,json,time,hashlib
from pathlib import Path
from dataclasses import replace
import numpy as np
import torch
from pdeflow.colab import training_preset
from pdeflow.coordinate.training import train,read_checkpoint
from pdeflow.coordinate.model import Autoencoder
from pdeflow.coordinate.dataset import Dataset,item_quantity
from pdeflow.coordinate.batching import StatePlan,forward
from pdeflow.coordinate.frozen import FrozenTargets
from pdeflow.coordinate.profiling import Timings
from pdeflow.coordinate.evolution_monitor import evaluate
from pdeflow.coordinate.evolution_telemetry import UpdateProbe
from pdeflow.field_losses import grouped_queries,differentiable_loss
from pdeflow.runtime import RuntimeSpec,write_json,atomic_write,synchronize


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--device',default='cpu',choices=['cpu','mps']);p.add_argument('--a1-updates',type=int,default=120);p.add_argument('--a2-updates',type=int,default=200);p.add_argument('--run',action='store_true');p.add_argument('--fixed-a1',action='store_true',help='Explicit separate diagnostic: fit A1 on the same two trajectories/five states before freezing');a=p.parse_args()
    if not a.run:print('No execution. Add --run for an explicit two-solution Smoke-capacity overfit diagnostic.');return
    if not 2<=a.a1_updates<=200 or not 2<=a.a2_updates<=300:raise ValueError('Probe budgets are bounded: A1<=200,A2<=300')
    root=Path(a.output)
    if root.exists():raise FileExistsError('Use a new output root; all earlier probes are preserved')
    c=training_preset('pipeline-check',a.device,'fp32',{'train_count':12,'validation_count':12,'encoder_points':64,'query_points':256,'reference_shape':[16,16],'learning_rate':.002,'ae_steps':a.a1_updates,'evolution_steps':a.a2_updates,'checkpoint_every':a.a1_updates,'field_loss':{'type':'spectral_mse','spectral_weight':.1,'spectral_resolution':16,'spectral_frequency_power':0.},'training_options':{'validation_every':a.a1_updates,'validation_subset':2,'microbatch':2,'accumulation':1}})
    root.mkdir(parents=True);write_json(root/'config.json',c.to_dict());start=time.perf_counter()
    dev,_=RuntimeSpec(device=a.device,threads=2).resolve()
    if not a.fixed_a1:
        r=train(c.to_dict(),root/'autoencoder','autoencoder');cp,_=read_checkpoint(r['export'])
        model=Autoencoder(c).to(dev);model.load_state_dict(cp['autoencoder'])
    else:
        torch.manual_seed(c.seed);model=Autoencoder(c).to(dev);cp=None
    data=Dataset(c,root/'autoencoder',True);norms=data.metadata['normalizations'];items=[data.get('train',i) for i in (0,1)];heldout=[data.get('validation',i) for i in (0,1)]
    fixed=[]
    # FIXED times/queries only in this named overfit probe. Production sampling
    # remains arbitrary continuous time and is not replaced by these snapshots.
    for t in (.05,.237,.5,1.):
        rows=[]
        for i,item in enumerate(items):
            queries=[]
            for j,name in enumerate(('u','v') if item['family']=='wave' else ('u',)):
                g=torch.Generator().manual_seed(431+j+i*17);q,y,w,_=grouped_queries(c,[item_quantity(item,name)],g,norms[item['family']][j],None,[t]);n=norms[item['family']][j]
                queries.append((q[0,:,:2],(y[0]-n['mean'])/n['scale'],w[0]))
            rows.append(StatePlan(item,0.,t,t/2,queries,c.seed+i))
        fixed.append(rows)
    if a.fixed_a1:
        model.requires_grad_(False)
        for m in (model.encoder,model.decoder,model.coefficient):m.requires_grad_(True)
        ae_optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.002);ae_history=[]
        initial_rows=[]
        for rows in fixed[:1]:
            for r in rows:
                queries=[]
                for j,name in enumerate(('u','v') if r.item['family']=='wave' else ('u',)):
                    g=torch.Generator().manual_seed(431+j);q,y,w,_=grouped_queries(c,[item_quantity(r.item,name)],g,norms[r.item['family']][j],None,[0.]);n=norms[r.item['family']][j]
                    queries.append((q[0,:,:2],(y[0]-n['mean'])/n['scale'],w[0]))
                initial_rows.append(StatePlan(r.item,0.,0.,0.,queries,c.seed))
        state_rows=[initial_rows]+[[replace(r,ta=r.tb) for r in rows] for rows in fixed]
        for step in range(a.a1_updates):
            ae_optimizer.zero_grad(set_to_none=True)
            loss=forward(c,model,state_rows[step%len(state_rows)],'autoencoder',norms,dev,Timings(dev,False),None,differentiable_loss,item_quantity,{}).mean()
            loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],10.,error_if_nonfinite=True);ae_optimizer.step();ae_history.append(float(loss.detach()))
        write_json(root/'fixed-a1-protocol.json',{'qualification':'Separate diagnostic protocol, NOT production time sampler or a gate-qualified dependency','train_ids':[x['id'] for x in items],'times':[0.,.05,.237,.5,1.],'normalization_fit_ids':data.metadata['fitted_ids'],'updates':a.a1_updates,'losses':ae_history})
    model.requires_grad_(False)
    for p in model.parameters():p.grad=None
    model.evolution.requires_grad_(True);model.graph_encoder.requires_grad_(True)
    timings=Timings(dev,False);cache=FrozenTargets('named-fixed-overfit',128,timings,None,0);optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.002)
    frozen_before={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if k.startswith(('encoder.','decoder.','coefficient.'))};parameters=[p for p in model.parameters() if p.requires_grad]
    def report():
        train_report,_=evaluate(c,model,norms,items,resolution=16,include_oracle=True,include_physics=True)
        validation,_=evaluate(c,model,norms,heldout,resolution=16,include_oracle=True,include_physics=True)
        return {'fixed_training_collection':train_report,'independent_validation':validation}
    before=report();history=[]
    def gradient_probe():
        losses,terms=forward(c,model,fixed[2],'evolution',norms,dev,timings,cache,differentiable_loss,item_quantity,{},True)
        def gradient(term):
            grads=torch.autograd.grad(term.mean(),parameters,retain_graph=True,allow_unused=True)
            return torch.cat([(g if g is not None else torch.zeros_like(p)).float().reshape(-1) for p,g in zip(parameters,grads)])
        domain=gradient(terms['domain']);spectral=gradient(terms['spectral']);alignment=gradient(terms['alignment']);composition=gradient(terms['composition'])
        return {'domain_gradient_l2':float(domain.norm()),'weighted_spectral_gradient_l2':float(spectral.norm()),'domain_spectral_cosine':float(torch.dot(domain,spectral)/(domain.norm()*spectral.norm()).clamp_min(1e-20)),'alignment_gradient_l2':float(alignment.norm()),'composition_gradient_l2':float(composition.norm()),'spectral_power_zero':'Equivalent to spatial MSE on its structured16x16 grid (Parseval); not frequency selective. Extra random domain queries give a different quadrature estimator.'}
    gradients_before=gradient_probe()
    for step in range(a.a2_updates):
        optimizer.zero_grad(set_to_none=True);model.train();model.encoder.eval();components={};probe=UpdateProbe(model.named_parameters()) if step in (0,a.a2_updates-1) else None
        loss=forward(c,model,fixed[step%len(fixed)],'evolution',norms,dev,timings,cache,differentiable_loss,item_quantity,components).mean();loss.backward();norm=torch.nn.utils.clip_grad_norm_(parameters,10.,error_if_nonfinite=True);optimizer.step()
        row={'step':step+1,'fixed_time':fixed[step%len(fixed)][0].tb,'loss':float(loss.detach()),'gradient_norm_before_clip':float(norm)}
        if probe:row['actual_update']=probe.result()
        history.append(row)
    after=report();gradients_after=gradient_probe();cache.close()
    for k,v in model.state_dict().items():
        if k in frozen_before:assert torch.equal(v.detach().cpu(),frozen_before[k]),k
    atomic_write(root/'probe-weights.pt',lambda out:torch.save({'autoencoder':model.state_dict(),'config':c.to_dict(),'normalizations':norms,'stage':'named diagnostic fixed-collection overfit','updates':a.a2_updates},out))
    result={'qualification':'Small fixed TRAIN collection overfit and separate held-out diagnostics, not the current Medium/T4 run or production accuracy','before':before,'after':after,'gradients_before':gradients_before,'gradients_after':gradients_after,'history':history,'frozen_representation_unchanged':True,'seconds':time.perf_counter()-start,'architecture':'Unchanged Smoke physical-state v2; fixed2 realizations and4 times are diagnostic-only','fixed_a1_protocol':a.fixed_a1,'config':c.to_dict()}
    write_json(root/'result.json',result);print(json.dumps({'output':str(root),'seconds':result['seconds'],'before':before['fixed_training_collection']['ratio_to_held_initial'],'after':after['fixed_training_collection']['ratio_to_held_initial'],'validation_ratio_after':after['independent_validation']['ratio_to_held_initial'],'gradient_cosine_before':gradients_before['domain_spectral_cosine'],'gradient_cosine_after':gradients_after['domain_spectral_cosine']},indent=2))

if __name__=='__main__':main()
