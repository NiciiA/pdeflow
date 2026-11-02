"""Fixed observation-only validation; dynamics stage explicitly uses true physical nodes."""
import torch
from pdeflow.coordinate.evolution_monitor import evaluation_guard
from .sampling import generate,fgn_noise
from .objectives import predictions,quantity_mean


def validate(model,data,stage,cases=None,preview_root=None,update=None):
    c=model.c;device=next(model.parameters()).device;rows=[];images=[];labels=[]
    with evaluation_guard(model),torch.no_grad():
        for i in range(cases or c.validation_cases):
            batch=data.prepare(i,'validation').to(device)
            if stage=='dynamics':state=batch.target_state;scope='Oracle clean physical initial nodes, not observation-only reconstruction'
            elif stage=='central':
                state=model.initial(model.condition(batch.observations,batch.parameter,batch.known,batch.families),batch.parameter,batch.known,batch.families,fgn_noise(c,c.batch_size,torch.Generator(),device,True));scope='Deterministic zero-noise branch, not ensemble mean'
            else:
                context=model.condition(batch.observations,batch.parameter,batch.known,batch.families);state,_=generate(model,context,batch.parameter,batch.known,batch.families,torch.Generator().manual_seed(c.seed+991));scope='Observation-only fixed-seed prediction; no reference field enters model'
            initial=predictions(model,state,batch,True);truth=batch.y0;count=batch.domain_queries;initial_mse=quantity_mean((initial[...,:count]-truth[...,:count]).square().mean(-1),batch.families);constant0=quantity_mean(truth[...,:count].square().mean(-1),batch.families);trajectory=torch.zeros_like(initial_mse);constantt=torch.zeros_like(initial_mse)
            if c.kind['trajectory']:
                pred=predictions(model,state,batch,False);truth=batch.yt;trajectory=quantity_mean((pred[...,:count]-truth[...,:count]).square().mean(-1),batch.families);constantt=quantity_mean(truth[...,:count].square().mean(-1),batch.families)
            wi=.5 if c.kind['trajectory'] else 1.;wt=.5 if c.kind['trajectory'] else 0.;mse=wi*initial_mse+wt*trajectory;base=wi*constant0+wt*constantt
            if preview_root is not None and i==0:
                from .geometry import regular
                xy=regular((12,12)).to(device)*state.new_tensor(c.domain['lengths'])+state.new_tensor(c.domain['origins']);time=.237*c.interval[1] if c.kind['trajectory'] else 0.;pred=model.decode(state,xy,batch.families,time).cpu().numpy()
                for j in range(min(2,len(state))):
                    item=data.dataset.get('validation',int(batch.ids[j].split(':')[1]));family=item['family']
                    for quantity,name in enumerate(('u','v') if family=='wave' else ('u',)):
                        norm=data.norms[family][quantity];field=pred[j,:,quantity].reshape(12,12)*norm['scale']+norm['mean'];truth=item['solver'].frame(time,item['parameter'],c.interval,(12,12),name);physical_mse=float(__import__('numpy').mean((field-truth)**2));images.append(__import__('numpy').stack([truth,field,abs(field-truth)]));labels.append({'id':item['id'],'family':family,'quantity':name,'physical_time':time,'checkpoint_update':update,'rmse':physical_mse**.5,'normalized_mse':physical_mse/norm['scale']**2,'scope':scope,'stage':stage})
            for j in range(len(state)):
                rows.append({'id':batch.ids[j],'family':('heat','wave')[int(batch.families[j])],'initial_normalized_mse':float(initial_mse[j]),'trajectory_normalized_mse':float(trajectory[j]) if c.kind['trajectory'] else None,'physical_target_time':float(batch.times[j]) if c.kind['trajectory'] else 0.,'mse':float(mse[j]),'constant_training_mean_mse':float(base[j]),'scope':scope})
    if preview_root is not None and images:
        from pathlib import Path
        import numpy as np
        from pdeflow.runtime import atomic_write,write_json,file_digest
        path=Path(preview_root)/'preview.npz';atomic_write(path,lambda f:np.savez_compressed(f,frames=np.stack(images)));write_json(path.with_suffix('.json'),{'rows':labels,'array_sha256':file_digest(path),'checkpoint_update':update,'stage':stage,'meaning':'Fresh current-stage prediction, scope explicitly distinguishes oracle physical dynamics from observation-only inference'})
    a=sum(r['mse'] for r in rows)/len(rows);b=sum(r['constant_training_mean_mse'] for r in rows)/len(rows)
    return {'stage':stage,'records':rows,'uniform_domain_normalized_mse':a,'constant_training_mean_mse':b,'mse_ratio':a/max(b,1e-12),'passed':a<=c.gate_ratio*b,'criterion':'Matched uniform spatial domain MSE, equal active quantities/cases and initial/trajectory terms; not RB-vs-MSE loss comparison','accuracy':'Minimal baseline gate, not scientific adequacy','test_split_used':False}
