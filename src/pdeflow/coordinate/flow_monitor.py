"""Owner-thread CURRENT B preview, using measurements rather than an oracle z0."""
from pathlib import Path
import numpy as np
import torch
from pdeflow.runtime import atomic_write,write_json,file_digest
from .evolution_monitor import evaluation_guard
from .joint import active_mask
from .physics import family_index,get
from .training import normalized_parameter


def publish(root,c,ae,velocity,norms,mean,scale,items,update):
    from .engine import observation_batch
    dev=next(velocity.parameters()).device;g=torch.Generator().manual_seed(c.seed+317);rows=[];frames=[]
    with evaluation_guard(ae),evaluation_guard(velocity),torch.no_grad():
        for item in items:
            f=torch.tensor([family_index(item['family'])],device=dev);known=torch.ones(1,dtype=torch.bool,device=dev)
            p=normalized_parameter(c,torch.tensor([item['parameter']],device=dev),f)
            obs=observation_batch(c,[item],g,norms,dev);latent=torch.randn(1,c.tokens,c.width,generator=g).to(dev)*active_mask(c,known,f)[...,None]
            context=velocity.condition(p,obs,known,f,ae.graph_encoder(f))
            for k in range(4):latent=latent+velocity(latent,torch.tensor([k/4],device=dev),conditioning=context)/4
            z=latent*scale.to(dev)+mean.to(dev);t=.237*c.interval[1]
            axes=[torch.arange(12,device=dev)*c.lengths[j]/12+c.origins[j] for j in range(2)];xy=torch.stack(torch.meshgrid(*axes,indexing='ij'),-1).reshape(-1,2)
            queries=torch.cat([xy,torch.full((len(xy),1),t,device=dev)],-1)
            for j,name in enumerate(get(item['family']).state):
                norm=norms[item['family']][j];prediction=(ae.decode(z,queries,p,known,name,f)[0]*norm['scale']+norm['mean']).cpu().numpy().reshape(12,12)
                reference=item['solver'].frame(t,item['parameter'],c.interval,(12,12),name);mse=float(np.mean((prediction-reference)**2))
                frames.append(np.stack([reference,prediction,np.abs(prediction-reference)]))
                rows.append({'id':item['id'],'family':item['family'],'quantity':name,'physical_time':t,'checkpoint_update':update,'rmse':mse**.5,'normalized_mse':mse/norm['scale']**2,'observations':int(obs[2].sum()),'parameter_status':'known','prediction':'Current B, one fixed-noise 4-Euler sample; no oracle initial state'})
    root=Path(root);path=root/'flow-preview.npz';atomic_write(path,lambda out:np.savez_compressed(out,frames=np.stack(frames)))
    result={'stage':'flow','checkpoint_update':update,'rows':rows,'array':path.name,'array_sha256':file_digest(path),'meaning':'Fixed validation examples, fresh current B prediction. B0/B1 FM and B2/B3 combined losses are not directly comparable. Execution preview, not accuracy evidence.'}
    write_json(root/'flow-preview.json',result);return result
