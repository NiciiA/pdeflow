"""Bounded deterministic observation→initial-latent regression, separate from B."""
from pathlib import Path
import time,hashlib
import numpy as np
import torch
from torch import nn
from pdeflow.runtime import write_json,atomic_write,file_digest,digest
from pdeflow.core import GridSpec,DomainSpec
from .inference import CoordinateBundle
from .diagnostic_inventory import oracle_model
from .dataset import example,state_samples,identity
from .training import normalized_parameter
from .engine import observation_batch
from .joint import active_mask,flow_loss
from .physics import family_index,get,problem
from .contracts import grid_queries
from .diagnostic_metrics import sensor_metrics
from .diagnostics import exact_sensors


class DirectLatent(nn.Module):
    """Small diagnostic head over frozen typed observation/physics context."""
    def __init__(self,c,spatial=False):
        super().__init__();self.cross=nn.MultiheadAttention(c.width,c.heads,batch_first=True);self.norm=nn.LayerNorm(c.width);self.output=nn.Linear(c.width,c.width)
        from .spatial_conditioning import SensorAttention
        self.sensor_attention=SensorAttention(c) if spatial else None
    def forward(self,queries,context,mask,sensors=None):
        h=self.cross(queries,context,context,need_weights=False)[0]
        if self.sensor_attention is not None:h=h+self.sensor_attention(queries,sensors)
        return self.output(self.norm(h+queries))*mask[...,None]


def train_baseline(checkpoint,output,evolution=None,representation=None,device='cpu',steps=20,train_cases=4,validation_cases=4,seed=194,learning_rate=.001,spatial=False):
    if not 1<=steps<=200 or not 1<=train_cases<=32 or not 1<=validation_cases<=16:raise ValueError('Bounded diagnostic:1..200 steps,1..32 train and1..16 validation cases')
    if not np.isfinite(learning_rate) or learning_rate<=0:raise ValueError('Positive finite diagnostic learning rate required')
    root=Path(output)
    if root.exists():raise FileExistsError('Use a new diagnostic baseline directory')
    bundle=CoordinateBundle(checkpoint,device);bundle.checkpoint_path=str(Path(checkpoint).resolve());ae,lineage=oracle_model(bundle,evolution,representation)
    if ae is None:raise ValueError(lineage['reason']+'; deterministic latent targets cannot be invented')
    c=bundle.config
    if identity(c)!=bundle.cp['dataset_identity']:raise ValueError('Current reference recipe differs from checkpoint dataset; use matching scientific source')
    root.mkdir(parents=True);dev=bundle.device;g=torch.Generator().manual_seed(seed);torch.manual_seed(seed);head=DirectLatent(c,spatial).to(dev);opt=torch.optim.Adam(head.parameters(),lr=learning_rate)
    frozen_before=digest({k:file_digest(v) for k,v in [('checkpoint',checkpoint),*([('evolution',evolution)] if evolution else [])]})
    grid=GridSpec((16,16),DomainSpec(c.lengths,c.origins),(0.,.01,.05,.237,.5,1.));q=grid_queries(grid).to(dev);data=[]
    # Fixed train/validation sensor sets for the requested tiny overfit control.
    # Reuse production training sampling, not the diagnostic reference grid.
    with torch.no_grad():
        for split,n in [('train',train_cases),('validation',validation_cases)]:
            for i in range(n):
                item=example(c,split,i);f=torch.tensor([family_index(item['family'])],device=dev);p=normalized_parameter(c,torch.tensor([item['parameter']],device=dev),f)
                known=torch.tensor([float(torch.rand((),generator=g))<c.joint['known_probability']],device=dev);obs=observation_batch(c,[item],g,bundle.cp['normalizations'],dev)
                z=ae.encode(torch.tensor(item['context'],device=dev)[None],state_samples(item,item['context'],0.,c,bundle.cp['normalizations'][item['family']]).to(dev)[None],p,f)
                target=(z-bundle.mean)/bundle.scale;mask=active_mask(c,known,f)
                context=bundle.velocity.condition(p,obs,known,f,ae.graph_encoder(f))[0].detach()
                raw=bundle.velocity.observation_encoder.tokenize(obs[:3],obs[3]);sensors=tuple(x.detach() for x in raw)
                queries=bundle.velocity.geometry.embedding()[None].detach()
                # Recover physical noisy measurements from the actual packed inputs.
                rows=[]
                for j in torch.where(obs[2][0])[0].tolist():
                    quantity=int(obs[3][0,j]);norm=bundle.cp['normalizations'][item['family']][quantity];point=obs[0][0,j].cpu().tolist()
                    rows.append({'field':('u','v')[quantity],'x':point[0],'y':point[1],'time':point[2],'value':float(obs[1][0,j,0])*norm['scale']+norm['mean'],'sigma':float(obs[1][0,j,1])*norm['scale']})
                data.append({'split':split,'item':item,'family':f,'parameter':p,'known':known,'mask':mask,'observations':rows,'target':target.detach(),'context':context,'queries':queries,'sensors':sensors})
    def predict(r):
        sensors=head.sensor_attention.prepare(*r['sensors']) if spatial else None
        return head(r['queries'],r['context'],r['mask'],sensors)
    def losses(rows):
        return [flow_loss(c,predict(r),r['target'],r['known'],r['family']) for r in rows]
    def score():
        result=[];previous=bundle.ae;bundle.ae=ae
        try:
            with torch.no_grad():
                for r in data:
                    initial=predict(r);z=initial*bundle.scale+bundle.mean;item=r['item'];fields={}
                    for j,name in enumerate(get(item['family']).state):
                        pred=bundle.decode(z,q,r['parameter'],r['known'],r['family'],name).cpu().numpy()[0];truth=item['solver'].evaluate(q.cpu().numpy(),item['parameter'],c.interval,name)
                        fields[name]={'mse':float(np.mean((pred-truth)**2)),'rmse':float(np.sqrt(np.mean((pred-truth)**2))),'training_mean_baseline_mse':float(np.mean((bundle.cp['normalizations'][item['family']][j]['mean']-truth)**2))}
                    sensors=exact_sensors(bundle,z,r['parameter'],r['known'],r['family'],r['observations'])
                    result.append({'id':item['id'],'split':r['split'],'family':item['family'],'known_parameter':bool(r['known'][0]),'latent_mse':float(flow_loss(c,initial,r['target'],r['known'],r['family'])),'fields':fields,'sensors':sensor_metrics(sensors,r['observations'])})
        finally:bundle.ae=previous
        return result
    def frozen_hash():
        h=hashlib.sha256()
        for index,module in enumerate((ae,bundle.velocity)):
            for name,tensor in module.state_dict().items():h.update(f'{index}:{name}'.encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    frozen_state_before=frozen_hash()
    before=score();history=[];start=time.perf_counter();training=[r for r in data if r['split']=='train']
    for iteration in range(steps):
        opt.zero_grad(set_to_none=True);loss=torch.stack(losses(training)).mean()
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite deterministic diagnostic loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(head.parameters(),10.,error_if_nonfinite=True);opt.step();history.append({'successful_update':iteration+1,'latent_loss':float(loss.detach()),'gradient_norm':float(norm)})
    after=score();frozen_state_after=frozen_hash()
    if frozen_state_before!=frozen_state_after:raise RuntimeError('Frozen diagnostic backbone changed')
    assert frozen_before==digest({k:file_digest(v) for k,v in [('checkpoint',checkpoint),*([('evolution',evolution)] if evolution else [])]})
    assert all(not p.requires_grad and p.grad is None for m in (ae,bundle.velocity) for p in m.parameters())
    report={'kind':'deterministic_observation_to_initial_latent_diagnostic_v1','checkpoint_sha256':file_digest(checkpoint),'lineage':lineage,'frozen_state_sha256_before':frozen_state_before,'frozen_state_sha256_after':frozen_state_after,'successful_updates':steps,'settings':{'seed':seed,'learning_rate':learning_rate,'optimizer':'Adam','train_cases':train_cases,'validation_cases':validation_cases,'loss':'Active group-balanced standardized initial-latent MSE, not a flow velocity objective'},'history':history,'before':before,'after':after,'seconds':time.perf_counter()-start,'timing_definition':'Optimization and after-scoring; frozen feature preparation and before-scoring excluded; not a throughput benchmark',
            'spatial_sensor_reader':spatial,'trainable_parameters':sum(p.numel() for p in head.parameters()),'trainable':'New DirectLatent diagnostic head, including direct reader when selected','frozen':'Existing observation/typed physics encoders, geometry, state encoder, coefficient decoder, spatial decoder and evolution; B processor is not used',
            'observations':'Production training count/placement/time/quantity/noise sampler; fixed saved sets for tiny subset overfit. Validation sets are independent and never optimized.',
            'training_ids':[r['item']['id'] for r in training],'heldout_ids':[r['item']['id'] for r in data if r['split']=='validation'],'initial_noise':'none; deterministic predictor, no generative integration',
            'limits':['Tiny subset/head fit is not a production model or posterior','Latent regression need not give decoded conditional means','Failure with an inaccurate frozen representation does not prove observations lack information','Decoded metrics use16² at explicit times, equal node/time means; not uniform continuous-time quadrature']}
    write_json(root/'baseline.json',report);write_json(root/'observations.json',[{'id':r['item']['id'],'split':r['split'],'rows':r['observations']} for r in data]);atomic_write(root/'head.pt',lambda f:torch.save({'head':head.state_dict(),'config':c.to_dict(),'checkpoint_sha256':file_digest(checkpoint),'seed':seed},f));return report
