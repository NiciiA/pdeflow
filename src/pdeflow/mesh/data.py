"""Clean reference targets and fresh sparse conditioning; only TRAIN fits scales."""
from dataclasses import dataclass
import numpy as np
import torch
from pdeflow.coordinate.dataset import Dataset,item_quantity,family_for
from pdeflow.coordinate.engine import observation_batch
from pdeflow.coordinate.temporal import sample_times
from pdeflow.field_losses import grouped_queries,AnalysisCache
from .state import parameter_normalize,pack
from .geometry import regular


@dataclass
class Batch:
    observations:tuple
    parameter:torch.Tensor
    known:torch.Tensor
    families:torch.Tensor
    target_state:torch.Tensor
    q0:torch.Tensor
    y0:torch.Tensor
    w0:torch.Tensor
    qt:torch.Tensor
    yt:torch.Tensor
    wt:torch.Tensor
    times:torch.Tensor
    ids:list
    domain_queries:int
    spectral0:torch.Tensor|None=None
    spectralt:torch.Tensor|None=None
    def slice(self,start,end):
        values={k:(tuple(x[start:end] for x in v) if k=='observations' else v[start:end] if torch.is_tensor(v) or k=='ids' else v) for k,v in self.__dict__.items()}
        return Batch(**values)
    def to(self,device):
        return Batch(**{k:(tuple(x.to(device) for x in v) if k=='observations' else v.to(device) if torch.is_tensor(v) else v) for k,v in self.__dict__.items()})


class Data:
    def __init__(self,c,root,create=True):
        self.c=c;self.reference=c.reference_config();self.dataset=Dataset(self.reference,root,create);self.norms=self.dataset.metadata['normalizations'];self.analysis=AnalysisCache();self.indices={}
        for split,count in [('train',self.reference.train_count),('validation',self.reference.validation_count),('test',self.reference.test_count)]:
            self.indices[split]={name:[i for i in range(count) if family_for(self.reference,split,i)==name] for name in c.family_weights}
    def prepare(self,index,split='train',fixed=False):
        c=self.c;r=self.reference;g=torch.Generator().manual_seed(c.seed+int(index)*100003+('train','validation','test').index(split)*1009);names=sorted(c.family_weights);weights=torch.tensor([c.family_weights[n] for n in names]);items=[]
        for j in range(c.batch_size):
            family=names[int(torch.multinomial(weights,1,generator=g))] if split=='train' and not fixed else names[j%len(names)]
            choices=self.indices[split][family];i=choices[int(torch.randint(len(choices),(),generator=g))] if split=='train' else choices[(index+j//len(names))%len(choices)];items.append(self.dataset.get(split,i))
        f=torch.tensor([int(x['family']=='wave') for x in items]);physical=torch.tensor([x['parameter'] for x in items]);p=parameter_normalize(c,physical,f);known=torch.rand(len(items),generator=g)<r.joint['known_probability']
        obs=observation_batch(r,items,g,self.norms,torch.device('cpu'));obs=list(obs)
        # Optional explicit corruption: actual metadata/noisy measurements are kept;
        # dropping time means dropping that sensor, never inventing a t=0 timestamp.
        drop=c.corruption.get('missing_probability',0.);valid=obs[2]
        if drop:valid=valid&(torch.rand(valid.shape,generator=g)>=drop)
        for key,selector in [('quantity_dropout',torch.randint(2,(len(items),1),generator=g)),('time_dropout',None)]:
            probability=c.corruption.get(key,0.)
            if probability:
                selected=torch.rand((len(items),1),generator=g)<probability
                criterion=obs[3].eq(selector) if selector is not None else obs[0][...,-1]>(c.interval[0]+c.interval[1])/2
                valid=valid&~(selected&criterion)
        obs[2]=valid;times=sample_times(r,len(items),g,'query').float();queries=[[],[]];targets=[[],[]];weight_rows=[[],[]];nodes=regular(c.node_shape).numpy()*np.asarray(c.domain['lengths'])+np.asarray(c.domain['origins']);anchor=[]
        for i,item in enumerate(items):
            ns=self.norms[item['family']];initial=np.zeros((c.nodes,2))
            fields=[];ys=[];ws=[]
            for j,name in enumerate(('u','v')):
                if item['family']=='heat' and j==1:
                    # Same bounded query layout as u; inactive v never contributes.
                    for phase in range(2):queries[phase][-1].append(queries[phase][-1][0].clone());targets[phase][-1].append(torch.zeros_like(targets[phase][-1][0]));weight_rows[phase][-1].append(torch.zeros_like(weight_rows[phase][-1][0]))
                    continue
                initial[:,j]=(item['solver'].evaluate(np.c_[nodes,np.zeros(len(nodes))],item['parameter'],r.interval,name)-ns[j]['mean'])/ns[j]['scale']
                for phase,time in enumerate((0.,float(times[i]))):
                    if j==0:queries[phase].append([]);targets[phase].append([]);weight_rows[phase].append([])
                    q,y,w,_=grouped_queries(r,[item_quantity(item,name)],g,ns[j],self.analysis,physical_times=[time]);queries[phase][-1].append(q[0,:,:2]);targets[phase][-1].append((y[0]-ns[j]['mean'])/ns[j]['scale']);weight_rows[phase][-1].append(w[0])
            anchor.append(initial)
        def tensor(rows):return torch.stack([torch.stack(x) for x in rows])
        state=pack(torch.tensor(np.stack(anchor),dtype=torch.float32),p)
        from pdeflow.field_losses import settings
        loss=settings(c.field_loss);spectral0=spectralt=None
        if loss.spectral_weight>0:
            arrays=[]
            for phase in (0,1):
                fields=[]
                for i,item in enumerate(items):
                    row=[]
                    for j,name in enumerate(('u','v')):
                        target=item['solver'].frame(0. if phase==0 else float(times[i]),item['parameter'],r.interval,(loss.spectral_resolution,)*2,name) if j==0 or item['family']=='wave' else np.zeros((loss.spectral_resolution,)*2)
                        norm=self.norms[item['family']][j];row.append((target-norm['mean'])/norm['scale'] if j==0 or item['family']=='wave' else target)
                    fields.append(row)
                arrays.append(torch.tensor(np.asarray(fields),dtype=torch.float32))
            spectral0,spectralt=arrays
        return Batch(tuple(obs),p,known,f,state,tensor(queries[0]),tensor(targets[0]),tensor(weight_rows[0]),tensor(queries[1]),tensor(targets[1]),tensor(weight_rows[1]),times,[x['id'] for x in items],r.query_points,spectral0,spectralt)
