"""Production measurement packing shared by inference and evaluation diagnostics."""
from dataclasses import dataclass
import torch
from .contracts import validate_rows
from .training import normalized_parameter
from .physics import get,family_index


@dataclass
class ConditionInputs:
    parameter:torch.Tensor
    known:torch.Tensor
    families:torch.Tensor
    points:torch.Tensor
    values:torch.Tensor
    sigma:torch.Tensor
    quantities:torch.Tensor
    observations:tuple


def prepare(c,normalizations,problem,rows,particles,device):
    points,values,sigma=validate_rows(c,rows,problem);phys=get(problem.family)
    f=torch.full((particles,),family_index(problem.family),device=device,dtype=torch.long)
    known=torch.full((particles,),phys.parameter in problem.known_parameters,dtype=torch.bool,device=device)
    p=normalized_parameter(c,torch.full((particles,),problem.known_parameters[phys.parameter],device=device,dtype=torch.float32),f) if known.all() else torch.zeros(particles,device=device)
    ns=normalizations[problem.family];ids=torch.tensor([0 if r['field']=='u' else 1 for r in rows],dtype=torch.long,device=device)
    n=max(1,len(rows));oq=torch.zeros(particles,n,3,device=device);features=torch.zeros_like(oq);valid=torch.zeros(particles,n,dtype=torch.bool,device=device);quantities=torch.zeros(particles,n,dtype=torch.long,device=device)
    if rows:
        oq[:,:len(rows)]=points.to(device);valid[:,:len(rows)]=True;quantities[:,:len(rows)]=ids
        for j in range(len(phys.state)):
            ix=ids.eq(j);indices=torch.where(ix)[0]
            features[:,indices,0]=(values.to(device)[ix]-ns[j]['mean'])/ns[j]['scale']
            features[:,indices,1]=sigma.to(device)[ix]/ns[j]['scale'];features[:,indices,2]=1
    return ConditionInputs(p,known,f,points,values,sigma,ids,(oq,features,valid,quantities))


def physical_rows(observations,normalizations,family,index=0):
    """Inverse quantity normalization for saving actual packed scalar measurements."""
    points,features,valid,quantities=observations;rows=[]
    for j in torch.where(valid[index])[0].tolist():
        quantity=int(quantities[index,j]);norm=normalizations[family][quantity];q=points[index,j].detach().cpu().tolist()
        rows.append({'field':('u','v')[quantity],'x':q[0],'y':q[1],'time':q[2],
                     'value':float(features[index,j,0])*norm['scale']+norm['mean'],'sigma':float(features[index,j,1])*norm['scale']})
    return rows
