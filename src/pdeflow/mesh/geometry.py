"""Regular internal periodic nodes, degree-normalized local geometry and cell levels."""
import math
import torch
from torch import nn


def regular(shape):
    return torch.stack(torch.meshgrid(*[torch.arange(n,dtype=torch.float32,device="cpu")/n for n in shape],indexing='ij'),-1).reshape(-1,2)


def displacement(a,b):
    angle=2*math.pi*(b-a)
    return torch.atan2(angle.sin(),angle.cos())/(2*math.pi)


def neighbors(nodes,k,chunk=256):
    rows=[]
    for start in range(0,len(nodes),chunk):
        distance=displacement(nodes[start:start+chunk,None],nodes[None]).square().sum(-1)
        ix=torch.arange(start,min(start+chunk,len(nodes)),device=nodes.device);distance[torch.arange(len(ix),device=nodes.device),ix]=torch.inf
        rows.append(distance.argsort(dim=-1,stable=True)[:,:k])
    return torch.cat(rows)


def periodic_features(delta):
    angle=2*math.pi*delta
    return torch.cat([angle.sin(),angle.cos()],-1)


class Geometry(nn.Module):
    def __init__(self,c):
        super().__init__();fine=regular(c.node_shape);coarse=regular(c.coarse_shape);ix=neighbors(fine,c.neighbors)
        parent=(fine*torch.tensor(c.coarse_shape,device="cpu")).floor().long();parent=parent[:,0]*c.coarse_shape[1]+parent[:,1]
        delta=displacement(fine[:,None],fine[ix]);edge=torch.cat([delta,delta.square().sum(-1,keepdim=True),periodic_features(delta),torch.full((len(fine),c.neighbors,1),1/len(fine),device="cpu")],-1)
        self.register_buffer('nodes',fine);self.register_buffer('coarse',coarse);self.register_buffer('neighbors',ix);self.register_buffer('edges',edge);self.register_buffer('parent',parent)
        self.register_buffer('cell_weights',torch.full((len(fine),),1/len(fine),device="cpu"))
        self.register_buffer('coarse_mass',torch.bincount(parent,minlength=len(coarse)).float()/len(fine))
    def restrict(self,x):
        # FP32 aggregation; no dtype mismatch under native FP16 autocast.
        out=torch.zeros(len(x),len(self.coarse),x.shape[-1],device=x.device,dtype=torch.float32)
        out.index_add_(1,self.parent,x.float()*self.cell_weights[None,:,None])
        return (out/self.coarse_mass[None,:,None]).to(x.dtype)
    def prolong(self,x):return x[:,self.parent]


class LocalGraph(nn.Module):
    def __init__(self,width):
        super().__init__();self.norm=nn.LayerNorm(width);self.message=nn.Sequential(nn.Linear(2*width+8,width),nn.SiLU(),nn.Linear(width,width));self.update=nn.Sequential(nn.Linear(2*width,width),nn.SiLU(),nn.Linear(width,width))
    def forward(self,x,geometry):
        h=self.norm(x);target=h[:,:,None].expand(-1,-1,geometry.neighbors.shape[1],-1);source=h[:,geometry.neighbors];edge=geometry.edges[None].expand(len(x),-1,-1,-1).to(h.dtype)
        message=self.message(torch.cat([target,source,edge],-1)).float().mean(2).to(h.dtype)
        return x+self.update(torch.cat([h,message],-1))
