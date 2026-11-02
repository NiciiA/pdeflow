"""Sparse sensors -> local physical nodes, coarse cell memory and global context."""
import math
import torch
from torch import nn
from .geometry import Geometry,LocalGraph,periodic_features
from .numerical import fp32_matmul
from pdeflow.coordinate.model import Attention,Block


class Coordinates(nn.Module):
    def __init__(self,c,temporal=True):
        super().__init__();self.c=c;self.temporal=temporal;self.size=16+(9 if temporal else 0)
    def forward(self,q):
        xy=(q[...,:2]-q.new_tensor(self.c.domain['origins']))/q.new_tensor(self.c.domain['lengths']);phase=2*math.pi*xy[...,None]*q.new_tensor([1.,2.,4.,8.]);features=torch.cat([phase.sin().flatten(-2),phase.cos().flatten(-2)],-1)
        if self.temporal:
            t=(q[...,-1:]-self.c.interval[0])/(self.c.interval[1]-self.c.interval[0]);a=2*math.pi*t*q.new_tensor([1.,2.,4.,8.]);features=torch.cat([features,t,a.sin(),a.cos()],-1)
        return features


class SensorReader(nn.Module):
    """Exact streaming set attention; valid learned null makes zero sensors defined."""
    def __init__(self,c):
        super().__init__();self.c=c;self.coords=Coordinates(c);w=c.width;self.token=nn.Sequential(nn.Linear(self.coords.size+5,w),nn.SiLU(),nn.Linear(w,w));self.quantity=nn.Embedding(3,w);self.null=nn.Parameter(torch.zeros(1,1,w));self.q=nn.Linear(w,w);self.k=nn.Linear(w,w);self.v=nn.Linear(w,w);self.out=nn.Linear(w,w)
        self.bias=nn.Sequential(nn.Linear(8,32),nn.SiLU(),nn.Linear(32,1))
    def forward(self,queries,anchors,obs):
        q,features,valid,quantities=obs;q=torch.where(valid[...,None],q,torch.zeros_like(q));features=torch.where(valid[...,None],features,torch.zeros_like(features));quantities=torch.where(valid,quantities,torch.zeros_like(quantities))
        sigma=features[...,1:2];token=self.token(torch.cat([self.coords(q),features,torch.log1p(sigma),sigma.eq(0).to(features.dtype)],-1))+self.quantity(quantities)
        token=torch.cat([self.null.expand(len(q),-1,-1),token],1);mask=torch.cat([torch.ones(len(q),1,dtype=torch.bool,device=q.device),valid],1);q=torch.cat([q.new_zeros(len(q),1,3),q],1);quantities=torch.cat([quantities.new_full((len(q),1),2),quantities],1)
        keys=self.k(token);values=self.v(token);query=self.q(queries);pieces=[]
        xy=(q[...,:2]-q.new_tensor(self.c.domain['origins']))/q.new_tensor(self.c.domain['lengths']);t=(q[...,-1:]-self.c.interval[0])/(self.c.interval[1]-self.c.interval[0])
        for start in range(0,len(anchors),self.c.query_chunk):
            stop=min(len(anchors),start+self.c.query_chunk);Q=query[:,start:stop].float();maximum=den=num=None
            for low in range(0,len(token[0]),self.c.sensor_chunk):
                high=min(len(token[0]),low+self.c.sensor_chunk);delta=xy[:,None,low:high]-anchors[None,start:stop,None]
                position=periodic_features(delta);time=t[:,None,low:high].expand(-1,stop-start,-1,-1);kind=torch.nn.functional.one_hot(quantities[:,low:high],3)[:,None].expand(-1,stop-start,-1,-1).to(position.dtype)
                bias=self.bias(torch.cat([position,time,kind],-1)).squeeze(-1)
                if low==0:bias=torch.cat([torch.zeros_like(bias[:,:,:1]),bias[:,:,1:]],-1)
                logits=fp32_matmul(Q,keys[:,low:high].transpose(-1,-2))/math.sqrt(self.c.width)+bias.float();logits=logits.masked_fill(~mask[:,None,low:high],-torch.inf);current=logits.amax(-1,keepdim=True)
                if maximum is None:maximum=current;weight=(logits-maximum).exp();den=weight.sum(-1,keepdim=True);num=fp32_matmul(weight,values[:,low:high])
                else:
                    new=torch.maximum(maximum,current);factor=(maximum-new).exp();weight=(logits-new).exp();den=den*factor+weight.sum(-1,keepdim=True);num=num*factor+fp32_matmul(weight,values[:,low:high]);maximum=new
            pieces.append((num/den).to(queries.dtype))
        return self.out(torch.cat(pieces,1))


class MultiscaleBlock(nn.Module):
    def __init__(self,c):
        super().__init__();self.local=LocalGraph(c.width);self.global_block=Block(c.width,c.heads);self.read=Attention(c.width,c.heads);self.write=Attention(c.width,c.heads);self.coarse=nn.Linear(c.width,c.width);self.norm=nn.LayerNorm(c.width);self.exchange_scale=nn.Parameter(torch.tensor(.1))
    def forward(self,x,memory,geometry):
        x=self.local(x,geometry);coarse=geometry.restrict(x);memory=memory+self.read(self.norm(memory),self.norm(coarse));memory=self.global_block(memory);coarse=coarse+self.write(self.norm(coarse),self.norm(memory));x=x+self.exchange_scale*geometry.prolong(self.coarse(self.norm(coarse)));return x,memory


class Processor(nn.Module):
    def __init__(self,c):
        super().__init__();self.geometry=Geometry(c);self.memory=nn.Parameter(torch.randn(c.global_tokens,c.width)*.02);self.blocks=nn.ModuleList([MultiscaleBlock(c) for _ in range(c.depth)])
    def forward(self,x,context=None,modulations=None):
        memory=self.memory[None].expand(len(x),-1,-1)
        if context is not None:memory=memory+context[:,None]
        for index,block in enumerate(self.blocks):
            if modulations is not None:
                scale,shift=modulations[index];x=x*(1+scale)+shift
            x,memory=block(x,memory,self.geometry)
        return x,memory


class Conditioner(nn.Module):
    def __init__(self,c):
        super().__init__();self.c=c;self.geometry=Geometry(c);self.position=nn.Linear(16,c.width);self.sensor=SensorReader(c);self.physical=nn.Sequential(nn.Linear(5,c.width),nn.SiLU(),nn.Linear(c.width,c.width));self.processor=Processor(c)
        from dataclasses import replace
        from pdeflow.coordinate.graph import GraphEncoder
        self.graph=GraphEncoder(c.width)
    def forward(self,observations,parameter,known,families):
        # Actual known inputs only. Unknown parameters are never hidden in context.
        phys=torch.stack([torch.where(known,parameter,torch.zeros_like(parameter)),known.float(),families.float(),torch.ones_like(parameter),torch.zeros_like(parameter)],-1)
        graph=self.graph(families);context=self.physical(phys)+graph
        xy=self.geometry.nodes*parameter.new_tensor(self.c.domain['lengths'])+parameter.new_tensor(self.c.domain['origins']);q=torch.cat([xy,torch.zeros(len(xy),1,device=xy.device)],-1)
        coords=Coordinates(self.c,False)(q);x=self.position(coords)[None]+context[:,None];x=x+self.sensor(x,self.geometry.nodes,observations)
        return self.processor(x,context)
