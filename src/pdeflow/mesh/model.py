"""M1–M5 share physical node states; none requires an oracle latent encoder."""
import math
import torch
from torch import nn
from .backbone import Coordinates,Conditioner,Processor
from .geometry import Geometry,periodic_features
from .state import pack,enforce,active_mask


class StateHead(nn.Module):
    def __init__(self,c,stochastic=False):
        super().__init__();self.c=c;self.processor=Processor(c);self.norm=nn.LayerNorm(c.width);self.field=nn.Linear(c.width,2);self.parameter=nn.Linear(c.width,1);self.stochastic=stochastic
        if stochastic:
            self.modulate=nn.ModuleList([nn.Linear(c.noise_dim,2*c.width,bias=False) for _ in range(c.depth)])
            self.local=nn.Linear(c.local_noise_dim,c.width,bias=False)
    def forward(self,context,parameter,known,families,noise=None):
        nodes,memory=context;modulations=None
        if self.stochastic:
            if noise is None:raise ValueError('FGN requires explicit global/local noise, zero for its deterministic branch')
            global_noise,local_noise=noise
            if global_noise.shape!=(len(nodes),self.c.noise_dim) or local_noise.shape!=(len(nodes),self.c.nodes,self.c.local_noise_dim):raise ValueError('Wrong explicit FGN noise shape')
            nodes=nodes+self.local(local_noise);modulations=[tuple(x[:,None] for x in layer(global_noise).chunk(2,-1)) for layer in self.modulate]
        features,global_features=self.processor(nodes,memory.mean(1),modulations)
        state=pack(self.field(self.norm(features)),self.parameter(self.norm(global_features).mean(1)).squeeze(-1))
        return enforce(self.c,state,parameter,known,families)


class ConditionalStateNetwork(nn.Module):
    """Vector output means velocity in M4, VP-v prediction in M5; objective/sampler differ."""
    def __init__(self,c):
        super().__init__();self.c=c;self.input=nn.Linear(2,c.width);self.time=nn.Sequential(nn.Linear(10,c.width),nn.SiLU(),nn.Linear(c.width,c.width));self.processor=Processor(c);self.norm=nn.LayerNorm(c.width);self.fields=nn.Linear(c.width,2);self.scalar=nn.Linear(c.width,1)
    def forward(self,state,s,context,known,families):
        nodes,memory=context;phase=2*math.pi*s[:,None]*s.new_tensor([1.,2.,4.,8.]);t=torch.cat([s[:,None],phase.sin(),phase.cos(),state[:,-1,:1]],-1);time=self.time(t)
        x=self.input(state[:,:self.c.nodes])+nodes+time[:,None];x,m=self.processor(x,memory.mean(1)+time)
        result=pack(self.fields(self.norm(x)),self.scalar(self.norm(m).mean(1)).squeeze(-1));return result*active_mask(self.c,known,families)


class Evolution(nn.Module):
    """Direct physical-time operator, complete wave state, exact initial identity."""
    def __init__(self,c):
        super().__init__();self.c=c;self.input=nn.Linear(2,c.width);self.context=nn.Sequential(nn.Linear(11,c.width),nn.SiLU(),nn.Linear(c.width,c.width));self.family=nn.Embedding(2,c.width);self.processor=Processor(c);self.adapters=nn.ModuleList([nn.Sequential(nn.LayerNorm(c.width),nn.Linear(c.width,c.width//2),nn.SiLU(),nn.Linear(c.width//2,2)) for _ in range(2)])
    def forward(self,state,time,families):
        if not torch.is_tensor(time):time=state.new_full((len(state),),float(time))
        elif time.ndim==0:time=time.expand(len(state))
        tau=(time-self.c.interval[0])/(self.c.interval[1]-self.c.interval[0]);phase=2*math.pi*tau[:,None]*tau.new_tensor([1.,2.,4.,8.]);physical=torch.cat([tau[:,None],phase.sin(),phase.cos(),state[:,-1,:]],-1);context=self.context(physical)+self.family(families)
        x=self.input(state[:,:self.c.nodes])+context[:,None];x,_=self.processor(x,context)
        heat=self.adapters[0](x);wave=self.adapters[1](x);delta=torch.where(families[:,None,None].eq(0),heat,wave);values=state[:,:self.c.nodes]+tau[:,None,None]*delta
        values=torch.stack([values[...,0],values[...,1]*families[:,None].eq(1)],-1)
        return torch.cat([values,state[:,-1:]],1)


class Decoder(nn.Module):
    """Smooth local periodic kernel plus learned state-only coordinate residual.

    No observation/context shortcut. All-node kernel logits are query-chunked;
    locality is smooth, so physical derivatives do not cross kNN selection seams.
    """
    def __init__(self,c):
        super().__init__();self.c=c;self.geometry=Geometry(c);self.coords=Coordinates(c,False);self.input=nn.Sequential(nn.Linear(4,c.width),nn.SiLU(),nn.Linear(c.width,c.width));self.residual=nn.Sequential(nn.Linear(c.width+16,c.width),nn.SiLU(),nn.Linear(c.width,2));self.log_sharpness=nn.Parameter(torch.tensor(math.log(2*min(c.node_shape)**2/math.pi**2)))
        nn.init.zeros_(self.residual[-1].weight);nn.init.zeros_(self.residual[-1].bias)
    def forward(self,state,xy,families,chunk=None):
        if xy.ndim==2:xy=xy[None].expand(len(state),-1,-1)
        node=state[:,:self.c.nodes];static=state[:,-1,0][:,None,None].expand(-1,self.c.nodes,1);family=families[:,None,None].expand(-1,self.c.nodes,1).to(state.dtype);features=self.input(torch.cat([node,static,family],-1));pieces=[]
        for start in range(0,xy.shape[1],chunk or self.c.query_chunk):
            q=xy[:,start:start+(chunk or self.c.query_chunk)];unit=(q-q.new_tensor(self.c.domain['origins']))/q.new_tensor(self.c.domain['lengths']);angle=2*math.pi*(unit[:,:,None]-self.geometry.nodes[None,None]);distance=(1-angle.cos()).sum(-1)
            weights=(-self.log_sharpness.exp()*distance.float()).softmax(-1);interpolated=weights@node.float();learned=self.residual(torch.cat([(weights@features.float()).to(q.dtype),self.coords(q)],-1)).float();values=interpolated+learned;values=torch.stack([values[...,0],values[...,1]*families[:,None].eq(1)],-1);pieces.append(values)
        return torch.cat(pieces,1)


class MeshModel(nn.Module):
    def __init__(self,c):
        super().__init__();self.c=c;self.conditioner=Conditioner(c);self.decoder=Decoder(c);self.evolution=Evolution(c) if c.kind['trajectory'] else None
        self.state_head=StateHead(c,c.architecture=='mesh_m3_fgn_v1') if c.architecture in ('mesh_m1_v1','mesh_m2_v1','mesh_m3_fgn_v1') else None
        self.generative=ConditionalStateNetwork(c) if c.architecture in ('mesh_m4_flow_v1','mesh_m5_diffusion_v1') else None
    def condition(self,observations,parameter,known,families):return self.conditioner(observations,parameter,known,families)
    def initial(self,context,parameter,known,families,noise=None):
        if self.state_head is None:raise ValueError('Iterative candidates require their declared flow/diffusion sampler')
        return self.state_head(context,parameter,known,families,noise)
    def evolve(self,state,time,families):
        if self.evolution is None:
            if torch.is_tensor(time):valid=bool(time.eq(self.c.interval[0]).all())
            else:valid=float(time)==self.c.interval[0]
            if not valid:raise ValueError('M1 reconstructs the initial state only; choose M2–M5 for physical-time evolution')
            return state
        return self.evolution(state,time,families)
    def decode(self,state,xy,families,time=0.,chunk=None):return self.decoder(self.evolve(state,time,families),xy,families,chunk)
    def query(self,state,points,families):
        """All quantities, bounded expansion; shared grid times evolve only once.

        Differentiable time queries retain independent coordinates through the
        original bounded path. Detached grouping is never used for u_t residuals.
        """
        shared=points.ndim==2
        if shared and not points.requires_grad and len(points)>2*self.c.evolution_batch:
            times=points[:,-1].detach().cpu();unique=torch.unique(times)
            if len(unique)*4<len(points):
                indices=[];values=[]
                for t in unique:
                    ix=torch.where(times.eq(t))[0].to(points.device);evolved=self.evolve(state,float(t),families);values.append(self.decoder(evolved,points[ix,:2],families));indices.append(ix)
                order=torch.cat(indices);return torch.cat(values,1)[:,order.argsort()]
        if points.ndim==2:points=points[None].expand(len(state),-1,-1)
        b,n,_=points.shape;owner=torch.arange(b,device=state.device)[:,None].expand(-1,n).reshape(-1);q=points.reshape(-1,3);out=[]
        for start in range(0,len(q),self.c.evolution_batch):
            stop=start+self.c.evolution_batch;ix=owner[start:stop];part=q[start:stop];evolved=self.evolve(state[ix],part[:,-1],families[ix]);out.append(self.decoder(evolved,part[:,:2,None].transpose(1,2),families[ix]).squeeze(1))
        return torch.cat(out).reshape(b,n,2)
    def decode_points(self,state,points,quantities,families):
        values=self.query(state,points,families)
        if quantities.ndim==1:quantities=quantities[None].expand(len(state),-1)
        return values.gather(-1,quantities[...,None]).squeeze(-1)
