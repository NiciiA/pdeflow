"""Factorized physical-state flow + graph-conditioned learned dynamics, no trajectory code."""
import math
from contextvars import ContextVar
from contextlib import contextmanager
import torch
from torch import nn
from .joint import layout,options,active_mask
from .graph import GraphEncoder

_coordinate_derivatives=ContextVar('coordinate_derivatives',default=False)

@contextmanager
def derivative_attention():
    """Existing math attention, needed for higher derivatives through T on CUDA."""
    token=_coordinate_derivatives.set(True)
    try:yield
    finally:_coordinate_derivatives.reset(token)

class Coordinates(nn.Module):
    def __init__(self,c):super().__init__();self.c=c;self.size=2*c.dimension*c.frequencies+17
    def forward(self,q):
        space=(q[...,:2]-q.new_tensor(self.c.origins))/q.new_tensor(self.c.lengths)
        a=space[...,None]*2*math.pi*torch.arange(1,self.c.frequencies+1,device=q.device,dtype=q.dtype)
        t=q[...,-1:]/self.c.interval[1];b=t*2*math.pi*torch.arange(1,9,device=q.device,dtype=q.dtype)
        return torch.cat([a.sin().flatten(-2),a.cos().flatten(-2),t,b.sin(),b.cos()],-1)

class Space(nn.Module):
    def __init__(self,c):super().__init__();self.c=c;self.size=2*c.dimension*c.frequencies
    def forward(self,q):
        x=(q[...,:2]-q.new_tensor(self.c.origins))/q.new_tensor(self.c.lengths)
        a=x[...,None]*2*math.pi*torch.arange(1,self.c.frequencies+1,device=q.device,dtype=q.dtype)
        return torch.cat([a.sin().flatten(-2),a.cos().flatten(-2)],-1)

class Attention(nn.Module):
    """Math path supports second coordinate derivatives on CPU/MPS/CUDA."""
    def __init__(self,w,h):
        super().__init__();self.h=h;self.q=nn.Linear(w,w);self.k=nn.Linear(w,w,bias=False);self.v=nn.Linear(w,w);self.out=nn.Linear(w,w)
    def forward(self,q,k,valid=None,bias=None,derivatives=False):
        b,n,w=q.shape;m=k.shape[1];h=self.h
        Q=self.q(q).reshape(b,n,h,w//h).transpose(1,2);K=self.k(k).reshape(b,m,h,w//h).transpose(1,2);V=self.v(k).reshape(b,m,h,w//h).transpose(1,2)
        additive=Q.new_zeros(b,1,n,m)
        if bias is not None:additive=additive+bias[:,None]
        if valid is not None:additive=additive.masked_fill(~valid[:,None,None,:],-torch.inf)
        if derivatives or _coordinate_derivatives.get() or not self.training:
            a=(Q.float()@K.float().transpose(-1,-2)/math.sqrt(w//h)+additive.float()).softmax(-1);r=(a@V.float()).to(q.dtype)
        else:r=torch.nn.functional.scaled_dot_product_attention(Q,K,V,attn_mask=additive)
        return self.out(r.transpose(1,2).reshape(b,n,w))

class Block(nn.Module):
    def __init__(self,w,h,cross=False):
        super().__init__();self.self_attn=Attention(w,h);self.n=nn.LayerNorm(w);self.cross=Attention(w,h) if cross else None;self.n2=nn.LayerNorm(w);self.ff=nn.Sequential(nn.Linear(w,2*w),nn.SiLU(),nn.Linear(2*w,w));self.film=nn.Linear(w,2*w) if cross else None
    def forward(self,x,context=None,time=None,mask=None):
        y=self.n(x)
        if time is not None:
            scale,shift=self.film(time).chunk(2,-1);y=y*(1+scale[:,None])+shift[:,None]
        x=x+self.self_attn(y,y,mask)
        if self.cross is not None:x=x+self.cross(self.n2(x),context)
        x=x+self.ff(self.n2(x))
        return x if mask is None else x*mask[...,None]

class Geometry(nn.Module):
    def __init__(self,c):
        super().__init__();self.c=c;self.groups=layout(c)
        positions=[];ids=[];local=[]
        for j,g in enumerate(self.groups):
            positions.extend([[.5]*c.dimension]*g.globals);ids.extend([j]*g.globals);local.extend([False]*g.globals)
            if g.anchor_shape:
                a=torch.cartesian_prod(*[(torch.arange(n,device='cpu')+.5)/n for n in g.anchor_shape]).reshape(-1,c.dimension)
                positions.extend(a.tolist());ids.extend([j]*len(a));local.extend([True]*len(a))
        self.register_buffer('anchors',torch.tensor(positions));self.register_buffer('group_ids',torch.tensor(ids));self.register_buffer('local',torch.tensor(local))
        self.group_embedding=nn.Embedding(3,c.width);self.kind=nn.Embedding(2,c.width);self.position=nn.Linear(2*c.dimension,c.width)
    def embedding(self):
        a=2*math.pi*self.anchors
        return self.group_embedding(self.group_ids)+self.kind(self.local.long())+self.position(torch.cat([a.sin(),a.cos()],-1))
    def distance(self,coordinates,anchors):
        xy=(coordinates[...,:self.c.dimension]-coordinates.new_tensor(self.c.origins))/coordinates.new_tensor(self.c.lengths)
        angle=2*math.pi*(xy[...,None,:]-anchors)
        return (1-angle.cos()).sum(-1)/(2*math.pi**2)

class ObservationEncoder(nn.Module):
    def __init__(self,c):
        super().__init__();self.coords=Coordinates(c);w=c.width
        self.input=nn.Sequential(nn.Linear(self.coords.size+5,w),nn.SiLU(),nn.Linear(w,w));self.quantity=nn.Embedding(3,w)
        self.null=nn.Parameter(torch.zeros(1,1,w));self.queries=nn.Parameter(torch.randn(options(c)['observation_tokens'],w)*.02)
        self.aggregate=Attention(w,c.heads);self.layers=nn.ModuleList([Block(w,c.heads) for _ in range(options(c)['observation_depth'])])
    def forward(self,observations,quantity_ids=None,time_present=None):
        q,f,valid=observations[:3]
        if quantity_ids is None and len(observations)>3:quantity_ids=observations[3]
        # Masked padding must be inert even with explicit quantity IDs or NaNs.
        q=torch.where(valid[...,None],q,torch.zeros_like(q));f=torch.where(valid[...,None],f,torch.zeros_like(f))
        sigma=f[...,1:2].clamp_min(0);features=torch.cat([f,torch.log1p(sigma),sigma.eq(0).to(f.dtype)],-1)
        quantity_ids=torch.zeros_like(valid,dtype=torch.long) if quantity_ids is None else quantity_ids
        if time_present is not None:
            if time_present.shape!=valid.shape:raise ValueError('Time presence mask must match observations')
            q=torch.cat([q[...,:-1],torch.where(time_present,q[...,-1],torch.zeros_like(q[...,-1]))[...,None]],-1)
            features=features.clone();features[...,2]=time_present.to(features.dtype)
        points=self.input(torch.cat([self.coords(q),features],-1))+self.quantity(torch.where(valid,quantity_ids,torch.zeros_like(quantity_ids)))
        points=torch.cat([self.null.expand(len(q),-1,-1),points],1);valid=torch.cat([torch.ones(len(q),1,dtype=torch.bool,device=q.device),valid],1)
        x=self.queries[None].expand(len(q),-1,-1);x=x+self.aggregate(x,points,valid)
        for layer in self.layers:x=layer(x)
        return x


class StateEncoder(nn.Module):
    """No coefficient or cross-quantity information enters the state encoder."""
    def __init__(self,c):
        super().__init__();self.c=c;self.coords=Space(c);self.geometry=Geometry(c);w=c.width
        self.fields=nn.ModuleList([nn.Sequential(nn.Linear(self.coords.size+1,w),nn.SiLU(),nn.Linear(w,w)) for _ in range(2)])
        self.queries=nn.Parameter(torch.randn(c.tokens,w)*.02);self.aggregate=Attention(w,c.heads)
        self.processor=nn.ModuleList([Block(w,c.heads) for _ in range(options(c)['encoder_depth'])])
    def forward(self,coordinates,values,families):
        slots=[];emb=self.geometry.embedding()
        for j,g in enumerate(layout(self.c)[:2]):
            q=(self.queries[g.start:g.stop]+emb[g.start:g.stop])[None].expand(len(values),-1,-1)
            points=self.fields[j](torch.cat([self.coords(coordinates),values[...,j:j+1]],-1))
            bias=-self.geometry.distance(coordinates,self.geometry.anchors[g.start:g.stop]).transpose(1,2)*max(g.anchor_shape or (1,))**2*4
            bias=bias*self.geometry.local[g.start:g.stop][None,:,None]
            z=q+self.aggregate(q,points,bias=bias)
            for layer in self.processor:z=layer(z)
            if j==1:z=z*families[:,None,None].eq(1)
            slots.append(z)
        g=layout(self.c)[-1];slots.append(values.new_zeros(len(values),g.stop-g.start,self.c.width))
        return torch.cat(slots,1)

class Coefficient(nn.Module):
    def __init__(self,c):
        super().__init__();self.c=c;w=c.width;self.types=nn.Embedding(2,w)
        self.encoder=nn.ModuleList([nn.Sequential(nn.Linear(1,w),nn.SiLU(),nn.Linear(w,w)) for _ in range(2)])
        self.heads=nn.ModuleList([nn.Sequential(nn.LayerNorm(w),nn.Linear(w,w),nn.SiLU(),nn.Linear(w,1)) for _ in range(2)])
    def encode(self,p,families):
        return torch.stack([m(p[:,None]) for m in self.encoder],1)[torch.arange(len(p),device=p.device),families.long()]+self.types(families.long())
    def forward(self,z,p=None,known=None,physical=False,families=None):
        families=torch.zeros(len(z),dtype=torch.long,device=z.device) if families is None else families
        g=layout(self.c)[-1];h=z[:,g.start:g.stop].mean(1)
        value=torch.stack([m(h).sigmoid().squeeze(-1) for m in self.heads],1)[torch.arange(len(z),device=z.device),families.long()]
        if p is not None:value=torch.where(torch.ones_like(value,dtype=torch.bool) if known is None else known,p,value)
        if physical:
            lo=value.new_tensor([self.c.diffusivity_range[0],self.c.wave_speed_range[0]])[families.long()];hi=value.new_tensor([self.c.diffusivity_range[1],self.c.wave_speed_range[1]])[families.long()]
            value=lo*torch.exp(value*torch.log(hi/lo))
        return value

class TypedContext(nn.Module):
    def __init__(self,c):
        super().__init__();self.c=c;w=c.width;self.parameter=nn.Sequential(nn.Linear(1,w),nn.SiLU(),nn.Linear(w,w));self.types=nn.Embedding(2,w);self.status=nn.Embedding(4,w);self.domain=nn.Linear(6,w)
    def forward(self,p,known,families):
        status=torch.where(known,0,2).long();scalar=self.parameter(torch.where(known,p,torch.zeros_like(p))[:,None])+self.types(families.long())+self.status(status)
        box=p.new_tensor([*self.c.origins,*self.c.lengths,*self.c.interval]).expand(len(p),-1)
        return torch.stack([scalar,self.domain(box)],1)

class SpatialDecoder(nn.Module):
    """Space only. Smooth ALL-anchor kernel; no coefficient/trajectory/time shortcut."""
    def __init__(self,c):
        super().__init__();self.c=c;self.coords=Space(c);self.geometry=Geometry(c);w=c.width;self.query=nn.Linear(self.coords.size,w);self.quantity=nn.Embedding(2,w)
        self.global_attention=Attention(w,c.heads);self.local_attention=Attention(w,c.heads)
        self.layers=nn.ModuleList([nn.Sequential(nn.LayerNorm(w),nn.Linear(w,w),nn.SiLU(),nn.Linear(w,w)) for _ in range(options(c)['decoder_depth'])]);self.output=nn.ModuleList([nn.Linear(w,1) for _ in range(2)])
    def forward(self,z,xy,quantity=0,families=None):
        if xy.ndim==2:xy=xy[None].expand(len(z),-1,-1)
        g=layout(self.c)[quantity];tokens=z[:,g.start:g.stop]+self.geometry.embedding()[None,g.start:g.stop]
        query=self.query(self.coords(xy))+self.quantity.weight[quantity]
        h=query+self.global_attention(query,tokens[:,:g.globals],derivatives=True)
        local=tokens[:,g.globals:]
        if local.shape[1]:
            bias=-self.geometry.distance(xy,self.geometry.anchors[g.start+g.globals:g.stop])*max(g.anchor_shape)**2*4
            h=h+self.local_attention(query,local,bias=bias,derivatives=True)
        for layer in self.layers:h=h+layer(h)
        value=self.output[quantity](h).squeeze(-1)
        return value if quantity==0 or families is None else value*families[:,None].eq(1)

class Evolution(nn.Module):
    """Autonomous two-time learned operator on COMPLETE state, identity at equal times."""
    def __init__(self,c):
        super().__init__();self.c=c;w=c.width;depth=c.evolution.get('depth',max(1,c.layers//2))
        self.time=nn.Sequential(nn.Linear(35,w),nn.SiLU(),nn.Linear(w,w));self.scalar=nn.Linear(1,w);self.graph=nn.Linear(w,w);self.geometry=Geometry(c)
        self.processor=nn.ModuleList([Block(w,c.heads,True) for _ in range(depth)]);self.output=nn.Linear(w,w)
    def forward(self,state,ta,tb,parameter,families,graph):
        ta=torch.as_tensor(ta,device=state.device,dtype=state.dtype).expand(len(state));tb=torch.as_tensor(tb,device=state.device,dtype=state.dtype).expand(len(state));dt=(tb-ta)/self.c.interval[1]
        a=dt[:,None]*2*math.pi*torch.arange(1,17,device=state.device,dtype=state.dtype)
        time=self.time(torch.cat([dt[:,None],ta[:,None]/self.c.interval[1],tb[:,None]/self.c.interval[1],a.sin(),a.cos()],-1))
        context=torch.stack([self.graph(graph),self.scalar(parameter[:,None])],1);mask=active_mask(self.c,torch.zeros(len(state),dtype=torch.bool,device=state.device),families)
        h=state+self.geometry.embedding()[None];processing_mask=mask.clone();coefficient_group=layout(self.c)[-1];processing_mask[:,coefficient_group.start:]=False
        for layer in self.processor:h=layer(h,context,time,processing_mask)
        delta=self.output(h);g=layout(self.c)[-1]
        delta=torch.cat([delta[:,:g.start],torch.zeros_like(delta[:,g.start:])],1)
        return (state+dt[:,None,None]*delta)*mask[...,None]

class Autoencoder(nn.Module):
    def __init__(self,c,training_encoder=True):
        super().__init__();self.c=c;self.encoder=StateEncoder(c) if training_encoder else None;self.coefficient=Coefficient(c);self.decoder=SpatialDecoder(c);self.graph_encoder=GraphEncoder(c.width,c.evolution.get('graph_depth',2));self.evolution=Evolution(c)
    def encode_state(self,q,values,families):return self.encoder(q,values,families)
    def encode(self,q,values,parameter,families=None):
        families=torch.zeros(len(values),dtype=torch.long,device=values.device) if families is None else families
        if values.ndim==2:values=torch.stack([values,torch.zeros_like(values)],-1)
        z=self.encode_state(q,values,families);g=layout(self.c)[-1]
        return torch.cat([z[:,:g.start],self.coefficient.encode(parameter,families)[:,None].expand(-1,g.stop-g.start,-1)],1)
    def decode_parameter(self,z,parameter=None,known=None,physical=False,families=None):return self.coefficient(z,parameter,known,physical,families)
    def evolve(self,z,ta,tb,parameter=None,known=None,families=None,graph=None):
        families=torch.zeros(len(z),dtype=torch.long,device=z.device) if families is None else families
        p=self.decode_parameter(z,parameter,known,False,families);graph=self.graph_encoder(families) if graph is None else graph
        return self.evolution(z,ta,tb,p,families,graph)
    def decode_state(self,z,xy,quantity=0,families=None):return self.decoder(z,xy,quantity,families)
    def decode(self,z,queries,parameter=None,known=None,quantity='u',families=None):
        families=torch.zeros(len(z),dtype=torch.long,device=z.device) if families is None else families
        j=1 if quantity in ('v','initial_v') else 0
        if queries.ndim==3:
            return torch.cat([self.decode(z[b:b+1],queries[b],parameter[b:b+1] if parameter is not None else None,known[b:b+1] if known is not None else None,quantity,families[b:b+1]) for b in range(len(z))],0)
        if quantity in ('initial','initial_u','initial_v'):return self.decode_state(z,queries[:,:2],j,families)
        graph=self.graph_encoder(families)
        # Grouping is an efficiency path only. Derivative probes evolve at EVERY
        # query time differentiably, so u_t includes evolution's time dependence.
        if queries.requires_grad:
            n=len(queries);b=len(z);state=z[:,None].expand(-1,n,-1,-1).reshape(b*n,self.c.tokens,self.c.width)
            f=families[:,None].expand(-1,n).reshape(-1);p=parameter[:,None].expand(-1,n).reshape(-1) if parameter is not None else None;k=known[:,None].expand(-1,n).reshape(-1) if known is not None else None
            evolved=self.evolve(state,0.,queries[None,:,-1].expand(b,-1).reshape(-1),p,k,f,graph[:,None].expand(-1,n,-1).reshape(b*n,-1))
            xy=queries[None,:,:2].expand(b,-1,-1).reshape(b*n,1,2)
            return self.decode_state(evolved,xy,j,f).reshape(b,n)
        pieces=[];positions=[]
        for t in torch.unique(queries[:,-1],sorted=True):
            indices=torch.where(queries[:,-1].eq(t))[0];evolved=self.evolve(z,0.,t,parameter,known,families,graph)
            pieces.append(self.decode_state(evolved,queries[indices,:2],j,families));positions.append(indices)
        joined=torch.cat(pieces,1);order=torch.argsort(torch.cat(positions));return joined[:,order]

class Velocity(nn.Module):
    def __init__(self,c):
        super().__init__();self.c=c;w=c.width;self.geometry=Geometry(c);self.observation_encoder=ObservationEncoder(c);self.context_encoder=TypedContext(c);self.graph=nn.Linear(w,w)
        self.time=nn.Sequential(nn.Linear(3,w),nn.SiLU(),nn.Linear(w,w));self.inputs=nn.ModuleList([nn.Linear(w,w) for _ in range(3)]);self.outputs=nn.ModuleList([nn.Linear(w,w) for _ in range(3)]);self.processor=nn.ModuleList([Block(w,c.heads,True) for _ in range(c.layers)])
    def condition(self,parameter,observations,known=None,families=None,graph=None):
        if families is None:families=torch.zeros(len(parameter),dtype=torch.long,device=parameter.device)
        if known is None:known=torch.ones_like(families,dtype=torch.bool)
        if graph is None:raise ValueError('Flow requires the validated PDE graph encoding')
        context=torch.cat([self.observation_encoder(observations),self.context_encoder(parameter,known,families),self.graph(graph)[:,None]],1)
        return context,active_mask(self.c,known,families)
    def forward(self,latent,flow_time,parameter=None,observations=None,known=None,families=None,graph=None,conditioning=None):
        context,mask=conditioning if conditioning is not None else self.condition(parameter,observations,known,families,graph)
        s=flow_time.reshape(-1);time=self.time(torch.stack([s,s.sin(),s.cos()],-1));h=torch.cat([m(latent[:,g.start:g.stop]) for g,m in zip(layout(self.c),self.inputs)],1)*mask[...,None]+self.geometry.embedding()[None]
        for layer in self.processor:
            if self.training and self.c.training_options.get('activation_checkpointing',False):
                from torch.utils.checkpoint import checkpoint
                h=checkpoint(layer,h,context,time,mask,use_reentrant=False)
            else:h=layer(h,context,time,mask)
        return torch.cat([m(h[:,g.start:g.stop]) for g,m in zip(layout(self.c),self.outputs)],1)*mask[...,None]

def decode_chunks(ae,latent,queries,parameter=None,chunk_size=1024,known=None,quantity='u',families=None):
    if chunk_size<1:raise ValueError('Positive query chunk required')
    return torch.cat([ae.decode(latent,q,parameter,known,quantity,families) for q in queries.split(chunk_size)],1)
