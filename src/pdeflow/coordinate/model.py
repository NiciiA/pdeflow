"""Set encoder, fixed latent processor and coordinate decoder. No positional set order."""
import math
import torch
from torch import nn

class Coordinates(nn.Module):
    def __init__(self,c):
        super().__init__();self.c=c;self.size=2*c.dimension*c.frequencies+3
    def forward(self,q):
        c=self.c;space=(q[...,:c.dimension]-q.new_tensor(c.origins))/q.new_tensor(c.lengths)
        angle=space[...,None]*(2*math.pi)*torch.arange(1,c.frequencies+1,device=q.device,dtype=q.dtype)
        tau=(q[...,-1:]-c.interval[0])/(c.interval[-1]-c.interval[0])
        return torch.cat([angle.sin().flatten(-2),angle.cos().flatten(-2),tau,tau.square(),torch.sin(math.pi*tau)],-1)

class Autoencoder(nn.Module):
    def __init__(self,c):
        super().__init__();self.coords=Coordinates(c);w=c.width
        self.reference=nn.Sequential(nn.Linear(self.coords.size+2,w),nn.GELU(),nn.Linear(w,w))
        self.latent_queries=nn.Parameter(torch.randn(c.tokens,w)*.05)
        self.encoder=nn.MultiheadAttention(w,c.heads,batch_first=True,dropout=0.)
        self.query=nn.Sequential(nn.Linear(self.coords.size+1,w),nn.GELU(),nn.Linear(w,w))
        self.decoder=nn.MultiheadAttention(w,c.heads,batch_first=True,dropout=0.)
        self.output=nn.Sequential(nn.Linear(2*w,w),nn.GELU(),nn.Linear(w,1))
    def encode(self,coordinates,values,parameter):
        p=parameter[:,None,None].expand(-1,values.shape[1],1)
        points=self.reference(torch.cat([self.coords(coordinates),values[...,None],p],-1))
        queries=self.latent_queries[None].expand(len(values),-1,-1)
        return self.encoder(queries,points,points,need_weights=False)[0]+queries
    def decode(self,latent,queries,parameter):
        if queries.ndim==2:queries=queries[None].expand(len(latent),-1,-1)
        q=self.query(torch.cat([self.coords(queries),parameter[:,None,None].expand(-1,queries.shape[1],1)],-1))
        # Training can use SDPA. Evaluation retains the math attention path for
        # second spatial derivatives in the physical diagnostics/guidance.
        context=self.decoder(q,latent,latent,need_weights=not self.training)[0]
        return self.output(torch.cat([q,context],-1)).squeeze(-1)

class Velocity(nn.Module):
    def __init__(self,c):
        super().__init__();self.coords=Coordinates(c);w=c.width
        self.obs=nn.Sequential(nn.Linear(self.coords.size+3,w),nn.GELU(),nn.Linear(w,w))
        self.null=nn.Parameter(torch.zeros(1,1,w))
        self.known=nn.Sequential(nn.Linear(5,w),nn.SiLU(),nn.Linear(w,w))
        self.processor=nn.TransformerEncoder(nn.TransformerEncoderLayer(w,c.heads,2*w,dropout=0.,batch_first=True,activation='gelu'),c.layers,enable_nested_tensor=False)
        self.cross=nn.MultiheadAttention(w,c.heads,batch_first=True,dropout=0.)
        self.output=nn.Sequential(nn.LayerNorm(w),nn.Linear(w,w))
    def forward(self,latent,flow_time,parameter,observations):
        q,f,valid=observations
        # Replace masked features/coordinates before embedding; padding cannot leak NaNs or content.
        q=torch.where(valid[...,None],q,torch.zeros_like(q));f=torch.where(valid[...,None],f,torch.zeros_like(f))
        obs=self.obs(torch.cat([self.coords(q),f],-1));obs=torch.cat([self.null.expand(len(latent),-1,-1),obs],1)
        mask=torch.cat([torch.ones(len(latent),1,dtype=torch.bool,device=latent.device),valid],1)
        s=flow_time.reshape(-1);p=parameter.reshape(-1)
        condition=self.known(torch.stack([s,s.sin(),s.cos(),p,p.square()],-1))
        h=latent+condition[:,None]
        if self.training and self.coords.c.training_options.get('activation_checkpointing',False):
            from torch.utils.checkpoint import checkpoint
            for layer in self.processor.layers:h=checkpoint(layer,h,use_reentrant=False,preserve_rng_state=True)
        else:h=self.processor(h)
        h=h+self.cross(h,obs,obs,key_padding_mask=~mask,need_weights=False)[0]
        return self.output(h)


def decode_chunks(ae,latent,queries,parameter,chunk_size=1024):
    if type(chunk_size)!=int or not 1<=chunk_size<=65536:raise ValueError('Query chunk size must be1..65536')
    return torch.cat([ae.decode(latent,queries[i:i+chunk_size],parameter) for i in range(0,len(queries),chunk_size)],1)
