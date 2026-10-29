"""Periodic U-Net with explicit physical-grid features and fixed-time packing."""
import torch
from torch import nn
from torch.nn import functional as F
from pdeflow.conditioning import make_condition_encoder
from . import convolution,ResidualBlock

class MultiscaleUNet(nn.Module):
    def __init__(self,layout,config,condition_kind):
        super().__init__();self.layout=layout;self.config=config;d=layout.grid.dimension;conv=convolution(d);width=config.width;cw=config.conditioning_width or width
        self.condition_encoder=make_condition_encoder(condition_kind,cw,d)
        self.time_encoder=nn.Sequential(nn.Linear(3,cw),nn.SiLU(),nn.Linear(cw,cw))
        self.condition_projection=conv(cw,width,1)
        extra=4*d+len(layout.grid.times) if config.coordinate_features else 0
        self.input=conv(layout.channels*len(layout.grid.times)+extra,width,3,padding=1,padding_mode='circular')
        widths=[width*min(2**level,4) for level in range(config.levels)]
        self.encoders=nn.ModuleList(nn.Sequential(*[ResidualBlock(conv,w) for _ in range(config.blocks)]) for w in widths)
        self.down=nn.ModuleList(conv(a,b,1) for a,b in zip(widths,widths[1:]))
        self.up=nn.ModuleList(conv(b+a,a,1) for a,b in zip(widths[:-1],widths[1:]))
        self.decoders=nn.ModuleList(nn.Sequential(*[ResidualBlock(conv,w) for _ in range(config.blocks)]) for w in widths[:-1])
        self.output=conv(width,layout.channels*len(layout.grid.times),3,padding=1,padding_mode='circular')
    def features(self,state):
        g=self.layout.grid;coordinates=torch.meshgrid(*g.coordinates(device=state.device,dtype=state.dtype),indexing='ij');rows=[]
        for x,o,L,h in zip(coordinates,g.domain.origins,g.domain.lengths,g.spacing):
            phase=2*torch.pi*(x-o)/L;rows.extend((phase.sin(),phase.cos(),torch.full_like(x,__import__('math').log(h)),torch.full_like(x,L)))
        rows.extend(torch.full_like(coordinates[0],t) for t in g.times)
        return torch.stack(rows)[None].expand(len(state),-1,*g.shape)
    def block(self,module,x):
        if self.config.activation_checkpointing and self.training and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            return checkpoint(module,x,use_reentrant=False)
        return module(x)
    def forward(self,state,generation_time,condition):
        self.layout.unpack(state)
        if generation_time.shape!=(len(state),):raise ValueError('One generation coordinate per model batch item required')
        s=generation_time;embed=self.time_encoder(torch.stack((s,torch.sin(torch.pi*s),torch.cos(torch.pi*s)),1))
        known=self.condition_encoder(condition)
        embed=embed.reshape(len(state),-1,*([1]*self.layout.grid.dimension))
        if known.ndim==2:known=known.reshape(len(state),-1,*([1]*self.layout.grid.dimension))
        h=self.input(torch.cat((state,self.features(state)),1) if self.config.coordinate_features else state)+self.condition_projection(embed+known)
        skips=[];pool={1:F.avg_pool1d,2:F.avg_pool2d,3:F.avg_pool3d}[self.layout.grid.dimension]
        for level,encoder in enumerate(self.encoders):
            h=self.block(encoder,h);skips.append(h)
            if level<len(self.down):h=self.down[level](pool(h,2))
        for level in reversed(range(len(self.up))):
            # Nearest replication of pooled model features, never resampling output fields.
            h=F.interpolate(h,size=skips[level].shape[2:],mode='nearest')
            h=self.block(self.decoders[level],self.up[level](torch.cat((h,skips[level]),1)))
        return self.output(F.silu(h))
