"""Passive bounded snapshots of actual states; never a controller input."""
from dataclasses import dataclass,asdict
import math
import numpy as np
import torch

@dataclass(frozen=True)
class SnapshotSpec:
    max_frames:int=9
    max_side:int=48
    max_particles:int=4
    field_indices:tuple[int,...]=(0,)
    time_indices:tuple[int,...]=(0,)
    def validate(self):
        if not 2<=self.max_frames<=16 or not 4<=self.max_side<=128 or not 1<=self.max_particles<=4:raise ValueError('Snapshots:2–16 frames,4–128 display nodes/axis,1–4 particles')
        if not self.field_indices or len(self.field_indices)>3 or not self.time_indices or len(self.time_indices)>8:raise ValueError('Snapshots need1–3 fields and1–8 saved physical times')
        return self

class SnapshotCollector:
    def __init__(self,bundle,problem,inference):
        self.spec=SnapshotSpec(**inference.snapshots).validate();self.problem=problem;self.bundle=bundle;self.n=min(inference.particles,self.spec.max_particles)
        self.fields=tuple(self.spec.field_indices);self.times=tuple(self.spec.time_indices)
        if any(type(c)!=int or not 0<=c<len(problem.fields) for c in self.fields) or any(type(t)!=int or not 0<=t<len(problem.grid.times) for t in self.times):raise ValueError('Snapshot fields/times must belong to the current physical layout')
        self.steps=set(np.linspace(0,inference.steps,self.spec.max_frames).round().astype(int).tolist());side=min(self.spec.max_side,24) if problem.grid.dimension==3 else self.spec.max_side
        while self.spec.max_frames*self.n*len(self.fields)*len(self.times)*math.prod(min(n,side) for n in problem.grid.shape)*12>16*1024**2:side-=1
        self.strides=[math.ceil(n/side) for n in problem.grid.shape];self.frames=[];self.arrays={k:[] for k in ('current_model','current_physical','endpoint_physical')};self.calls=0
        self.parents=np.arange(inference.particles)
    def select(self,value):
        value=value[:self.n][:,self.fields][:,:,self.times]
        return value[(slice(None),)*3+tuple(slice(None,None,k) for k in self.strides)].detach().cpu().float().numpy().copy()
    def record(self,x,s,step,estimate=None,parents=None,force=False):
        if parents is not None:self.parents=self.parents[np.asarray(parents,dtype=int)]
        if step not in self.steps and not force:return
        if self.frames and self.frames[-1]['step']==step:return
        if len(self.frames)>=self.spec.max_frames:
            if not force:return
            self.frames.pop();[v.pop() for v in self.arrays.values()]
        with torch.no_grad():
            available=estimate is not None and bool(torch.isfinite(estimate).all())
            model=self.bundle.layout.unpack(x);physical=self.bundle.transform.decode(model)
            endpoint=self.bundle.transform.decode(self.bundle.layout.unpack(estimate)) if available else torch.zeros_like(physical)
            for key,value in [('current_model',model),('current_physical',physical),('endpoint_physical',endpoint)]:self.arrays[key].append(self.select(value))
        self.frames.append({'generation_position':float(s),'step':int(step),'endpoint_available':available,'particle_slots':list(range(self.n)),'parents_at_previous_snapshot':self.parents[:self.n].tolist(),'meaning':'Actual captured state; intermediate fields are not physical-time evolution or verified PDE solutions'})
        self.parents=np.arange(len(self.parents))
    def finish(self):
        g=self.problem.grid
        return {'metadata':{'schema_version':1,'spec':asdict(self.spec),'frames':self.frames,'field_indices':list(self.fields),'field_names':[self.problem.fields[c].name for c in self.fields],'time_indices':list(self.times),'native_grid':asdict(g),'display_shape':[len(range(0,n,k)) for n,k in zip(g.shape,self.strides)],'coordinates':{a:c[::k].tolist() for a,c,k in zip(g.axis_names,g.coordinates(),self.strides)},'display_transfer':'strided view of actual native fields, no synthesized intermediate states','lineage':'Parents refer to previous captured population slots. Missing parent slots are explicit gaps, not smooth paths.','diagnostic_model_evaluations':self.calls,'byte_limit':16*1024**2},'arrays':{k:np.stack(v) for k,v in self.arrays.items()}}
