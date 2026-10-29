"""Bounded linked views and local diagnostics from actual saved numerical fields."""
from functools import lru_cache
from pathlib import Path
import math
import numpy as np
from pdeflow.core import ProblemSpec
from pdeflow.runtime import read_json

@lru_cache(maxsize=2)
def saved_fields(path,stamp):
    with np.load(Path(path)/'result.npz',allow_pickle=False) as data:
        samples=data['samples'][0];w=data['weights'][0] if 'weights' in data else np.ones(len(samples))/len(samples)
        mean=np.einsum('p,pc...->c...',w,samples.astype(float))
        truth=data['reference'][0].copy() if 'reference' in data else None
    return mean,truth,samples,w

def plane(p,value,field_index,time_index,normal,point,max_side=64):
    g=p.grid;axes=[a for a in range(g.dimension) if g.dimension!=3 or a!=normal]
    v=value[field_index,time_index];v=np.take(v,point[normal],axis=normal) if g.dimension==3 else v
    strides=[max(1,math.ceil(n/max_side)) for n in v.shape];v=v[tuple(slice(None,None,k) for k in strides)]
    return {'values':v.tolist(),'shape':list(v.shape),'axes':[g.axis_names[a] for a in axes],'coordinates':{g.axis_names[a]:g.coordinates()[a][::k].tolist() for a,k in zip(axes,strides)},'normal':normal if g.dimension==3 else None,'slice_coordinate':float(g.coordinates()[normal][point[normal]]) if g.dimension==3 else None,'native_shape':list(g.shape),'display_downsampled':any(k>1 for k in strides),'physical_time':g.times[time_index] if g.temporal_kind!='static' else None,'vmin':float(v.min()),'vmax':float(v.max()),'reference':None,'error':None,'observations':[],'units':p.fields[field_index].units,'statistic':'weighted ensemble mean; display striding only'}

def local_values(p,values,time_index,point):
    g=p.grid;idx=tuple(point);t=time_index
    center=lambda c:float(values[(c,t,*idx)])
    def neighbor(c,axis,offset):
        other=list(idx);other[axis]=(other[axis]+offset)%g.shape[axis]
        return float(values[(c,t,*other)])
    lap=lambda c:sum((neighbor(c,a,1)-2*center(c)+neighbor(c,a,-1))/h**2 for a,h in enumerate(g.spacing))
    derivative=lambda c:float(np.gradient(values[(c,slice(None),*idx)],g.times,edge_order=2)[t])
    quantities={f.name:center(c) for c,f in enumerate(p.fields)}
    if p.family=='heat':
        dt=derivative(0);diff=p.known_parameters['diffusivity']*lap(0)
        return {'formula':'∂τu = DΔu','terms':{'∂τu':dt,'DΔu':diff,'residual':dt-diff},'fields':quantities,'meaning':'Second-order physical-time derivative and periodic central-difference Laplacian at the native grid point'}
    if p.family=='darcy':
        a=center(0);u=center(1);flux=[];div=0.
        for axis,h in enumerate(g.spacing):
            ap,am=neighbor(0,axis,1),neighbor(0,axis,-1);qp=-(2*a*ap/(a+ap))*(neighbor(1,axis,1)-u)/h;qm=-(2*a*am/(a+am))*(u-neighbor(1,axis,-1))/h
            flux.append({'axis':g.axis_names[axis],'positive_face_flux':qp,'negative_face_flux':qm});div+=(qp-qm)/h
        forcing=float(np.asarray(p.known_fields['f'])[idx])
        return {'formula':'q = −a∇u; ∇·q − f = 0','terms':{'forcing f':forcing,'flux divergence':div,'residual':div-forcing},'fields':quantities,'flux':flux,'meaning':'Harmonic-face conservative FV fluxes; zero-mean pressure gauge; static fields, no physical-time animation'}
    pars=p.known_parameters;rate=pars['k_f']*center(0)*center(1)-pars['k_r']*center(2);terms={'reaction r':rate}
    for c,name in enumerate(('a','b','c')):
        dt=derivative(c);diff=pars['D_'+name.upper()]*lap(c);reaction=rate if c==2 else -rate
        terms.update({f'∂τ{name}':dt,f'D_{name.upper()}Δ{name}':diff,f'{name} reaction contribution':reaction,f'{name} residual':dt-diff-reaction})
    return {'formula':'A+B ⇌ C; r=k_f ab−k_r c','terms':terms,'fields':quantities,'meaning':'Actual local mass-action reaction and native FD diffusion/physical-time terms; no inferred hidden initial condition used'}

def inspect(directory,field_index=0,time_index=0,point=None,sample_index=-1):
    directory=Path(directory);p=ProblemSpec.from_dict(read_json(directory/'problem.json'));g=p.grid
    if not 0<=field_index<len(p.fields) or not 0<=time_index<len(g.times):raise ValueError('Inspection field/time outside saved layout')
    point=point or [n//2 for n in g.shape]
    if len(point)!=g.dimension or any(type(k)!=int or not 0<=k<n for k,n in zip(point,g.shape)):raise ValueError('Inspection point requires one native index per spatial axis')
    mean,truth,population,weights=saved_fields(str(directory.resolve()),(directory/'result.npz').stat().st_mtime_ns)
    statistic='weighted ensemble mean'
    if sample_index>=0:
        if sample_index>=len(population):raise ValueError('Selected particle outside saved population')
        mean=population[sample_index].astype(float);statistic=f'individual particle {sample_index}'
    elif sample_index==-2:
        mean=np.sqrt(np.einsum('p,pc...->c...',weights,(population-mean)**2));truth=None;statistic='weighted empirical spread; no physical PDE residual'
    elif sample_index!=-1:raise ValueError('Choose weighted mean, spread, or a saved particle')
    from pdeflow.observations import ObservationSet
    from pdeflow.workbench.observations import observation_rows
    obs=ObservationSet.from_dict(read_json(directory/'observations.json'),g)
    rows=observation_rows(obs,p)
    views=[]
    for normal in range(3) if g.dimension==3 else [None]:
        data=plane(p,mean,field_index,time_index,normal,point)
        if truth is not None:
            ref=plane(p,truth,field_index,time_index,normal,point);data['reference']=ref['values'];data['error']=np.abs(np.asarray(data['values'])-np.asarray(ref['values'])).tolist();data['vmin']=min(data['vmin'],ref['vmin']);data['vmax']=max(data['vmax'],ref['vmax'])
        selected=[r for r in rows if r['field']==p.fields[field_index].name and (g.temporal_kind=='static' or r['time']==g.times[time_index]) and (g.dimension!=3 or abs(r[g.axis_names[normal]]-float(g.coordinates()[normal][point[normal]]))<1e-8)]
        data['observations']=selected[::max(1,math.ceil(len(selected)/512))]
        views.append(data)
    from pdeflow.observations import ObservationSet
    obs=ObservationSet.from_dict(read_json(directory/'observations.json'),g);matches=[]
    for k,(index,valid) in enumerate(zip(obs.operator.indices.tolist(),obs.mask[0].tolist())):
        c,t,*space=index
        if valid and c==field_index and t==time_index and space==point:
            measured=float(obs.values[0,k]);matches.append({'measured':measured,'predicted':float(mean[(c,t,*space)]),'residual':float(mean[(c,t,*space)])-measured})
    return {'point':point,'coordinate':{a:float(g.coordinates()[j][k]) for j,(a,k) in enumerate(zip(g.axis_names,point))},'field':p.fields[field_index].name,'physical_time':g.times[time_index] if g.temporal_kind!='static' else None,'native_grid':__import__('dataclasses').asdict(g),'views':views,'statistic':statistic,'diagnostics':local_values(p,mean,time_index,point) if sample_index!=-2 else {'formula':'Empirical spread','terms':{},'fields':{},'meaning':'Spread is not a physical solution or calibrated uncertainty; local PDE residual is not applicable'},'observations_at_point':matches,'reference_value':float(truth[(field_index,time_index,*point)]) if truth is not None else None,'meaning':'Local equation terms are computed on the native numerical grid. Surface/heatmap displays are bounded views.'}

def generation(directory,frame=None,particle=0,kind='current_physical',axis=2,index=0,field_index=None,time_index=None):
    directory=Path(directory);meta=read_json(directory/'generation.json')
    if frame is None:return meta
    if kind not in ('current_model','current_physical','endpoint_physical'):raise ValueError('Choose actual model state, decoded current state or endpoint estimate')
    if not 0<=frame<len(meta['frames']) or not 0<=particle<len(meta['frames'][frame]['particle_slots']):raise ValueError('Snapshot frame/particle not captured')
    c=meta['field_indices'].index(field_index) if field_index is not None else 0
    t=meta['time_indices'].index(time_index) if time_index is not None else 0
    with np.load(directory/'generation.npz',allow_pickle=False) as arrays:value=arrays[kind][frame,particle,c,t].copy()
    axes=list(meta['coordinates']);coords=meta['coordinates']
    if value.ndim==3:
        if not 0<=axis<3 or not 0<=index<value.shape[axis]:raise ValueError('Snapshot display slice outside captured grid')
        value=np.take(value,index,axis);axes.pop(axis)
    return {'values':value.tolist(),'shape':list(value.shape),'axes':axes,'coordinates':{a:coords[a] for a in axes},'vmin':float(value.min()),'vmax':float(value.max()),'reference':None,'error':None,'observations':[],'physical_time':meta['native_grid']['times'][meta['time_indices'][t]] if meta['native_grid']['temporal_kind']!='static' else None,'slice_coordinate':None,'units':'model coordinates' if kind=='current_model' else 'physical field units','statistic':kind,'frame':meta['frames'][frame],'native_grid':meta['native_grid'],'display_transfer':meta['display_transfer']}
