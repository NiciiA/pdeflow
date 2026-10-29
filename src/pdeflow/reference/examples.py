"""Visible editable examples, independent of neural checkpoints."""
from copy import deepcopy
from pdeflow.core import DomainSpec,GridSpec
from pdeflow.symbolic import presets
from .solvers import ReferenceProblem

def example(name='heat-2d',shape=None):
    family,dimension=name.rsplit('-',1);dimension=int(dimension.removesuffix('d'))
    if family not in ('heat','darcy','chemistry') or dimension not in ([1,2,3] if family=='heat' else [2,3]):raise ValueError('Examples: heat-1d/2d/3d, darcy-2d/3d, chemistry-2d/3d')
    grid=GridSpec(tuple(shape or [8]*dimension),DomainSpec((1.,)*dimension),(0.,) if family=='darcy' else (0.,.05,.1,.15),'periodic_cell_centers' if family=='darcy' else 'periodic_nodes_endpoint_excluded','static' if family=='darcy' else 'transient')
    import dataclasses
    preset=deepcopy(next(p for p in presets() if p['id']==family))
    domain=dataclasses.asdict(grid)
    if family=='darcy':domain['times']=[]
    request={'schema_version':1,'equations':preset['equations'],'declarations':preset['declarations'],'domain':domain,'boundary':{'type':'periodic','status':'known','details':'Opposite sides connect'},'initial':{'status':'not_applicable' if family=='darcy' else 'unknown','details':'Inverse targets; full reference inputs are separate'},
        'measurements':{'rows':[],'status':'unknown','noise_model':'independent_gaussian','noise_std':.03},'observation_operator':{'kind':'point_on_grid','status':'known','details':'Exact saved nodes/times'},'units':'nondimensional','preset':family}
    if family=='heat':inputs={'initial_u':{'fourier':{'offset':.2,'modes':[{'wave':[1]*dimension,'amplitude':.6,'phase':0.}]}}}
    elif family=='darcy':
        inputs={'permeability':{'expression':'exp(0.3*cos(2*pi*x))'}}
        request['known_fields']={'f':{'expression':'sin(2*pi*x)'}}
    else:inputs={'initial_a':{'expression':'.3+.1*cos(2*pi*x)'},'initial_b':{'expression':'.4+.1*sin(2*pi*x)'},'initial_c':.1}
    return ReferenceProblem(request,inputs)

def from_problem(problem,seed):
    """Compatibility shortcut: explicitly record all former demo generator inputs."""
    import numpy as np
    from dataclasses import replace
    p=problem;spec=example(f'{p.family}-{p.grid.dimension}d',p.grid.shape)
    import dataclasses
    request=spec.request;request['domain']=dataclasses.asdict(p.grid)
    if p.family=='darcy':request['domain']['times']=[]
    request['known_fields']=p.known_fields
    for declaration in request['declarations']:
        key='diffusivity' if p.family=='heat' and declaration['name']=='D' else declaration['name']
        if key in p.known_parameters:declaration['value']=p.known_parameters[key]
    if p.family=='heat':
        from pdeflow.numerics import smooth_initial
        inputs={'initial_u':smooth_initial(np.random.default_rng(seed),p.grid).tolist()}
    elif p.family=='darcy':
        from pdeflow.darcy.data import draw_permeability
        inputs={'permeability':draw_permeability(np.random.default_rng(seed),p.grid,.5,1.).tolist()}
    else:
        from pdeflow.specialists.chemistry import positive_initial
        values=positive_initial(p.grid,seed);inputs={'initial_'+n:values[i].tolist() for i,n in enumerate(('a','b','c'))}
    return replace(spec,request=request,inputs=inputs,seed=seed)
