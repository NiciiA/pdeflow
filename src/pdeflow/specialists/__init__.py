"""Small explicit physical-specialist registry, separate from backbone/formulation."""
from dataclasses import asdict
from typing import Protocol
import numpy as np
import torch
from pdeflow.core import Capabilities, Conditioning, GridSpec, TransformBundle


class Specialist(Protocol):
    id: str
    fields: tuple[str, ...]
    conditioning_schema: list[str]
    encoder: str
    representation: str
    transform_type: type[TransformBundle]
    capabilities: dict
    def validate_config(self, config): ...
    def fit_transform(self, data, ids): ...
    def default_problem(self, grid): ...
    def demo(self, problem, seed): ...
    def baseline(self, problem, observations, inference): ...
    def extra_metrics(self, samples, problem): ...
    def output_diagnostics(self, state, layout, transform): ...
    def validate_observations(self, problem, observations):
        if observations.operator.channels != len(self.fields) or observations.operator.grid != problem.grid:
            raise ValueError('Observation channels/grid must match specialist physical fields')
    def validate(self, problem, formulation, stepper, schedule): ...
    def recognize(self, pairs): ...
    def request_problem(self, request, declarations): ...
    def physics(self, problem, condition, spatial='fd'): ...
    def generate(self, config, output): ...
    def load_data(self, path, config): ...
    def sample_spec(self, config, specification, rng): ...
    def solve_spec(self, config, specification): ...


class HeatSpecialist(Specialist):
    representation='affine_physical_u'
    id='heat';fields=('u',);conditioning_schema=['family','log_diffusivity'];encoder='heat_scalars';transform_type=TransformBundle
    capabilities=asdict(Capabilities())
    def validate_config(self,c):
        from pdeflow.equations import heat_problem
        self.validate(heat_problem(c.grid,.1),c.formulation,c.inference.stepper,c.inference.schedule)
        if c.grid.temporal_kind!='transient' or c.grid.convention!='periodic_nodes_endpoint_excluded' or len(c.grid.times)<3 or c.grid.times[0]!=0:raise ValueError('Heat requires nodal trajectories with >=3 physical times starting at zero')
    def validate(self,p,formulation,stepper,schedule):
        from pdeflow.equations import heat_expression
        Capabilities().validate(p,formulation,stepper,schedule)
        if p.grid.temporal_kind!='transient' or p.grid.convention!='periodic_nodes_endpoint_excluded' or len(p.grid.times)<3 or p.grid.times[0]!=0:raise ValueError('Heat requires transient periodic nodes with >=3 times starting at zero')
        if p.known_fields or p.operator_spec!=heat_expression().to_dict():raise ValueError('No compatible implemented heat residual for this operator/known-field schema')
    def recognize(self,pairs):
        from pdeflow.symbolic import Node
        u=Node('symbol',name='u');d=Node('symbol',name='D')
        heat=(Node('dτ',(u,)),Node('mul',(d,Node('lap',(u,)))))
        return len(pairs)==1 and (pairs[0]==heat or pairs[0]==(Node('sub',heat),Node('number',value=0)))
    def request_problem(self,r,d):
        from pdeflow.equations import heat_problem
        if r.get('known_fields'):raise ValueError('Heat has no known spatial-field condition; remove unsupported known_fields')
        if set(d)!={'u','D'} or d['u']['kind']!='field' or d['u']['status']!='unknown' or d['D']['kind']!='parameter' or d['D']['status']!='known':raise ValueError('Heat requires unknown scalar u and known scalar D')
        if r['initial']['status']!='unknown':raise ValueError('Heat initial field is unknown; use explicit observations for known values')
        p=heat_problem(GridSpec.from_dict(r['domain']),d['D']['value']);self.validate(p,'independent_gaussian_linear_cfm','heun','uniform');return p
    def physics(self,p,condition,spatial='fd'):
        from pdeflow.equations import make_physics
        return make_physics(p,condition.known_parameters,spatial)
    def generate(self,c,output):
        from pdeflow.data import _generate_heat
        return _generate_heat(c,output)
    def load_data(self,path,c):
        from pdeflow.data import _load_heat
        return _load_heat(path,c)
    def fit_transform(self,data,ids):return data.transform() if hasattr(data,'transform') else self.transform_type.fit(data.values[data.indices('train')],ids)
    def baseline(self,p,obs,c):
        from pdeflow.evaluation.baseline import inverse_baseline
        return {'fourier_inverse':inverse_baseline(p.grid,p.known_parameters['diffusivity'],obs,c.basis_degree,c.ridge)}
    def demo(self,p,seed):
        from pdeflow.numerics import heat_reference,smooth_initial
        return torch.from_numpy(heat_reference(smooth_initial(np.random.default_rng(seed),p.grid),p.known_parameters['diffusivity'],p.grid))[None,None]
    def extra_metrics(self,samples,p):return {}
    def default_problem(self,grid):
        from pdeflow.equations import heat_problem
        return heat_problem(grid,.07)
    def output_diagnostics(self,state,layout,transform):return {}

    def sample_spec(self,c,spec,rng):
        d=c.data
        spec['parameters']={'diffusivity':float(rng.uniform(*d.diffusivity_range))}
        spec['initial_distribution']={'bandwidth':float(rng.uniform(*(d.bandwidth_range or (d.bandwidth,d.bandwidth)))),'amplitude':float(rng.uniform(*d.amplitude_range)),'offset':float(rng.uniform(*d.offset_range))}
    def solve_spec(self,c,spec):
        from pdeflow.numerics import smooth_initial,heat_reference
        from pdeflow.equations import heat_problem
        d=spec['initial_distribution'];initial=d['amplitude']*smooth_initial(np.random.default_rng(int(spec['initial_seed'])),c.grid,d['bandwidth'])+d['offset']
        diffusivity=spec['parameters']['diffusivity']
        trajectory=heat_reference(initial,diffusivity,c.grid)
        return heat_problem(c.grid,diffusivity),trajectory[None],{'converged':True,'solver':'exact_fft_heat','mean_drift':float(np.max(np.abs(trajectory.mean(tuple(range(1,c.grid.dimension+1)))-initial.mean())))}


class DarcySpecialist(Specialist):
    representation='affine_log_a,affine_pressure'
    id='darcy';fields=('a','u');conditioning_schema=['forcing_field_f'];encoder='forcing_field'
    capabilities={'family':'darcy','dimensions':[2,3],'boundary':'periodic','gauge':'zero_mean_pressure','coefficient':'positive_spatial_scalar','representation':'log_a,u;static_T1','forcing':'known_mean_zero_cell_field','flux':'harmonic_face_FV','formulation':'independent_gaussian_linear_cfm'}
    @property
    def transform_type(self):
        from pdeflow.darcy import DarcyTransform
        return DarcyTransform
    def validate_config(self,c):
        from pdeflow.darcy import problem,default_forcing
        self.validate(problem(c.grid,default_forcing(c.grid)),c.formulation,c.inference.stepper,c.inference.schedule)
        if c.data.bandwidth_range is not None:raise ValueError('Darcy currently uses one declared correlation bandwidth, not bandwidth_range')
        if c.data.storage!='sharded_npz':raise ValueError('Darcy datasets use restartable sharded_npz')
    def validate(self,p,formulation,stepper,schedule):
        from pdeflow.darcy import operator_spec
        if p.family!=self.id or p.operator_spec!=operator_spec():raise ValueError('Darcy family/operator capability mismatch')
        if p.grid.dimension not in (2,3) or p.grid.convention!='periodic_cell_centers' or p.grid.temporal_kind!='static':raise ValueError('Darcy requires a static 2D/3D cell-centred grid')
        if p.grid.domain.boundary!='periodic' or p.grid.domain.geometry!='cartesian':raise ValueError('Darcy supports periodic Cartesian boxes only')
        if tuple(f.name for f in p.fields)!=self.fields or any(f.role!='unknown' or f.units!='nondimensional' for f in p.fields) or p.unknown_parameters or p.known_parameters:raise ValueError('Darcy jointly reconstructs unknown physical a,u only; no hidden coefficient or unknown scalar parameters')
        if p.initial_condition!='not_applicable':raise ValueError('Static Darcy has no initial condition')
        if set(p.known_fields)!={'f'}:raise ValueError('Darcy requires explicitly known forcing field f')
        f=np.asarray(p.known_fields['f'],dtype=float)
        if f.shape!=p.grid.shape or not np.isfinite(f).all() or abs(f.mean())>1e-12*max(1,float(np.abs(f).max())):raise ValueError('Forcing must be finite, match the grid and have zero mean (periodic compatibility)')
        if formulation!='independent_gaussian_linear_cfm' or stepper not in ('euler','heun') or schedule not in ('uniform','quadratic'):raise ValueError('Darcy supports linear CFM and compatible Euler/Heun ODE schedules only')
    def recognize(self,pairs):
        from pdeflow.symbolic import Node
        a=Node('symbol',name='a');u=Node('symbol',name='u');f=Node('symbol',name='f')
        lhs=Node('mul',(Node('number',value=-1),Node('div',(Node('mul',(a,Node('grad',(u,),rank=1)),rank=1),))))
        return len(pairs)==1 and (pairs[0]==(lhs,f) or pairs[0]==(Node('sub',(lhs,f)),Node('number',value=0)))
    def request_problem(self,r,d):
        from pdeflow.darcy import problem
        if set(d)!={'a','u','f'} or d['a']['kind'] not in ('field','coefficient') or d['u']['kind']!='field' or d['f']['kind']!='coefficient' or any(d[n]['status']!='unknown' for n in ('a','u')) or d['f']['status']!='known':raise ValueError('Darcy needs unknown a,u and known forcing f; known/hidden permeability is not this joint specialist')
        if r['domain']['times'] or r['initial']['status']!='not_applicable':raise ValueError('Darcy is static: remove physical times and mark initial information not applicable')
        grid={**r['domain'],'times':[0.],'temporal_kind':'static'}
        if grid.get('convention')!='periodic_cell_centers':raise ValueError('Darcy uses cell centres; set the discretization convention explicitly')
        if set(r.get('known_fields',{}))!={'f'}:raise ValueError('Supply the known mean-zero forcing field f under λ')
        p=problem(GridSpec.from_dict(grid),r['known_fields']['f']);self.validate(p,'independent_gaussian_linear_cfm','heun','uniform');return p
    def physics(self,p,condition,spatial='fd'):
        from pdeflow.darcy import DarcyPhysics
        if spatial != 'fd':raise ValueError('Darcy implements the harmonic finite-volume residual only')
        return DarcyPhysics(p.grid,condition.known_fields['f'])
    def generate(self,c,output):
        from pdeflow.darcy.data import generate
        return generate(c,output)
    def load_data(self,path,c):
        from pdeflow.darcy.data import load
        return load(path,c)
    def fit_transform(self,data,ids):return data.transform()
    def baseline(self,p,obs,c):
        from pdeflow.darcy.inverse import inverse_baseline
        return {'darcy_inverse':inverse_baseline(p,obs,c.basis_degree,c.ridge)}
    def demo(self,p,seed):
        from pdeflow.darcy.data import draw_permeability
        from pdeflow.darcy import solve
        a=draw_permeability(np.random.default_rng(seed),p.grid,.5,1.)
        u,_=solve(a,p.known_fields['f'],p.grid)
        return torch.from_numpy(np.stack((a,u)))[None,:,None]
    def default_problem(self,grid):
        from pdeflow.darcy import problem,default_forcing
        return problem(grid,default_forcing(grid))
    def output_diagnostics(self,state,layout,transform):
        pressure=layout.unpack(state)[:,1]*transform.scale[1]+transform.mean[1]
        return {'pressure_projection_removed_mean_rms':float(pressure.mean(tuple(range(1,pressure.ndim))).square().mean().sqrt()),'pressure_projection':'Subtract spatial mean on every decode, including guidance; changes generated distribution, enforces gauge only, not PDE accuracy'}
    def extra_metrics(self,samples,p):
        a,u=samples[:,0],samples[:,1];axes=tuple(range(1,u.ndim))
        return {'permeability_min':float(a.min()),'permeability_max':float(a.max()),'positivity_violations':int((a<=0).sum()),'pressure_gauge_max_abs_mean':float(u.mean(axes).abs().max()),'static_fields':True,'inverse_baseline_status':'implemented restricted Fourier implicit-adjoint BFGS; no oracle inputs', 'pressure_only_identifiability':'not guaranteed'}

    def sample_spec(self,c,spec,rng):
        spec['coefficient_distribution']={'amplitude':float(rng.uniform(*c.data.amplitude_range)),'offset':float(rng.uniform(*c.data.offset_range)),'bandwidth':c.data.bandwidth}
    def solve_spec(self,c,spec):
        from pdeflow.darcy.data import draw_permeability
        from pdeflow.darcy import problem,solve
        from pdeflow.numerics import smooth_initial
        d=spec['coefficient_distribution'];a=draw_permeability(np.random.default_rng(int(spec['initial_seed'])),c.grid,d['amplitude'],d['bandwidth'],d['offset'])
        f=smooth_initial(np.random.default_rng(int(spec['forcing_seed'])),c.grid,d['bandwidth']);f-=f.mean()
        u,diag=solve(a,f,c.grid,c.provider.solver_rtol,c.provider.solver_max_iterations)
        return problem(c.grid,f),np.stack((a,u))[:,None],diag


from .chemistry import ChemistrySpecialist

_REGISTRY={'heat':HeatSpecialist(),'darcy':DarcySpecialist(),'chemistry':ChemistrySpecialist()}

def get_specialist(name)->Specialist:
    if name not in _REGISTRY:raise ValueError(f'Unsupported specialist {name}; available: {list(_REGISTRY)}')
    return _REGISTRY[name]

def specialists():return tuple(_REGISTRY.values())
