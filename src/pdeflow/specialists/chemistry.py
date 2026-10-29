"""Concrete periodic three-species reconstruction specialist."""
from dataclasses import asdict
import numpy as np
import torch
from pdeflow.core import GridSpec,TransformBundle
from pdeflow.chemistry import PARAMETERS,ConcentrationTransform,ChemistryPhysics,problem,operator_spec,solve
from pdeflow.numerics import smooth_initial
from . import Specialist


def positive_initial(grid,seed,floor=.05,amplitude=.5,bandwidth=1.2):
    rng=np.random.default_rng(seed)
    return np.stack([floor+amplitude/(1+np.exp(-smooth_initial(rng,grid,bandwidth))) for _ in range(3)])


class ChemistrySpecialist(Specialist):
    representation='affine_sqrt_concentrations'
    id='chemistry';fields=('a','b','c');encoder='chemistry_scalars';conditioning_schema=list(PARAMETERS);transform_type=ConcentrationTransform
    capabilities={'family':'chemistry','dimensions':[2,3],'boundary':'periodic','species':['a','b','c'],'unknowns':'concentration_trajectories','known_parameters':list(PARAMETERS),'rate':'k_f*a*b-k_r*c','representation':'affine_sqrt_concentration','formulation':'independent_gaussian_linear_cfm'}
    def validate_config(self,c):
        if c.chemistry is None or c.provider is None:raise ValueError('Chemistry training needs explicit chemistry and common provider configurations')
        self.validate(self.default_problem(c.grid),c.formulation,c.inference.stepper,c.inference.schedule)
    def validate(self,p,formulation,stepper,schedule):
        g=p.grid
        if p.family!=self.id or p.operator_spec!=operator_spec():raise ValueError('Chemistry family/kinetic-law capability mismatch')
        if g.dimension not in (2,3) or g.temporal_kind!='transient' or g.convention!='periodic_nodes_endpoint_excluded' or g.domain.boundary!='periodic' or len(g.times)<3 or g.times[0]!=0:raise ValueError('Chemistry requires periodic nodal 2D/3D trajectories with >=3 physical times from zero')
        if tuple(f.name for f in p.fields)!=self.fields or any(f.role!='unknown' or f.units!='nondimensional' for f in p.fields) or p.known_fields or p.unknown_parameters or p.initial_condition!='unknown':raise ValueError('Chemistry reconstructs unknown a,b,c concentrations only; initial constraints belong in observations')
        if set(p.known_parameters)!=set(PARAMETERS) or any(not np.isfinite(v) for v in p.known_parameters.values()) or min(p.known_parameters[n] for n in PARAMETERS[:3])<=0 or min(p.known_parameters[n] for n in PARAMETERS[3:])<0:raise ValueError('Chemistry requires known positive diffusivities and nonnegative forward/reverse kinetic rates')
        if formulation!='independent_gaussian_linear_cfm' or stepper not in ('euler','heun') or schedule not in ('uniform','quadratic'):raise ValueError('Unsupported chemistry formulation/ODE combination')
    def recognize(self,pairs):
        from pdeflow.symbolic import Parser,declaration
        from pdeflow.symbolic.algebra import equation_key
        declarations={n:declaration(n,'parameter' if n in PARAMETERS else 'definition' if n=='r' else 'field') for n in (*self.fields,*PARAMETERS,'r')}
        definitions={left.name:right for left,right in pairs if left.op=='symbol' and left.name=='r'}
        equations=['a_τ=D_A lap(a)-(k_f*a*b-k_r*c)','b_τ=D_B lap(b)-(k_f*a*b-k_r*c)','c_τ=D_C lap(c)+(k_f*a*b-k_r*c)']
        expected=[Parser(e,declarations,['x','y','z'],True).equation() for e in equations]
        dynamic=[(l,r) for l,r in pairs if not (l.op=='symbol' and l.name=='r')]
        if len(dynamic)!=3 or len(pairs) not in (3,4):return False
        return frozenset(equation_key(l,r,definitions) for l,r in dynamic)==frozenset(equation_key(l,r) for l,r in expected)
    def request_problem(self,r,d):
        if set(d) not in (set((*self.fields,*PARAMETERS,'r')),set((*self.fields,*PARAMETERS))):raise ValueError('Chemistry currently uses fixed names a,b,c,D_A,D_B,D_C,k_f,k_r,r; renaming is unsupported')
        if any(d[n]['kind']!='field' or d[n]['status']!='unknown' for n in self.fields) or any(d[n]['kind']!='parameter' or d[n]['status']!='known' or d[n].get('value') is None for n in PARAMETERS) or ('r' in d and d['r']['kind']!='definition'):raise ValueError('Declare unknown species, known scalar kinetic/diffusion parameters and derived reaction r')
        if r.get('known_fields') or r['initial']['status']!='unknown':raise ValueError('Known initial species values must be supplied as observations, not hidden conditions')
        p=problem(GridSpec.from_dict(r['domain']),{n:d[n]['value'] for n in PARAMETERS});self.validate(p,'independent_gaussian_linear_cfm','heun','uniform');return p
    def sample_spec(self,c,spec,rng):
        q=c.chemistry
        spec['parameters']={n:float(rng.uniform(*(q.diffusivity_range if n in PARAMETERS[:3] else q.forward_rate_range if n=='k_f' else q.reverse_rate_range))) for n in PARAMETERS}
        spec['initial_distribution']={'floor':q.initial_floor,'amplitude':q.initial_amplitude,'bandwidth':c.data.bandwidth}
        spec['solver']={'physical_step':q.reference_step,'max_steps':c.provider.solver_max_iterations}
    def solve_spec(self,c,spec):
        d=spec['initial_distribution'];x=positive_initial(c.grid,int(spec['initial_seed']),**d)
        values,diagnostics=solve(x,spec['parameters'],c.grid,spec['solver']['physical_step'],spec['solver']['max_steps'])
        return problem(c.grid,spec['parameters']),values,diagnostics
    def physics(self,p,condition,spatial='fd'):return ChemistryPhysics(p.grid,condition,spatial)
    def generate(self,c,output):
        from pdeflow.data.providers import prepare
        return prepare(c,output,True)
    def load_data(self,path,c):raise ValueError('Chemistry uses the common physical provider; set provider mode and explicit evaluation manifests')
    def fit_transform(self,data,ids):return data.transform
    def baseline(self,p,obs,c):return {} # No chemistry inverse baseline is claimed.
    def default_problem(self,grid):return problem(grid,dict(D_A=.02,D_B=.03,D_C=.01,k_f=.7,k_r=.3))
    def demo(self,p,seed):return torch.from_numpy(solve(positive_initial(p.grid,seed),p.known_parameters,p.grid)[0])[None]
    def extra_metrics(self,samples,p):
        a,b,c=samples[:,0],samples[:,1],samples[:,2];axes=tuple(range(2,a.ndim))
        invariant_a=(a+c).sum(axes)*p.grid.cell_volume;invariant_b=(b+c).sum(axes)*p.grid.cell_volume
        total=(a+b+c).sum(axes)*p.grid.cell_volume
        return {'minimum_concentration':float(samples.min()),'positivity_violations':int((samples<0).sum()),'invariant_a_plus_c_max_drift':float((invariant_a-invariant_a[:,:1]).abs().max()),'invariant_b_plus_c_max_drift':float((invariant_b-invariant_b[:,:1]).abs().max()),'total_species_sum_max_change':float((total-total[:,:1]).abs().max()),'invariant_projection':False,'identifiability':'one sparsely observed species does not guarantee unique hidden species recovery'}
    def output_diagnostics(self,state,layout,transform):
        coordinate=TransformBundle.decode(transform,layout.unpack(state))
        return {'sqrt_coordinate_negative_fraction':float((coordinate<0).float().mean()),'positivity_parameterization':'decoded square of affine square-root coordinates; zeros exact; negative root branches fold; no stoichiometric projection'}
