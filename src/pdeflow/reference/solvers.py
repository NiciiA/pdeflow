"""Declared reference capabilities, distinct from learned reconstruction."""
from dataclasses import dataclass
from typing import Protocol
import numpy as np
from pdeflow.core import ProblemSpec
from .fields import field_array

@dataclass(frozen=True)
class ReferenceProblem:
    request: dict
    inputs: dict
    solver: str='auto'
    accuracy: dict | None=None
    seed: int=53

class ReferenceSolver(Protocol):
    identity: str
    def solve(self,problem: ProblemSpec,inputs: dict,accuracy: dict)->tuple[np.ndarray,dict]:...

def compile_problem(spec: ReferenceProblem) -> tuple[ProblemSpec,dict]:
    from pdeflow.symbolic import analyze
    from copy import deepcopy
    from pdeflow.core import GridSpec
    request=deepcopy(spec.request)
    domain=request['domain']
    grid=GridSpec.from_dict({**domain,'times':[0.]} if not domain.get('times') else domain)
    if __import__('math').prod(grid.shape)*len(grid.times)>8_388_608 or __import__('math').prod(grid.shape)>1_048_576:raise ValueError('Reference exceeds explicit spatial/time budget; reduce grid/time layout before evaluating inputs')
    request['known_fields']={name:field_array(value,grid).tolist() for name,value in request.get('known_fields',{}).items()}
    report=analyze(request)
    if not report['problem']:raise ValueError('Reference capability unavailable: '+'; '.join(report['reasons']))
    p=ProblemSpec.from_dict(report['problem'])
    if np.prod(p.grid.shape)*len(p.grid.times)*len(p.fields)>(8_388_608 if p.family=='heat' else 2_000_000):raise ValueError('Reference exceeds family field-time budget; reduce the explicit grid/time layout')
    if spec.solver not in ('auto','analytical','numerical','manufactured'):raise ValueError('Solver must be auto, analytical, numerical or manufactured')
    return p,report

class HeatSolver:
    identity='periodic_heat_v1'
    def solve(self,p,inputs,accuracy):
        from pdeflow.numerics import heat_reference
        if set(inputs)=={'initial_u','analytical_reference'}:
            from .analytical import verify
            candidate=inputs['analytical_reference']
            if not isinstance(candidate,dict) or set(candidate)!={'expression'}:raise ValueError('Analytical reference needs exactly an expression')
            values,verification=verify(p,inputs['initial_u'],candidate['expression'])
            if accuracy['selection']=='manufactured':return values[None],{'solver':'manufactured_function_v1','classification':'manufactured/unverified reference function; not a PDE solution claim','verification':verification,'forcing':'No forcing derived: forced heat is unsupported by this specialist','reference_only':True}
            if not verification['passed']:raise ValueError('Analytical reference is inconsistent: '+str(verification['measured']))
            return values[None],{'solver':'verified_heat_expression_v1','classification':verification['classification'],'converged':True,'verification':verification,'floating_point':'CPU float64','reference_only':True}
        if set(inputs)!={'initial_u'}:raise ValueError('Heat reference requires exactly initial_u; no hidden initial default')
        initial=field_array(inputs['initial_u'],p.grid)
        requested=accuracy['selection'];analytic=isinstance(inputs['initial_u'],dict) and set(inputs['initial_u'])=={'fourier'}
        if requested=='analytical' and not analytic:raise ValueError('Analytical heat solving requires an explicit finite Fourier input. Arrays/expressions use a numerical spectral interpolant, not an exact continuum claim.')
        if analytic and requested!='numerical':
            fs=inputs['initial_u']['fourier'];times=p.grid.times
            values=[]
            for t in times:
                value=np.full(p.grid.shape,float(fs.get('offset',0)))
                for mode in fs.get('modes',[]):
                    k2=sum((2*np.pi*k/L)**2 for k,L in zip(mode['wave'],p.grid.domain.lengths))
                    value+=(field_array({'fourier':{'modes':[mode]}},p.grid))*np.exp(-p.known_parameters['diffusivity']*k2*(t-times[0]))
                values.append(value)
            kind='analytic spectral evolution of an explicitly finite-mode initial condition'
            values=np.asarray(values)
        else:
            values=heat_reference(initial,p.known_parameters['diffusivity'],p.grid)
            kind='numerical spectral interpolant evolution from supplied grid data'
        return values[None],{'solver':self.identity,'classification':kind,'converged':True,'floating_point':'CPU float64','spatial':'periodic Fourier; no duplicated endpoint','time':'exponential mode evolution at actual saved times','mean_drift':float(np.ptp(values.mean(tuple(range(1,values.ndim))))),'accuracy':accuracy}

class DarcySolver:
    identity='harmonic_fv_projected_cg_v1'
    def solve(self,p,inputs,accuracy):
        from pdeflow.darcy import solve
        if accuracy['selection']=='analytical':raise ValueError('No analytical variable-permeability Darcy reference is implemented; select Numerical or Auto')
        if set(inputs)!={'permeability'}:raise ValueError('Darcy reference requires explicit permeability; forcing f comes from the physical request')
        a=field_array(inputs['permeability'],p.grid)
        u,diagnostics=solve(a,p.known_fields['f'],p.grid,accuracy.get('rtol',1e-10),accuracy.get('max_iterations',4000))
        return np.stack((a,u))[:,None],{**diagnostics,'solver':self.identity,'classification':'numerical conservative finite-volume reference','boundary':'periodic','gauge':'zero spatial mean pressure','accuracy':accuracy}

class ChemistrySolver:
    identity='positive_uniformization_strang_exact_reaction_v1'
    def solve(self,p,inputs,accuracy):
        from pdeflow.chemistry import solve
        if accuracy['selection']=='analytical':raise ValueError('No analytical reaction–diffusion reference is implemented; select Numerical or Auto')
        if set(inputs)!={'initial_a','initial_b','initial_c'}:raise ValueError('Chemistry reference requires all three explicit initial species fields')
        initial=np.stack([field_array(inputs['initial_'+n],p.grid) for n in ('a','b','c')])
        values,diagnostics=solve(initial,p.known_parameters,p.grid,accuracy.get('max_step',.0025),accuracy.get('max_iterations',4000))
        return values,{**diagnostics,'solver':self.identity,'classification':'numerical positive Strang split FD diffusion/exact local reaction','accuracy':accuracy,'boundary':'periodic'}

SOLVERS={'heat':HeatSolver(),'darcy':DarcySolver(),'chemistry':ChemistrySolver()}

def solve(spec: ReferenceProblem):
    p,report=compile_problem(spec)
    if spec.solver=='manufactured' and p.family!='heat':raise ValueError('Manufactured expression references currently support scalar heat only; use the existing numerical solver for this family')
    accuracy=dict(spec.accuracy or {})
    allowed={'heat':set(),'darcy':{'rtol','max_iterations'},'chemistry':{'max_step','max_iterations'}}[p.family]
    if set(accuracy)-allowed:raise ValueError(f'Accuracy options for {p.family}: {sorted(allowed)}')
    for key,value in accuracy.items():
        if not np.isfinite(value) or value<=0:raise ValueError('Reference accuracy values must be finite positive')
    if 'max_iterations' in accuracy and (type(accuracy['max_iterations'])!=int or accuracy['max_iterations']>100000):raise ValueError('max_iterations must be an integer <=100000')
    accuracy['selection']=spec.solver
    values,diagnostics=SOLVERS[p.family].solve(p,spec.inputs,accuracy)
    if values.shape!=(len(p.fields),len(p.grid.times),*p.grid.shape) or not np.isfinite(values).all():raise ValueError('Invalid reference solution layout/values')
    return p,values,{'canonical_equation':report['canonical_source'],**diagnostics}

def capabilities():
    return [{'specialist':name,'solver':solver.identity,'dimensions':[1,2,3] if name=='heat' else [2,3],'boundary':'periodic Cartesian','analytical':'explicit finite Fourier modes only' if name=='heat' else 'unavailable','required_inputs':['initial_u'] if name=='heat' else ['permeability'] if name=='darcy' else ['initial_a','initial_b','initial_c']} for name,solver in SOLVERS.items()]
