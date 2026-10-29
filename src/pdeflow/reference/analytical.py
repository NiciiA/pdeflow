"""Editable periodic heat examples and independent numerical formula verification."""
from dataclasses import replace
import numpy as np
from .fields import field_array


def normalize_expression(source):
    import re
    source=source.replace('\\left','').replace('\\right','').replace('\\cdot','*').replace('\\pi','pi').replace('π','pi').replace('τ','t').replace('^','**').replace('²','**2')
    source=re.sub(r'\\(sin|cos|exp|sqrt|tanh)\b',r'\1',source)
    source=source.replace('{','(').replace('}',')')
    source=re.sub(r'(?<=\d)(?=pi\b|[xyzDt]\b|\()', '*',source)
    source=re.sub(r'\b(sin|cos|exp|sqrt|tanh)\s+\(',r'\1(',source)
    source=re.sub(r'(?<=[A-Za-z0-9_)])\s+(?=[A-Za-z0-9_(])','*',source)
    source=re.sub(r'(?<=\))(?=[A-Za-z(])','*',source)
    return source


def templates(problem):
    if problem.family!='heat':return []
    g=problem.grid;axes=g.axis_names
    modes=[f'cos(2*pi*({a}-({o}))/{L})' for a,o,L in zip(axes,g.domain.origins,g.domain.lengths)]
    product='*'.join(modes);k=sum(1/L**2 for L in g.domain.lengths)
    initial=[('product',product,4*k,'Product of axis modes'),('single-axis',modes[0],4/g.domain.lengths[0]**2,'Single axis mode')]
    rows=[]
    for name,expr,factor,label in initial:
        rows.append({'id':name,'label':label,'initial_expression':expr,'reference_expression':f'exp(-({factor})*pi**2*D*t)*({expr})','distribution':'Smooth low-mode diagnostic; exact probability under random training generator is unmeasured'})
    one=rows[1];two=modes[0].replace('2*pi','4*pi');decay=16/g.domain.lengths[0]**2
    rows.append({'id':'multi-mode','label':'Two-mode superposition','initial_expression':f'0.6*({product})+0.25*({two})','reference_expression':f'0.6*({rows[0]["reference_expression"]})+0.25*exp(-({decay})*pi**2*D*t)*({two})','distribution':'Low-mode diagnostic; not a claim of exact in-distribution probability'})
    high=modes[0].replace('2*pi','6*pi')
    if min(g.shape)>6:rows.append({'id':'challenge','label':'Challenge: higher spatial frequency','initial_expression':high,'reference_expression':f'exp(-({36/g.domain.lengths[0]**2})*pi**2*D*t)*({high})','distribution':'Explicit distribution-shift challenge; performance unmeasured'})
    return [{**r,'domain':__import__('dataclasses').asdict(g),'decay_note':'Diffusion reduces each mode by exp(-D |k|² t). Saved times are preserved; reference norms and absolute errors matter after decay.','roles':{'initial_expression':'Reference-only u(x,0)','reference_expression':'Complete evaluation-only u_ref(x,t)','unknown':'u trajectory reconstructed from disclosed measurements','D':'Known diffusivity, not a generation or guidance parameter'}} for r in rows]


def verify(problem,initial,expression):
    """Check candidate with independent small-step coordinate/time differences.

    Numerical consistency is not a general analytic proof; boundaries compare
    values and first normal derivatives at translated periodic points.
    """
    if problem.family!='heat' or problem.grid.times[0]!=0:raise ValueError('Analytical candidates currently require periodic heat with physical time starting at zero')
    g=problem.grid;D=problem.known_parameters['diffusivity'];source=normalize_expression(expression)
    def evaluate(grid,t):return field_array({'expression':source},grid,{'D':D,'t':t})
    if isinstance(initial,dict) and set(initial)=={'expression'}:initial={'expression':normalize_expression(initial['expression'])}
    u0=field_array(initial,g);values=np.stack([evaluate(g,t) for t in g.times])
    norm=max(float(np.max(np.abs(values))),1.)
    initial_error=float(np.max(np.abs(values[0]-u0)))
    max_residual=0.;boundary=0.;boundary_gradient=0.
    # Numerical derivative offsets independent of the discrete native FD residual.
    ht=min(1e-5,min(np.diff(g.times))/1000)
    for t,u in zip(g.times,values):
        du=(evaluate(g,t+ht)-evaluate(g,t-ht))/(2*ht);lap=np.zeros_like(u)
        for axis,L in enumerate(g.domain.lengths):
            h=L*1e-4;origin=list(g.domain.origins)
            def shifted(offset):
                o=origin.copy();o[axis]+=offset
                return replace(g,domain=replace(g.domain,origins=tuple(o)))
            plus=evaluate(shifted(h),t);minus=evaluate(shifted(-h),t)
            lap+=(plus-2*u+minus)/h**2
            translated=evaluate(shifted(L),t)
            boundary=max(boundary,float(np.max(np.abs(translated-u))))
            grad=(plus-minus)/(2*h)
            translated_grad=(evaluate(shifted(L+h),t)-evaluate(shifted(L-h),t))/(2*h)
            boundary_gradient=max(boundary_gradient,float(np.max(np.abs(translated_grad-grad))))
        max_residual=max(max_residual,float(np.max(np.abs(du-D*lap))))
    tolerances={'initial_max_abs':1e-8*norm,'periodic_value_max_abs':1e-8*norm,'periodic_gradient_max_abs':1e-6*norm,'PDE_max_abs':2e-4*norm}
    measured={'initial_max_abs':initial_error,'periodic_value_max_abs':boundary,'periodic_gradient_max_abs':boundary_gradient,'PDE_max_abs':max_residual}
    passed=all(measured[k]<=v for k,v in tolerances.items())
    template=next((r['id'] for r in templates(problem) if normalize_expression(r['reference_expression'])==source and isinstance(initial,dict) and normalize_expression(initial.get('expression',''))==normalize_expression(r['initial_expression'])),None)
    report={'passed':passed,'classification':'verified analytic template (numerically checked)' if template and passed else 'user-supplied candidate; numerical consistency check' if passed else 'inconsistent user-supplied candidate','template_id':template,'measured':measured,'tolerances':tolerances,'method':'CPU float64 independent coordinate/time central differences and periodic value/gradient shifts; finite test, not symbolic proof','physical_time_reference_norms':[float(np.sqrt(np.mean(v*v))) for v in values]}
    return values,report
