"""Bounded allowlisted scalar spatial fields. No Python evaluation."""
import ast
import math
import numpy as np
from pdeflow.core import GridSpec

FUNCTIONS={'sin':np.sin,'cos':np.cos,'exp':np.exp,'sqrt':np.sqrt,'tanh':np.tanh,'abs':np.abs}
OPERATORS={ast.Add:np.add,ast.Sub:np.subtract,ast.Mult:np.multiply,ast.Div:np.divide}

def field_array(value: dict | list | float, grid: GridSpec, parameters=None) -> np.ndarray:
    """Coordinates are physical x/y/z; expressions must be explicitly periodic."""
    if isinstance(value,dict):
        if set(value)=={'expression'}:
            source=value['expression']
            if not isinstance(source,str) or len(source)>2000:raise ValueError('Field expression must be <=2000 characters')
            try:tree=ast.parse(source,mode='eval')
            except SyntaxError as error:raise ValueError('Invalid field expression syntax') from error
            if len(list(ast.walk(tree)))>200:raise ValueError('Field expression exceeds200 AST nodes')
            coordinates=np.meshgrid(*[c.numpy() for c in grid.coordinates()],indexing='ij')
            names=dict(zip(grid.axis_names,coordinates));names['pi']=np.pi
            if parameters:
                if set(parameters)-{'D','t'} or any(not math.isfinite(v) for v in parameters.values()):raise ValueError('Only finite D and physical t are allowed as expression parameters')
                names.update(parameters)
            def visit(node):
                if isinstance(node,ast.Expression):return visit(node.body)
                if isinstance(node,ast.Constant) and type(node.value) in (int,float) and math.isfinite(node.value):return float(node.value)
                if isinstance(node,ast.Name) and node.id in names:return names[node.id]
                if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.USub,ast.UAdd)):return (-1 if isinstance(node.op,ast.USub) else 1)*visit(node.operand)
                if isinstance(node,ast.BinOp):
                    if type(node.op) in OPERATORS:return OPERATORS[type(node.op)](visit(node.left),visit(node.right))
                    if isinstance(node.op,ast.Pow) and isinstance(node.right,ast.Constant) and type(node.right.value) in (int,float) and -8<=node.right.value<=8:return np.power(visit(node.left),node.right.value)
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in FUNCTIONS and len(node.args)==1 and not node.keywords:return FUNCTIONS[node.func.id](visit(node.args[0]))
                raise ValueError('Unsupported field expression: use coordinates, pi, arithmetic, powers[-8,8], sin/cos/exp/sqrt/tanh/abs')
            with np.errstate(all='ignore'):value=visit(tree)
        elif set(value)=={'fourier'}:
            spec=value['fourier']
            if set(spec)-{'offset','modes'} or len(spec.get('modes',[]))>64:raise ValueError('Finite Fourier input supports offset and <=64 modes')
            x=np.meshgrid(*[c.numpy() for c in grid.coordinates()],indexing='ij');value=np.full(grid.shape,float(spec.get('offset',0)))
            for mode in spec.get('modes',[]):
                if set(mode)-{'wave','amplitude','phase'}:raise ValueError('Unknown Fourier mode property')
                wave=mode['wave']
                if len(wave)!=grid.dimension or any(type(k)!=int or abs(k)>=n/2 for k,n in zip(wave,grid.shape)):raise ValueError('Fourier modes need integer wave numbers strictly below Nyquist in every axis')
                phase=float(mode.get('phase',0))+sum(2*np.pi*k*(a-o)/length for k,a,o,length in zip(wave,x,grid.domain.origins,grid.domain.lengths))
                value+=float(mode['amplitude'])*np.cos(phase)
        else:raise ValueError('Field input is an array, scalar, expression or explicit finite Fourier specification')
    array=np.asarray(value,dtype=np.float64)
    if array.ndim==0:array=np.full(grid.shape,float(array))
    if array.shape!=grid.shape or not np.isfinite(array).all():raise ValueError(f'Field must be finite and have spatial shape{grid.shape}')
    return array
