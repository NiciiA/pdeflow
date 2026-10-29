"""Bounded polynomial normalization for the implemented equation templates only."""
from fractions import Fraction

def polynomial(node,definitions=None,depth=0):
    definitions=definitions or {}
    if depth>32:raise ValueError('Equation expansion depth exceeded')
    if node.op=='symbol' and node.name in definitions:return polynomial(definitions[node.name],definitions,depth+1)
    if node.op=='number':return {():Fraction(node.value)} if node.value else {}
    if node.op in ('add','sub','mul','power'):
        a=polynomial(node.args[0],definitions,depth+1)
        if node.op=='power':
            n=int(node.args[1].value)
            result={():Fraction(1)}
            for _ in range(n):result=multiply(result,a)
            return result
        b=polynomial(node.args[1],definitions,depth+1)
        if node.op=='mul':return multiply(a,b)
        result=a.copy()
        for k,v in b.items():result[k]=result.get(k,0)+(v if node.op=='add' else -v)
        return {k:v for k,v in result.items() if v}
    # Derivatives and named fields are atoms; no general differential identities.
    atom=(node.op,node.name,tuple(atom_key(a) for a in node.args))
    return {(repr(atom),):Fraction(1)}

def atom_key(node):return (node.op,node.name,node.value,tuple(atom_key(a) for a in node.args))

def multiply(a,b):
    if len(a)*len(b)>128:raise ValueError('Equation expansion exceeds 128 products')
    result={}
    for x,v in a.items():
        for y,w in b.items():
            key=tuple(sorted(x+y));result[key]=result.get(key,0)+v*w
    return {k:v for k,v in result.items() if v}

def equation_key(left,right,definitions=None):
    from pdeflow.symbolic import Node
    p=polynomial(Node('sub',(left,right)),definitions)
    if not p:return ()
    scale=p[sorted(p)[0]]
    return tuple((k,v/scale) for k,v in sorted(p.items()))
