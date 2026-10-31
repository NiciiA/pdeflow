"""Bounded typed directed operator graphs, safe canonical parsing and message passing."""
from dataclasses import dataclass,asdict
import ast
import torch
from torch import nn

VOCAB=('field_u','field_v','scalar_D','scalar_c','coordinate_x','coordinate_y','coordinate_t','dt1','dt2','lap2','mul','power2','residual_equal')
GRAPH_VERSION='typed_periodic_dynamic_graph_v1'
@dataclass(frozen=True)
class Node:
    op:str
    signature:str
    units:tuple # exponents: field U, length L, time T
@dataclass(frozen=True)
class PDEGraph:
    nodes:tuple
    edges:tuple # (operand,operation,argument_position)
    family:str
    def validate(self):
        for n in self.nodes:
            if n.op not in VOCAB or n.signature not in ('scalar','field','coordinate'):raise ValueError('Unsupported graph node/type')
        args={}
        for a,b,pos in self.edges:
            if not 0<=a<len(self.nodes) or not 0<=b<len(self.nodes) or pos<0:raise ValueError('Invalid directed argument edge')
            if (b,pos) in args:raise ValueError('Duplicate argument position')
            args[b,pos]=a
        for i,n in enumerate(self.nodes):
            arity={'dt1':1,'dt2':1,'lap2':1,'power2':1,'mul':2,'residual_equal':2}.get(n.op,0)
            children=[self.nodes[args[i,j]] for j in range(arity) if (i,j) in args]
            if any(b==i and pos>=arity for _,b,pos in self.edges):raise ValueError('Extra graph argument')
            if n.op in ('dt1','dt2','lap2','mul','residual_equal') and n.signature!='field':raise ValueError('Operator output type mismatch')
            if len(children)!=arity:raise ValueError('Missing ordered graph arguments')
            if n.op in ('dt1','dt2','lap2'):
                if children[0].signature!='field':raise ValueError('Derivatives require a dynamic field, not a scalar parameter')
                u=list(children[0].units);u[2 if n.op.startswith('dt') else 1]-=2 if n.op in ('dt2','lap2') else 1
                if tuple(u)!=n.units:raise ValueError('Derivative units/order mismatch')
            if n.op=='power2' and (children[0].signature!='scalar' or tuple(2*x for x in children[0].units)!=n.units):raise ValueError('Squared scalar units mismatch')
            if n.op=='mul' and tuple(a+b for a,b in zip(children[0].units,children[1].units))!=n.units:raise ValueError('Multiplication unit mismatch')
            if n.op=='residual_equal' and (children[0].units!=children[1].units or children[0].signature!='field' or children[1].signature!='field'):raise ValueError('Equality requires matching field units')
        pending=set(range(len(self.nodes)));resolved=set()
        while pending:
            ready={i for i in pending if all(a in resolved for a,b,_ in self.edges if b==i)}
            if not ready:raise ValueError('PDE operator graph must be acyclic')
            resolved|=ready;pending-=ready
        # Registry recognizes exact structure, not arbitrary graph embeddings.
        if self.family not in ('heat','wave'):raise ValueError('Static/weak/unregistered equations are not executable')
        return self
    def to_dict(self):return {'version':GRAPH_VERSION,'family':self.family,'nodes':[asdict(n) for n in self.nodes],'edges':self.edges}

def canonical(family):
    if family not in ('heat','wave'):raise ValueError('Only dynamic heat/wave graphs registered; static/weak execution unsupported')
    wave=family=='wave';param='scalar_c' if wave else 'scalar_D';pu=(0,1,-1) if wave else (0,2,-1);rhs=(1,0,-2) if wave else (1,0,-1)
    nodes=[Node('field_u','field',(1,0,0)),Node('field_v','field',(1,0,-1)),Node(param,'scalar',pu),Node('coordinate_x','coordinate',(0,1,0)),Node('coordinate_y','coordinate',(0,1,0)),Node('coordinate_t','coordinate',(0,0,1)),Node('dt2' if wave else 'dt1','field',rhs),Node('lap2','field',(1,-2,0))]
    edges=[(0,6,0),(0,7,0)]
    if wave:nodes.append(Node('power2','scalar',(0,2,-2)));edges.append((2,8,0));param_index=8
    else:param_index=2
    product=len(nodes);nodes.append(Node('mul','field',rhs));edges.extend([(param_index,product,0),(7,product,1)])
    residual=len(nodes);nodes.append(Node('residual_equal','field',rhs));edges.extend([(6,residual,0),(product,residual,1)])
    return PDEGraph(tuple(nodes),tuple(edges),family).validate()

def parse_equation(text):
    """Normalize a tiny polynomial residual grammar; no eval/symbolic execution."""
    if text.count('=')!=1:raise ValueError('Use one supported equality')
    left,right=text.replace('^','**').split('=')
    def polynomial(n):
        if isinstance(n,ast.Constant) and isinstance(n.value,(int,float)):return {():float(n.value)}
        if isinstance(n,ast.Name) and n.id in ('u_t','u_tt','D','c'):return {(n.id,):1.}
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='lap' and len(n.args)==1 and isinstance(n.args[0],ast.Name) and n.args[0].id=='u' and not n.keywords:return {('lap_u',):1.}
        if isinstance(n,ast.UnaryOp) and isinstance(n.op,(ast.USub,ast.UAdd)):return {k:v*(-1 if isinstance(n.op,ast.USub) else 1) for k,v in polynomial(n.operand).items()}
        if isinstance(n,ast.BinOp):
            if isinstance(n.op,ast.Pow) and isinstance(n.right,ast.Constant) and n.right.value==2 and isinstance(n.left,ast.Name) and n.left.id=='c':return {('c','c'):1.}
            a=polynomial(n.left);b=polynomial(n.right)
            if isinstance(n.op,(ast.Add,ast.Sub)):
                r=a.copy()
                for k,v in b.items():r[k]=r.get(k,0)+v*(-1 if isinstance(n.op,ast.Sub) else 1)
                return r
            if isinstance(n.op,ast.Mult):
                r={}
                for k,v in a.items():
                    for j,w in b.items():key=tuple(sorted(k+j));r[key]=r.get(key,0)+v*w
                return r
        raise ValueError('Unsupported operator/type: use u_t=D*lap(u) or u_tt=c**2*lap(u)')
    try:a=polynomial(ast.parse(left.strip(),mode='eval').body);b=polynomial(ast.parse(right.strip(),mode='eval').body)
    except SyntaxError as error:raise ValueError('Invalid equation syntax') from error
    for k,v in b.items():a[k]=a.get(k,0)-v
    a={k:v for k,v in a.items() if abs(v)>1e-12}
    for family,derivative,param in [('heat','u_t',('D','lap_u')),('wave','u_tt',('c','c','lap_u'))]:
        scale=a.get((derivative,),0)
        if scale and a=={(derivative,):scale,param:-scale}:return canonical(family)
    raise ValueError('Equation is outside the registered constant-coefficient heat/wave structures; c and c² are distinct')

class GraphEncoder(nn.Module):
    def __init__(self,width,depth=2):
        super().__init__();self.node=nn.Embedding(len(VOCAB),width);self.types=nn.Embedding(3,width);self.units=nn.Linear(3,width);self.edge=nn.Embedding(2,width)
        self.messages=nn.ModuleList([nn.Sequential(nn.Linear(3*width,width),nn.SiLU(),nn.Linear(width,width)) for _ in range(depth)])
        self.updates=nn.ModuleList([nn.Sequential(nn.LayerNorm(width),nn.Linear(width,width),nn.SiLU()) for _ in range(depth)])
    def one(self,g):
        g.validate();dev=self.node.weight.device
        h=self.node(torch.tensor([VOCAB.index(n.op) for n in g.nodes],device=dev))+self.types(torch.tensor([('scalar','field','coordinate').index(n.signature) for n in g.nodes],device=dev))+self.units(torch.tensor([n.units for n in g.nodes],device=dev,dtype=self.node.weight.dtype))
        a=torch.tensor([e[0] for e in g.edges],device=dev);b=torch.tensor([e[1] for e in g.edges],device=dev);positions=torch.tensor([e[2] for e in g.edges],device=dev)
        for message,update in zip(self.messages,self.updates):
            m=message(torch.cat([h[a],h[b],self.edge(positions)],-1));aggregated=torch.zeros_like(h).index_add(0,b,m.to(h.dtype))
            # Directed edges also send an explicitly reverse structural message.
            reverse=torch.zeros_like(h).index_add(0,a,message(torch.cat([h[b],h[a],-self.edge(positions)],-1)).to(h.dtype))
            h=h+update(aggregated+reverse)
        return h.mean(0)
    def forward(self,families,graphs=None):
        # Graph pooling is node-relabel invariant; independent graph vectors retained per problem.
        if graphs is not None:return torch.stack([self.one(g) for g in graphs])
        heat=self.one(canonical('heat'));wave=self.one(canonical('wave'))
        return torch.stack([heat,wave])[families.long()]
