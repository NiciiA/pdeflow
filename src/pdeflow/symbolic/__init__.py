"""Bounded PDE notation parser. Editing support is independent of execution."""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any

from pdeflow.core import GridSpec
from pdeflow.equations import heat_problem

GREEK = {'tau': 'τ', 'nu': 'ν', 'lambda': 'λ', 'rho': 'ρ'}
STATUS = {'known', 'partially_known', 'unknown', 'not_applicable'}


@dataclass(frozen=True)
class Node:
    op: str
    args: tuple[Node, ...] = ()
    name: str = ''
    value: float = 0.
    rank: int = 0

    def json(self):
        return {'op': self.op, 'args': [a.json() for a in self.args], 'name': self.name, 'value': self.value, 'rank': self.rank}

    def source(self):
        if self.op == 'symbol': return self.name
        if self.op == 'number': return str(self.value)
        if self.op in {'add', 'sub', 'mul', 'quotient', 'power', 'dot'}:
            operator = {'add': '+', 'sub': '-', 'mul': '*', 'quotient': '/', 'power': '^', 'dot': '·'}[self.op]
            return f'({self.args[0].source()}{operator}{self.args[1].source()})'
        return f'{self.op}(' + ','.join(a.source() for a in self.args) + ')'


def normalize(text: str, names: list[str]) -> str:
    if len(text) > 4096: raise ValueError('Equation exceeds 4096 characters')
    text = text.replace('\\left', '').replace('\\right', '').replace('\\,', ' ').replace('\\!', '').replace('\\ ', ' ')
    text = re.sub(r'\\(?:mathbf|mathrm|operatorname)\{([A-Za-zνλρ][A-Za-zνλρ0-9_]*)\}', r' \1 ', text)
    for name, char in GREEK.items(): text = re.sub(r'\\' + name + r'\b', char, text)
    text = text.replace('\\Delta', 'Δ').replace('\\nabla', '∇').replace('\\partial', '∂').replace('\\cdot', '·').replace('\\times', '·')
    text = re.sub(r'∇\s*·', ' div ', text).replace('∇', ' grad ').replace('Δ', ' lap ')
    # Common directional convection, not a generic identity/simplification.
    text = re.sub(r'\(\s*([A-Za-z]+)\s*·\s*grad\s*\)\s*([A-Za-z]+)', r'adv(\1,\2)', text)
    # Explicit fraction derivative rules precede arithmetic fraction translation.
    for variable in ['τ', 't', 'x', 'y', 'z']:
        v = re.escape(variable)
        text = re.sub(r'\\frac\s*\{\s*∂\s*\^\s*\{?2\}?\s*([A-Za-zνλρ][A-Za-zνλρ0-9_]*)\s*\}\s*\{\s*∂\s*'+v+r'\s*\^\s*\{?2\}?\s*\}', lambda m: f'd{variable}d{variable}({m[1]})', text)
        text = re.sub(r'\\frac\s*\{\s*∂\s*([A-Za-zνλρ][A-Za-zνλρ0-9_]*)\s*\}\s*\{\s*∂\s*'+v+r'\s*\}', lambda m: f'd{variable}({m[1]})', text)
    text = re.sub(r'∂\s*_\s*\{?(ττ|tt|xx|yy|zz|τ|t|x|y|z)\}?\s*([A-Za-zνλρ][A-Za-zνλρ0-9_]*)', lambda m: 'd'+'d'.join(m[1])+f'({m[2]})', text)
    text = re.sub(r'∂\s*([A-Za-zνλρ][A-Za-zνλρ0-9_]*)\s*/\s*∂\s*(τ|t|x|y|z)', lambda m: f'd{m[2]}({m[1]})', text)
    for name in sorted(names, key=len, reverse=True):
        # Declared subscripted symbols win over derivative shorthand.
        text = re.sub(r'(?<![A-Za-z0-9_])'+re.escape(name)+r'_\s*\{?(ττ|tt|xx|yy|zz|τ|t|x|y|z)\}?', lambda m: name+'_'+m[1] if name+'_'+m[1] in names else 'd'+'d'.join(m[1])+f'({name})', text)
    # Small balanced-brace translator; never invokes a symbolic evaluator.
    def group(s, pos):
        while pos < len(s) and s[pos].isspace(): pos += 1
        if pos == len(s) or s[pos] != '{': raise ValueError('Expected a braced fraction argument')
        start, depth = pos + 1, 1; pos += 1
        while pos < len(s) and depth:
            depth += (s[pos] == '{') - (s[pos] == '}'); pos += 1
        if depth: raise ValueError('Incomplete braces')
        return s[start:pos-1], pos
    for _ in range(64):
        j = text.find('\\frac')
        if j < 0: break
        a, k = group(text, j+5); b, end = group(text, k)
        text = text[:j] + '(('+a+')/('+b+'))' + text[end:]
    text = re.sub(r'_\{([A-Za-z0-9]+)\}', r'_\1', text)
    text = text.replace('{', '(').replace('}', ')').replace('−', '-').replace('²', '^2')
    if '\\' in text or '∂' in text: raise ValueError('Unsupported LaTeX command or derivative; see the PDE grammar')
    return text


class Parser:
    def __init__(self, text: str, declarations: dict, coordinates: list[str], transient: bool):
        self.text = normalize(text, list(declarations)); self.declarations = declarations
        self.coordinates, self.transient = coordinates, transient
        self.tokens = re.findall(r'(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|[A-Za-zνλρτ][A-Za-zνλρτ0-9_]*|[()+*/^,=·-]', self.text)
        if ''.join(self.tokens) != re.sub(r'\s+', '', self.text): raise ValueError('Unsupported symbol in equation')
        if len(self.tokens) > 512: raise ValueError('Equation exceeds 512 tokens')
        self.pos = 0; self.depth = 0

    def peek(self): return self.tokens[self.pos] if self.pos < len(self.tokens) else ''
    def take(self):
        token = self.peek()
        if not token: raise ValueError('Incomplete equation')
        self.pos += 1; return token
    def expect(self, token):
        if self.take() != token: raise ValueError(f'Expected {token}')
    def expression(self, minimum=0):
        self.depth += 1
        if self.depth > 32: raise ValueError('Expression exceeds depth 32')
        token = self.take()
        if token == '-':
            operand = self.expression(25)
            left = Node('number', value=-operand.value) if operand.op == 'number' else Node('mul', (Node('number', value=-1), operand), rank=operand.rank)
        elif token == '+': left = self.expression(25)
        elif token == '(':
            left = self.expression(); self.expect(')')
        elif re.fullmatch(r'(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?', token):
            value = float(token)
            if not math.isfinite(value): raise ValueError('Nonfinite constant')
            left = Node('number', value=value)
        elif token in {'lap', 'grad', 'div', 'adv', 'dt', 'dtdt', 'dτ', 'dτdτ', 'dx', 'dy', 'dz', 'dxdx', 'dydy', 'dzdz'}:
            if token == 'adv':
                self.expect('('); a = self.expression(); self.expect(','); b = self.expression(); self.expect(')')
                if a.rank != 1 or b.rank != 1: raise ValueError('adv requires two declared vectors')
                left = Node('adv', (a, b), rank=1)
            else:
                a = self.expression(30); op = {'dt':'dτ','dtdt':'dτdτ'}.get(token,token)
                if op.startswith('d') and op != 'div':
                    var = op[1:2]
                    if var == 'τ' and not self.transient: raise ValueError('Physical time is not declared for this static problem')
                    if var != 'τ' and var not in self.coordinates: raise ValueError(f'Derivative coordinate {var} is absent from Ω')
                if op == 'grad' and a.rank != 0: raise ValueError('grad currently accepts scalar fields only')
                if op == 'div' and a.rank != 1: raise ValueError('div requires a declared vector')
                left = Node(op, (a,), rank=1 if op == 'grad' else 0 if op == 'div' else a.rank)
        elif token in self.declarations:
            d = self.declarations[token]; left = Node('symbol', name=token, rank=d['rank'])
        else: raise ValueError(f'Undefined symbol or unsupported operator: {token}')
        while self.peek():
            token = self.peek(); implicit = token not in {'+', '-', '*', '/', '^', '·', ')', '=', ','}
            precedence = {'+':10, '-':10, '*':20, '/':20, '·':20, '^':40}.get(token, 20 if implicit else -1)
            if precedence < minimum: break
            if not implicit: self.take()
            right = self.expression(precedence if token == '^' else precedence + 1)
            op = {'+':'add','-':'sub','*':'mul','/':'quotient','^':'power','·':'dot'}.get(token,'mul') if not implicit else 'mul'
            if op in {'add','sub'} and left.rank != right.rank: raise ValueError('Cannot add scalar and vector expressions')
            if op == 'mul' and left.rank and right.rank: raise ValueError('Vector multiplication requires an explicit dot product')
            if op == 'dot' and (left.rank != 1 or right.rank != 1):
                # LaTeX cdot is also ordinary scalar multiplication.
                if left.rank == right.rank == 0: op = 'mul'
                else: raise ValueError('Dot product requires two vectors or two scalars')
            if op in {'quotient','power'} and right.rank: raise ValueError('Scalar denominator/exponent required')
            if op == 'power' and left.rank: raise ValueError('Powers currently require a scalar base')
            if op == 'power' and (right.op != 'number' or right.value not in (0,1,2,3)): raise ValueError('Powers must be integers 0–3')
            left = Node(op, (left,right), rank=0 if op=='dot' else max(left.rank,right.rank))
        self.depth -= 1
        return left

    def equation(self):
        a=self.expression(); self.expect('='); b=self.expression()
        if self.peek(): raise ValueError('Unexpected trailing expression')
        if a.rank != b.rank: raise ValueError('Equation scalar/vector rank mismatch')
        return a,b


def declarations(request):
    result={}
    for d in request.get('declarations', []):
        name=d.get('name','')
        if not re.fullmatch(r'[A-Za-zνλρ][A-Za-z0-9_]*',name) or name in result or name in {'x','y','z','τ','t','lap','grad','div','adv','dt','dtdt','dx','dy','dz','dxdx','dydy','dzdz'}:
            raise ValueError(f'Invalid, reserved or duplicate declaration: {name}')
        if d.get('kind') not in {'field','parameter','coefficient','definition'} or d.get('rank') not in (0,1) or d.get('status') not in STATUS:
            raise ValueError(f'Invalid declaration for {name}')
        if not isinstance(d.get('units'),str) or not d['units']: raise ValueError(f'Units/scale must be declared for {name}')
        if d.get('value') is not None and (not isinstance(d['value'],(float,int)) or not math.isfinite(d['value'])): raise ValueError('Parameter value must be finite')
        result[name]=d
    if len(result)>32: raise ValueError('At most 32 declarations')
    return result


def analyze(request: dict[str, Any], *, observation_kinds=('point_on_grid',)) -> dict:
    if request.get('schema_version') != 1: raise ValueError('Unsupported mathematical request schema_version')
    allowed={'schema_version','equations','declarations','domain','boundary','initial','measurements','observation_operator','units','preset','known_fields'}
    if set(request)-allowed: raise ValueError('Unknown mathematical request keys')
    if not request.get('units'): raise ValueError('Declare units or nondimensionalization')
    decl=declarations(request); domain=request['domain']; dimension=len(domain['shape'])
    if dimension not in (1,2,3): raise ValueError('Spatial dimension must be 1–3')
    if any(type(n) is not int or n<4 for n in domain['shape']): raise ValueError('Ω requires integer grid sizes >=4')
    for key in ['origins','lengths']:
        values=domain['domain'][key]
        if len(values)!=dimension or any(not isinstance(n,(int,float)) or not math.isfinite(n) for n in values): raise ValueError('Invalid Ω coordinate extents')
    if any(n<=0 for n in domain['domain']['lengths']): raise ValueError('Domain lengths must be positive')
    times=domain.get('times',[])
    if any(not isinstance(n,(int,float)) or not math.isfinite(n) for n in times) or any(a>=b for a,b in zip(times,times[1:])): raise ValueError('Physical times must be finite and increasing')
    for section,keys in {'boundary':{'type','status','details'},'initial':{'status','details'},'measurements':{'rows','status','noise_model','noise_std'},'observation_operator':{'kind','status','details'}}.items():
        if set(request[section])-keys: raise ValueError(f'Unknown keys in {section}')
    equations=request.get('equations',[])
    if not 1<=len(equations)<=8 or any(not isinstance(e,str) or not e.strip() for e in equations): raise ValueError('Enter 1–8 nonempty equation rows')
    transient=bool(domain.get('times'))
    pairs=[Parser(e,decl,['x','y','z'][:dimension],transient).equation() for e in equations]
    definitions={}
    for left,right in pairs:
        if left.op=='symbol' and decl[left.name]['kind']=='definition':
            if left.name in definitions: raise ValueError('Duplicate rate definition')
            definitions[left.name]=right
    if set(definitions)!={n for n,d in decl.items() if d['kind']=='definition'}: raise ValueError('Every declared reaction rate needs one definition equation')
    def dependencies(node): return ({node.name} if node.op=='symbol' and node.name in definitions else set()).union(*(dependencies(a) for a in node.args))
    def visit(name,stack):
        if name in stack: raise ValueError('Cyclic rate definitions')
        for other in dependencies(definitions[name]): visit(other,stack|{name})
    for name in definitions: visit(name,set())
    for section in ['boundary','initial','observation_operator']:
        if request[section].get('status') not in STATUS: raise ValueError(f'Invalid information status for {section}')
    if not transient and request['initial']['status']!='not_applicable': raise ValueError('Static equations require initial information: not applicable')
    parsed=[{'lhs':a.json(),'rhs':b.json()} for a,b in pairs]
    canonical_source=[a.source()+'='+b.source() for a,b in pairs]
    from pdeflow.specialists import specialists
    recognized = next((sp for sp in specialists() if sp.recognize(pairs)), None)
    problem=None; reasons=[]
    if recognized is None:
        reasons.append('No implemented specialist recognizes this canonical equation; draft remains editable/exportable')
    else:
        if domain['domain']['boundary']!=request['boundary'].get('type'): reasons.append('Boundary declaration and Ω convention disagree')
        if request['boundary'].get('type')!='periodic' or request['boundary']['status']!='known': reasons.append('Implemented specialists require known periodic boundaries')
        if request['observation_operator'].get('kind') not in observation_kinds or request['observation_operator']['status']!='known': reasons.append('Known supported observation operator required: '+', '.join(observation_kinds))
        if request['measurements'].get('status') not in STATUS or request['measurements'].get('noise_model')!='independent_gaussian': reasons.append('Declared independent Gaussian measurement noise required')
        if request['units']!='nondimensional' or any(d['units']!='nondimensional' or d['rank']!=0 for d in decl.values()): reasons.append('Current specialists require nondimensional scalar declarations')
        if not reasons:
            try: problem=recognized.request_problem(request,decl).to_dict()
            except (ValueError,TypeError,KeyError) as error: reasons.append(str(error))
    return {'parsed':parsed,'canonical_source':canonical_source,'recognized_family':recognized.id if recognized else 'custom',
            'problem':problem,'status':{'editable':True,'parsed':True,'residual':bool(recognized),'data_generator':bool(recognized),'checkpoint':False,'reconstruction':False},'reasons':reasons}


def declaration(name,kind='field',rank=0,value=None,status=None,meaning=None):
    return {'name':name,'kind':kind,'rank':rank,'status':status or ('unknown' if kind=='field' else 'known'),'value':value,'units':'nondimensional','meaning':meaning or name}


def presets():
    catalog = [
      {'id':'heat','label':'Heat / diffusion','equations':[r'\frac{\partial u}{\partial \tau}=D\Delta u'],'declarations':[declaration('u'),declaration('D','parameter',value=.07)],'static':False},
      {'id':'advection','label':'Advection–diffusion','equations':[r'\partial_\tau u+\mathbf{v}\cdot\nabla u=D\Delta u'],'declarations':[declaration('u'),declaration('v','coefficient',rank=1),declaration('D','parameter',value=.07)],'static':False},
      {'id':'poisson','label':'Poisson','equations':[r'-\Delta u=f'],'declarations':[declaration('u'),declaration('f','coefficient')],'static':True},
      {'id':'darcy','label':'Darcy','equations':[r'-\nabla\cdot(a\nabla u)=f'],'declarations':[declaration('u'),declaration('a','coefficient',status='unknown'),declaration('f','coefficient')],'static':True},
      {'id':'burgers','label':'Viscous Burgers (1D)','equations':[r'\partial_\tau u+u u_x=\nu\Delta u'],'declarations':[declaration('u'),declaration('ν','parameter',value=.01)],'static':False},
      {'id':'wave','label':'Wave','equations':[r'\frac{\partial^2 u}{\partial \tau^2}=c_w^2\Delta u'],'declarations':[declaration('u'),declaration('c_w','parameter',value=1)],'static':False},
      {'id':'chemistry','label':'A+B ↔ C reaction–diffusion','equations':[r'\partial_\tau a=D_A\Delta a-r',r'\partial_\tau b=D_B\Delta b-r',r'\partial_\tau c=D_C\Delta c+r',r'r=k_f a b-k_r c'],'declarations':[declaration(n) for n in ['a','b','c']]+[declaration(n,'parameter',value=.01 if n.startswith('D') else 1) for n in ['D_A','D_B','D_C','k_f','k_r']]+[declaration('r','definition')],'static':False},
      {'id':'fokker_planck','label':'Fokker–Planck','equations':[r'\partial_\tau \rho=-\nabla\cdot(\mathbf{b}\rho)+D\Delta\rho'],'declarations':[declaration('ρ',meaning='probability density'),declaration('b','coefficient',rank=1),declaration('D','parameter',value=.1)],'static':False},
      {'id':'navier_stokes','label':'Incompressible Navier–Stokes','equations':[r'\partial_\tau\mathbf{v}+(\mathbf{v}\cdot\nabla)\mathbf{v}=-\nabla p+\nu\Delta\mathbf{v}+\mathbf{f}',r'\nabla\cdot\mathbf{v}=0'],'declarations':[declaration('v',rank=1,meaning='velocity'),declaration('p',meaning='pressure'),declaration('ν','parameter',value=.01),declaration('f','coefficient',rank=1)],'static':False},
    ]

    meanings = {
      'heat': {'u':'signed scalar diffusion field','D':'positive diffusivity'},
      'advection': {'u':'transported scalar','v':'known advection velocity','D':'positive diffusivity'},
      'poisson': {'u':'potential with a specified gauge','f':'compatible source field'},
      'darcy': {'u':'pressure with a specified gauge','a':'positive permeability field','f':'source / sink field'},
      'burgers': {'u':'scalar velocity','ν':'positive kinematic viscosity'},
      'wave': {'u':'displacement; initial velocity is also required','c_w':'positive wave speed'},
      'chemistry': {'a':'species A concentration (nonnegative)','b':'species B concentration (nonnegative)','c':'species C concentration (nonnegative)',
                    'D_A':'positive diffusivity of A','D_B':'positive diffusivity of B','D_C':'positive diffusivity of C','k_f':'positive forward mass-action rate','k_r':'positive reverse mass-action rate','r':'derived net reaction rate k_f a b - k_r c'},
      'fokker_planck': {'ρ':'nonnegative normalized probability density','b':'physical drift field','D':'positive physical diffusion'},
      'navier_stokes': {'v':'incompressible velocity vector','p':'pressure with a specified gauge','ν':'positive kinematic viscosity','f':'known body force vector'},
    }
    for preset in catalog:
        for d in preset['declarations']: d['meaning']=meanings[preset['id']][d['name']]
    return catalog
