from copy import deepcopy
import pytest
from pdeflow.runtime import Config
from pdeflow.symbolic import analyze, presets


def request(preset='heat',dimension=2):
    p=next(p for p in presets() if p['id']==preset)
    grid=Config.load(f'configs/experiments/cpu_{dimension}d.json').to_dict()['grid']
    return {'schema_version':1,'equations':deepcopy(p['equations']),'declarations':deepcopy(p['declarations']),
            'domain':{**grid,'times':[] if p['static'] else grid['times']},'boundary':{'type':'periodic','status':'known','details':''},
            'initial':{'status':'not_applicable' if p['static'] else 'unknown','details':''},
            'measurements':{'rows':[],'status':'unknown','noise_model':'independent_gaussian','noise_std':.03},
            'observation_operator':{'kind':'point_on_grid','status':'known','details':''},'units':'nondimensional','preset':preset}


@pytest.mark.parametrize('source',[r'u_\tau=D\Delta u',r'u_{\tau}=D\Delta u',r'\partial_\tau u=D\Delta u',r'\frac{\partial u}{\partial \tau}=D\Delta u','∂u/∂τ=D Δu',r'u_\tau-D\Delta u=0'])
def test_equivalent_heat_notations(source):
    r=request();r['equations']=[source];result=analyze(r)
    assert result['recognized_family']=='heat';assert result['problem']['known_parameters']=={'diffusivity':.07}


@pytest.mark.parametrize('preset',[p['id'] for p in presets()])
def test_all_presets_canonical_roundtrip_and_execution_boundary(preset):
    r=request(preset);a=analyze(r);r['equations']=a['canonical_source'];b=analyze(r)
    assert a['parsed']==b['parsed']
    assert (a['problem'] is not None)==(preset in ('heat','chemistry'))
    if preset in ('poisson','darcy'):assert not r['domain']['times'] and r['initial']['status']=='not_applicable'


@pytest.mark.parametrize('source,reason',[
 ('u_τ=', 'Incomplete'),('u_τ=q lap(u)','Undefined'),('u_τ=sin(u)','Undefined'),('u_z=D lap(u)','absent'),
 ('u_τ=D lap(u)+v','Undefined'),('u_τ=D lap(u)^4','Powers'),('u_τ=div(u)','vector'),('u_τ=∂_s u','Unsupported'),
 ('u_τ=__import__(u)','Unsupported'),('u_τ='+('('*40)+'u'+(')'*40),'depth'),
])
def test_failures_preserve_source_and_never_return_old_problem(source,reason):
    r=request();analyze(r);r['equations']=[source];saved=deepcopy(r)
    with pytest.raises(ValueError,match=reason):analyze(r)
    assert r==saved


def test_greek_nu_is_not_velocity_and_ranks_are_explicit():
    r=request('navier_stokes');assert analyze(r)['parsed'][0]['lhs']['rank']==1
    r['declarations']=[d for d in r['declarations'] if d['name']!='ν']
    with pytest.raises(ValueError,match='Undefined'):analyze(r)
    r=request('advection');r['declarations'][1]['rank']=0
    with pytest.raises(ValueError,match='Dot'):analyze(r)
    r=request('navier_stokes');r['equations']=['v=v^2']
    with pytest.raises(ValueError,match='scalar base'):analyze(r)


def test_coefficient_edit_unsupported_term_and_information_constraints():
    r=request();r['declarations'][1]['value']=.09
    assert analyze(r)['problem']['known_parameters']['diffusivity']==.09
    r['equations']=[r'u_\tau=D\Delta u+u'];assert analyze(r)['problem'] is None
    r=request();r['initial']['status']='known';assert analyze(r)['problem'] is None
    r=request();r['boundary']['type']='neumann';assert analyze(r)['problem'] is None
    r=request('darcy');r['initial']['status']='unknown'
    with pytest.raises(ValueError,match='Static'):analyze(r)


def test_rate_definition_cycles_and_undefined_forcing():
    r=request('chemistry');r['equations'][-1]='r=r'
    with pytest.raises(ValueError,match='Cyclic'):analyze(r)
    r=request('darcy');r['declarations']=[d for d in r['declarations'] if d['name']!='f']
    with pytest.raises(ValueError,match='Undefined'):analyze(r)


def test_declared_subscripted_coefficient_is_not_a_derivative():
    from pdeflow.symbolic import declaration
    r=request();r['declarations'].append(declaration('D_x','parameter',value=.03));r['equations']=['u_τ=D_x lap(u)']
    result=analyze(r)
    assert result['parsed'][0]['rhs']['args'][0]['name']=='D_x'
    assert result['problem'] is None
