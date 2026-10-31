"""Operator-v2 regressions; superseded trajectory tests preserved in recovery evidence."""
from pathlib import Path


from dataclasses import asdict,replace


import numpy as np


import pytest


import torch


from fastapi.testclient import TestClient


from pdeflow.coordinate.training import train,read_checkpoint


from pdeflow.coordinate.inference import CoordinateBundle,reconstruct


from pdeflow.coordinate.contracts import grid_queries


from pdeflow.coordinate.registry import runs


from pdeflow.core import GridSpec,DomainSpec


from pdeflow.equations import heat_problem


from pdeflow.reference import create,load


from pdeflow.reference.examples import example


from pdeflow.reference.imports import import_reference


from pdeflow.reference.observations import sample,ObservationSamplingSpec


from pdeflow.runtime import read_json,file_digest


from pdeflow.workbench.api import create_app


from pdeflow.local import run_lock


def test_reference_import_manufactured_sampling_reproducibility_noise_and_no_leak(tmp_path):
    spec=example('heat-2d',(4,4));created=create(spec,tmp_path/'reference');p,y,m=load(tmp_path/'reference')
    payload={'problem':p.to_dict(),'coordinates':{a:c.tolist() for a,c in zip(p.grid.axis_names,p.grid.coordinates())},'physical_times':list(p.grid.times),'fields':['u'],'values':y[0].tolist(),'units':'nondimensional','provenance':{'solver':'analytic finite mode'}}
    import_reference(payload,tmp_path/'import','a'*32);np.testing.assert_array_equal(load(tmp_path/'import')[1],y)
    bad={**payload,'coordinates':{**payload['coordinates'],'x':[0,0,.5,.75]}}
    with pytest.raises(ValueError):import_reference(bad,tmp_path/'bad','b'*32)
    sampled=[]
    for name in ('one','two'):
        sampled.append(sample(tmp_path/'import',ObservationSamplingSpec(count=3,fields=('u',),time_indices=(0,2),noise_std=.1,noise_mode='relative_rms'),tmp_path/name))
    assert sampled[0]['rows']==sampled[1]['rows'] and sampled[0]['count']==6
    assert all('sigma' in r for r in sampled[0]['rows'])
    obs=read_json(tmp_path/'one/observations.json');assert len(obs['measurement_noise_std'])==6;assert 'clean_values' not in obs
    from pdeflow.observations import ObservationSet
    restored=ObservationSet.from_dict(obs,p.grid);assert restored.to_dict()==obs
    from pdeflow.reference.solvers import ReferenceProblem,solve
    candidate=ReferenceProblem(spec.request,{'initial_u':{'expression':'cos(2*pi*x)'},'analytical_reference':{'expression':'cos(2*pi*x)'}},'manufactured')
    _,field,diagnostic=solve(candidate);assert not diagnostic['verification']['passed'];assert 'unverified' in diagnostic['classification'];assert np.isfinite(field).all()
    with pytest.raises(ValueError):solve(replace(candidate,solver='analytical'))
    singular=replace(candidate,inputs={'initial_u':0,'analytical_reference':{'expression':'1/(x-x)'}})
    with pytest.raises(ValueError):solve(singular)


def test_full_field_api_import_export_and_noise_row_roundtrip(tmp_path):
    create(example('heat-2d',(4,4)),tmp_path/'reference')
    from pdeflow.reference.imports import export_reference
    artifact=export_reference(tmp_path/'reference');app=create_app(workspace=tmp_path/'workspace')
    with TestClient(app) as client:
        imported=client.post('/api/references/import',json={'artifact':artifact});assert imported.status_code==201
        identifier=imported.json()['id'];again=client.get(f'/api/references/{identifier}/artifacts/reference-field.json');assert again.status_code==200
        assert again.json()['values']==artifact['values']
        observed=client.post(f'/api/references/{identifier}/sample',json={'sampling':{'pattern':'space_time','count':3,'noise_std':0,'noise_mode':'absolute'}});assert observed.status_code==201
        rows=observed.json()['rows'];validated=client.post('/api/observations/validate',json={'problem':artifact['problem'],'rows':rows});assert validated.status_code==200
        assert validated.json()['rows']==rows
        oversized=client.post(f'/api/references/{identifier}/sample',json={'sampling':{'pattern':'fixed_sensors','count':17}});assert oversized.status_code==422
