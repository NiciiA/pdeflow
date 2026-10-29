from dataclasses import replace
import numpy as np
import pytest
from pdeflow.reference.examples import example
from pdeflow.reference.solvers import solve
from pdeflow.workbench.inspection import local_values,plane
from pdeflow.numerics import fd_laplacian,time_derivative
from pdeflow.darcy import tensor_fluxes
import torch

@pytest.mark.parametrize('family',['heat','darcy','chemistry'])
def test_native_local_terms_match_independent_tensor_operator(family):
 p,v,_=solve(example(family+'-3d'));pt=[2,3,4];t=0
 report=local_values(p,v,t,pt)
 assert np.isfinite(list(report['terms'].values())).all()
 if family=='heat':
  x=torch.tensor(v[None]);expected=time_derivative(x,p.grid)-p.known_parameters['diffusivity']*fd_laplacian(x,p.grid)
  assert report['terms']['residual']==pytest.approx(float(expected[(0,0,t,*pt)]))
 if family=='darcy':
  from pdeflow.darcy import tensor_fluxes
  flux=tensor_fluxes(torch.tensor(v[None]),p.grid)
  expected=sum((q-q.roll(1,axis+1))/h for axis,(q,h) in enumerate(zip(flux,p.grid.spacing)))-torch.tensor(p.known_fields['f'],dtype=torch.float64)[None]
  assert report['terms']['residual']==pytest.approx(float(expected[(0,*pt)]),abs=1e-12)
 for normal in range(3):
  view=plane(p,v,0,t,normal,pt);np.testing.assert_array_equal(view['values'],np.take(v[0,t],pt[normal],axis=normal))


def test_reference_1024_native_and_bounded_display(tmp_path):
 from pdeflow.reference import create,preview,load
 from pdeflow.reference.observations import sample,ObservationSamplingSpec
 s=example('heat-2d',shape=[1024,1024]);s=replace(s,request={**s.request,'domain':{**s.request['domain'],'times':[0.,.05,.1]}})
 create(s,tmp_path/'ref');p,v,_=load(tmp_path/'ref');assert v.shape==(1,1,3,1024,1024)
 view=preview(tmp_path/'ref');assert view['shape']==[128,128] and view['native_shape']==[1024,1024] and view['display_downsampled']
 sampled=sample(tmp_path/'ref',ObservationSamplingSpec(count=4),tmp_path/'obs');assert sampled['count']==12
