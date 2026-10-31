"""Operator-v2 regressions; superseded trajectory tests preserved in recovery evidence."""
from dataclasses import replace


import numpy as np


import pytest


import torch


from pdeflow.coordinate.contracts import CoordinateConfig,grid_queries,validate_queries,validate_rows,padded_observations


from pdeflow.coordinate.model import Autoencoder,Velocity,decode_chunks


from pdeflow.core import GridSpec,DomainSpec


from pdeflow.equations import heat_problem


from pdeflow.coordinate.evaluation import error_summary,interpolate_grid


def test_interpolation_order_no_extrapolation_and_near_zero_errors():
    g=GridSpec((4,),DomainSpec((1.,)),(0.,.05,.1));values=np.array([np.arange(4)+t for t in g.times]);q=grid_queries(g).numpy();result=interpolate_grid(values,g,q)
    np.testing.assert_allclose(result,values.reshape(-1),atol=1e-7)
    q=q[::-1].copy();np.testing.assert_allclose(interpolate_grid(values,g,q),result[::-1],atol=1e-7)
    with pytest.raises(ValueError):interpolate_grid(values,g,np.array([[.1,.2]]))
    assert error_summary(np.ones((2,5)),np.zeros(5))['relative_l2_per_sample'] is None


def test_heteroscedastic_noise_weights_baseline():
    from pdeflow.observations import ObservationSet,PointOperator
    from pdeflow.evaluation.baseline import inverse_baseline
    g=GridSpec((8,),DomainSpec((1.,)),(0.,.05,.1));indices=torch.tensor([[0,0,0],[0,0,4]],dtype=torch.long)
    op=PointOperator(g,indices);values=torch.tensor([[1.,-10.]],dtype=torch.float64);mask=torch.ones_like(values,dtype=torch.bool)
    hetero=ObservationSet(op,values,mask,10.,.01,measurement_noise_std=(.01,10.))
    uniform=ObservationSet(op,values,mask,10.,.01)
    weighted=inverse_baseline(g,.07,hetero,ridge=.1).samples[0]
    unweighted=inverse_baseline(g,.07,uniform,ridge=.1).samples[0]
    assert abs(float(op.apply(weighted)[0,0])-1)<abs(float(op.apply(unweighted)[0,0])-1)
