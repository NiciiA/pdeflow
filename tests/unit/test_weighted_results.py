import pytest
import torch
from pdeflow.core import DomainSpec,GridSpec,ReconstructionResult
from pdeflow.equations import heat_problem
from pdeflow.observations import PointOperator,ObservationSet
from pdeflow.evaluation import metrics
from pdeflow.evaluation.weighted import quantile


def test_weighted_physical_metrics_and_quantiles():
 g=GridSpec((4,),DomainSpec((1.,)),(0.,.1,.2))
 p=heat_problem(g,.1);samples=torch.stack([torch.zeros(1,3,4),torch.full((1,3,4),10.)])[None]
 o=ObservationSet(PointOperator(g,torch.tensor([[0,0,0]])),torch.tensor([[0.]]),torch.ones(1,1,dtype=torch.bool),1.)
 r=ReconstructionResult(samples,{}, {},weights=torch.tensor([[.9,.1]],dtype=torch.float64))
 m=metrics(r,p,o,torch.ones(1,1,3,4))
 assert m['field_u_mae']==pytest.approx(0.,abs=1e-12)
 assert m['observation_rmse']==pytest.approx(10*.1**.5)
 assert m['ensemble_rms_spread']==pytest.approx(3.)
 assert quantile(samples[0],r.weights[0],.5).max()==0
 assert quantile(samples[0],r.weights[0],.95).min()==10
