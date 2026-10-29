"""Normalized empirical population summaries; no posterior-calibration claim."""
import torch


def weights(result):
    p=result.samples.shape[1]
    if result.weights is None:return torch.full((p,),1/p,dtype=torch.float64)
    w=result.weights.detach().cpu().double()
    if tuple(w.shape)!=(1,p) or not torch.isfinite(w).all() or (w<0).any() or not torch.allclose(w.sum(1),torch.ones(1,dtype=torch.float64),atol=1e-10):raise ValueError('Invalid normalized reconstruction weights')
    return w[0]


def mean(values,w):return (values*w.to(values).reshape(-1,*([1]*(values.ndim-1)))).sum(0)


def quantile(values,w,q):
    order=values.argsort(0);sorted_values=torch.gather(values,0,order)
    expanded=w.to(values).reshape(-1,*([1]*(values.ndim-1))).expand_as(values)
    sorted_weights=torch.gather(expanded,0,order);cdf=sorted_weights.cumsum(0)
    ix=(cdf<q).sum(0).clamp_max(len(w)-1)
    return torch.gather(sorted_values,0,ix[None])[0]
