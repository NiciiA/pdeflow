"""A1-only active quantity weights; global training normalization remains unchanged."""
import math
import torch

def settings(raw=None):
    value={'u':1.,'v':1.,'parameter':.1,**(raw or {})}
    if set(value)!={'u','v','parameter'} or any(not math.isfinite(x) or x<=0 for x in value.values()):raise ValueError('Positive finite A1 u/v/parameter weights required')
    return value

def combine(c,losses):
    weights=settings(c.representation_weights);w=losses[0].new_tensor([weights[name] for name in ('u','v')[:len(losses)]])
    return (torch.stack(losses)*w).sum()/w.sum()
