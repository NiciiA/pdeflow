"""Weights-only-safe ambient RNG state, additional to the trainer's local cursor RNG."""
import random
import numpy as np
import torch


def capture():
    n=np.random.get_state()
    return {'python':random.getstate(),'numpy':{'algorithm':n[0],'keys':n[1].tolist(),'position':n[2],'has_gauss':n[3],'cached_gaussian':n[4]},
            'mps':torch.mps.get_rng_state() if torch.backends.mps.is_available() else None}


def restore(state):
    if not state:return  # old checkpoints remain functional, not complete replay
    random.setstate(state['python']);n=state['numpy']
    np.random.set_state((n['algorithm'],np.array(n['keys'],dtype=np.uint32),n['position'],n['has_gauss'],n['cached_gaussian']))
    if state.get('mps') is not None and torch.backends.mps.is_available():torch.mps.set_rng_state(state['mps'])
