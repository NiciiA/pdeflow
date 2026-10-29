"""One nested nodal heat grid level; fresh fine generation, no noisy-state bridge."""
from dataclasses import replace
import torch
from pdeflow.inference import validate_compatibility
from pdeflow.observations import PointOperator,ObservationSet


def refine_request(problem,observations,fine_bundle,inference,max_grid_points):
    coarse=problem.grid;fine=fine_bundle.config.grid
    if problem.family!='heat' or fine_bundle.config.specialist!='heat':raise ValueError('Global refinement currently supports heat only')
    if fine.dimension!=coarse.dimension or fine.domain!=coarse.domain or fine.times!=coarse.times or fine.convention!=coarse.convention or fine.temporal_kind!=coarse.temporal_kind or fine.shape!=tuple(2*n for n in coarse.shape):raise ValueError('Need a compatible 2x nested nodal heat grid, identical domain/boundary/physical times')
    if fine.convention!='periodic_nodes_endpoint_excluded':raise ValueError('Only periodic nodal heat refinement implemented')
    import math
    if math.prod(fine.shape)>max_grid_points:raise ValueError('Fine physical grid exceeds max_grid_points')
    refined=replace(problem,grid=fine)
    validate_compatibility(refined,fine_bundle.config,inference)
    ix=observations.operator.indices.clone();ix[:,2:]*=2
    remapped=ObservationSet(PointOperator(fine,ix,observations.operator.channels),observations.values.clone(),observations.mask.clone(),observations.noise_std,observations.noiseless_tolerance,observations.field_noise_std,observations.field_tolerance,observations.measurement_noise_std)
    if remapped.operator.coordinates()!=observations.operator.coordinates():raise ValueError('Nested observation remap altered physical coordinates')
    return refined,remapped


def restrict_nodes(fine):
    """Diagnostic injection only; not a conservative cell-average restriction."""
    return fine[(...,)+tuple(slice(None,None,2) for _ in range(fine.ndim-3))]


def prolong_nodes(coarse,shape):
    """Periodic multilinear diagnostic interpolation, never used as source/measurement."""
    from itertools import product
    dimension=len(shape);out=coarse
    for j,n in enumerate(shape):
        axis=3+j
        if n!=2*out.shape[axis]:raise ValueError('Only 2x periodic nodal interpolation')
        expanded=torch.stack((out,.5*(out+torch.roll(out,-1,axis))),dim=axis+1)
        out=expanded.flatten(axis,axis+1)
    return out
