"""Finite real Fourier-basis ridge inverse reconstruction for periodic heat."""
from itertools import product
import time

import torch

from pdeflow.core import GridSpec, ReconstructionResult
from pdeflow.observations import ObservationSet


def fourier_heat_basis(grid: GridSpec, diffusivity: float, degree: int):
    if degree < 1 or degree >= min(grid.shape) / 2 or diffusivity <= 0:
        raise ValueError("Basis degree must be below Nyquist and diffusivity positive")
    coords = grid.coordinates()
    factors, eigenvalues = [], []
    for x, origin, length in zip(coords, grid.domain.origins, grid.domain.lengths):
        values, k2 = [torch.ones_like(x)], [0.]
        for k in range(1, degree + 1):
            angle = 2 * torch.pi * k * (x - origin) / length
            values.extend([2**.5 * angle.cos(), 2**.5 * angle.sin()])
            k2.extend([float((2 * torch.pi * k / length)**2)] * 2)
        factors.append(values)
        eigenvalues.append(k2)
    times = torch.tensor(grid.times, dtype=torch.float64)
    trajectories, frequencies = [], []
    for indices in product(range(2 * degree + 1), repeat=grid.dimension):
        initial = torch.ones(grid.shape, dtype=torch.float64)
        k2 = 0.
        for axis, index in enumerate(indices):
            shape = [1] * grid.dimension
            shape[axis] = grid.shape[axis]
            initial = initial * factors[axis][index].reshape(shape)
            k2 += eigenvalues[axis][index]
        trajectories.append(initial.unsqueeze(0) * torch.exp(-diffusivity * k2 * times).reshape(-1, *([1] * grid.dimension)))
        frequencies.append(k2)
    return torch.stack(trajectories).unsqueeze(1), torch.tensor(frequencies, dtype=torch.float64)


def inverse_baseline(grid: GridSpec, diffusivity: float, observations: ObservationSet, degree=1, ridge=.001):
    if ridge <= 0 or observations.values.shape[0] != 1:
        raise ValueError("Baseline requires positive ridge and one physical problem")
    start = time.perf_counter()
    basis, frequencies = fourier_heat_basis(grid, diffusivity, degree)
    mask = observations.mask[0].cpu()
    design = observations.operator.apply(basis).T[mask]
    target = observations.values[0].cpu().double()[mask]
    scales=observations.scales(target).cpu().double()[mask]
    weights=1/(len(target)**.5*scales)
    weighted_design=design*weights[:,None];weighted_target=target*weights
    scale=scales.tolist() if observations.measurement_noise_std is not None else max(observations.noise_std,observations.noiseless_tolerance)
    penalty = (1 + frequencies / (2 * torch.pi)**2).square()
    coefficients = torch.linalg.solve(weighted_design.T @ weighted_design + ridge * torch.diag(penalty), weighted_design.T @ weighted_target)
    reconstruction = torch.einsum("k,kct...->ct...", coefficients, basis).unsqueeze(0).unsqueeze(0)
    return ReconstructionResult(reconstruction, {"seconds": time.perf_counter() - start, "model_evaluations": 0,
                                "gradient_evaluations": 0, "basis_size": len(coefficients), "degree": degree, "ridge": ridge},
                                {"method": "real_fourier_heat_ridge", "initial_condition_unknown": True,
                                 "noise_scale_used": scale, "basis_mismatch": "truncated tensor-product Fourier basis versus smooth full-spectrum generator"})
