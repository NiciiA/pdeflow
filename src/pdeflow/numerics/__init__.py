"""Periodic tensor operators and independent CPU reference evolution."""
from __future__ import annotations

import numpy as np
import torch

from pdeflow.core import GridSpec


def fd_gradient(u: torch.Tensor, grid: GridSpec) -> tuple[torch.Tensor, ...]:
    return tuple((u.roll(-1, a) - u.roll(1, a)) / (2 * h)
                 for a, h in zip(range(u.ndim - grid.dimension, u.ndim), grid.spacing))


def fd_divergence(vector: tuple[torch.Tensor, ...], grid: GridSpec) -> torch.Tensor:
    if len(vector) != grid.dimension:
        raise ValueError("Divergence requires one component per spatial axis")
    return sum(fd_gradient(v, grid)[j] for j, v in enumerate(vector))


def fd_laplacian(u: torch.Tensor, grid: GridSpec) -> torch.Tensor:
    return sum((u.roll(-1, a) - 2 * u + u.roll(1, a)) / h**2
               for a, h in zip(range(u.ndim - grid.dimension, u.ndim), grid.spacing))


def spectral_laplacian(u: torch.Tensor, grid: GridSpec) -> torch.Tensor:
    axes = tuple(range(u.ndim - grid.dimension, u.ndim))
    k = torch.meshgrid(*(2 * torch.pi * torch.fft.fftfreq(n, d=h, device=u.device, dtype=u.dtype)
                         for n, h in zip(grid.shape, grid.spacing)), indexing="ij")
    return torch.fft.ifftn(-sum(ki**2 for ki in k) * torch.fft.fftn(u, dim=axes), dim=axes).real


def time_derivative(u: torch.Tensor, grid: GridSpec) -> torch.Tensor:
    # Time axis is always [B,C,T,*space], never a convolutional spatial axis.
    if len(grid.times) < 3:
        raise ValueError("Second order physical-time residuals require >=3 saved times")
    times = torch.tensor(grid.times, dtype=u.dtype, device=u.device)
    return torch.gradient(u, spacing=(times,), dim=(2,), edge_order=2)[0]


def wave_numbers_squared(grid: GridSpec) -> np.ndarray:
    k = np.meshgrid(*(2 * np.pi * np.fft.fftfreq(n, d=h)
                     for n, h in zip(grid.shape, grid.spacing)), indexing="ij")
    return sum(ki**2 for ki in k)


def heat_reference(initial: np.ndarray, diffusivity: float, grid: GridSpec) -> np.ndarray:
    if diffusivity <= 0 or not np.isfinite(diffusivity) or initial.shape != grid.shape:
        raise ValueError("Reference heat solve needs positive diffusivity and matching initial grid")
    coeff = np.fft.fftn(initial)
    k2 = wave_numbers_squared(grid)
    return np.stack([np.fft.ifftn(coeff * np.exp(-diffusivity * k2 * (t - grid.times[0]))).real
                     for t in grid.times])


def poisson_reference(forcing: np.ndarray, grid: GridSpec, mean_tolerance: float = 1e-10) -> np.ndarray:
    """Solve -Delta u=f with strict compatible forcing and zero-mean gauge."""
    if forcing.shape != grid.shape or not np.isfinite(forcing).all():
        raise ValueError("Poisson forcing must be finite and match grid")
    if abs(forcing.mean()) > mean_tolerance:
        raise ValueError("Periodic Poisson requires mean-zero forcing; no implicit projection")
    k2 = wave_numbers_squared(grid)
    divisor = np.where(k2 == 0, 1., k2)
    coeff = np.fft.fftn(forcing) / divisor
    coeff.flat[0] = 0.
    return np.fft.ifftn(coeff).real


def smooth_initial(rng: np.random.Generator, grid: GridSpec, bandwidth: float = 1.2) -> np.ndarray:
    modes = np.meshgrid(*(np.fft.fftfreq(n) * n for n in grid.shape), indexing="ij")
    filt = np.exp(-sum(m**2 for m in modes) / (2 * bandwidth**2))
    # Deterministic expected variance scaling; no per-trajectory target statistics.
    scale = np.sqrt(np.prod(grid.shape) / np.sum(filt**2))
    return np.fft.ifftn(np.fft.fftn(rng.normal(size=grid.shape)) * filt).real * scale
