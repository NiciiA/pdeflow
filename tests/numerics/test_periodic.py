import numpy as np
import pytest
import torch

from pdeflow.core import DomainSpec, GridSpec
from pdeflow.equations import HeatPhysics
from pdeflow.numerics import (fd_divergence, fd_gradient, fd_laplacian, heat_reference,
                              poisson_reference, smooth_initial, spectral_laplacian, time_derivative)


def manufactured(d, n, time_count=5):
    grid = GridSpec((n,) * d, DomainSpec(tuple(1. + j / 3 for j in range(d)), (.2,) * d),
                    tuple(np.linspace(0, .02, time_count)))
    coords = torch.meshgrid(*grid.coordinates(), indexing="ij")
    factors = [torch.cos(2 * torch.pi * (x - .2) / l) for x, l in zip(coords, grid.domain.lengths)]
    initial = torch.stack(factors).prod(0)
    k2 = sum((2 * np.pi / l)**2 for l in grid.domain.lengths)
    return grid, initial, k2, coords


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_spatial_operators_converge_and_spectral_is_exact(dimension):
    errors = []
    gradient_errors = []
    for n in (8, 16, 32):
        grid, u, k2, coords = manufactured(dimension, n)
        errors.append((fd_laplacian(u, grid) + k2 * u).square().mean().sqrt().item())
        exact_gradient = []
        for a in range(dimension):
            terms = [(-2 * np.pi / l * torch.sin(2 * torch.pi * (x - .2) / l)) if j == a
                     else torch.cos(2 * torch.pi * (x - .2) / l)
                     for j, (x, l) in enumerate(zip(coords, grid.domain.lengths))]
            exact_gradient.append(torch.stack(terms).prod(0))
        gradient_errors.append(sum((g - e).square().mean().item() for g, e in zip(fd_gradient(u, grid), exact_gradient))**.5)
        assert torch.allclose(spectral_laplacian(u, grid), -k2 * u, atol=2e-10, rtol=1e-10)
        # Divergence checked on analytic vector components, independent of grad implementation.
        divergence = fd_divergence(tuple(coords[j].new_tensor(1.) * torch.sin(2 * torch.pi * (coords[j] - .2) / grid.domain.lengths[j])
                                         for j in range(dimension)), grid)
        analytic = sum(2 * torch.pi / l * torch.cos(2 * torch.pi * (x - .2) / l) for x, l in zip(coords, grid.domain.lengths))
        assert (divergence - analytic).abs().max() < .65 * dimension
    assert errors[0] / errors[1] > 3.8 and errors[1] / errors[2] > 3.8
    assert gradient_errors[0] / gradient_errors[1] > 3.8


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_heat_reference_analytic_decay_and_mass(dimension):
    grid, u, k2, _ = manufactured(dimension, 12)
    d = .07
    trajectory = heat_reference((u + .4).numpy(), d, grid)
    for j, t in enumerate(grid.times):
        assert np.max(np.abs(trajectory[j] - (.4 + u.numpy() * np.exp(-d * k2 * t)))) < 1e-12
    assert np.max(np.abs(trajectory.mean(tuple(range(1, dimension + 1))) - .4)) < 1e-13
    a = smooth_initial(np.random.default_rng(19), grid)
    b = smooth_initial(np.random.default_rng(19), grid)
    assert np.array_equal(a, b)
    # Genuine variation along every physical spatial axis.
    for axis in range(dimension):
        assert np.max(np.abs(a - np.roll(a, 1, axis))) > 1e-3


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_heat_residual_convergence_against_analytic_solution(dimension):
    errors = []
    for n in (8, 16, 32):
        grid, initial, k2, _ = manufactured(dimension, n, n + 1)
        times = torch.tensor(grid.times, dtype=torch.float64)
        exact = initial * torch.exp(-.07 * k2 * times).reshape(-1, *([1] * dimension))
        residual = HeatPhysics(grid, torch.tensor([.07], dtype=torch.float64)).evaluate(exact[None, None])
        errors.append(residual.mean_square().sqrt().item())
    assert errors[0] / errors[1] > 3.8 and errors[1] / errors[2] > 3.8


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_poisson_compatibility_gauge_and_manufactured_solution(dimension):
    grid, exact, k2, _ = manufactured(dimension, 16)
    forcing = k2 * exact.numpy()
    solution = poisson_reference(forcing, grid)
    assert abs(solution.mean()) < 1e-14
    assert np.max(np.abs(solution - exact.numpy())) < 1e-12
    with pytest.raises(ValueError, match="mean-zero"):
        poisson_reference(forcing + .1, grid)


def test_nonuniform_physical_time_polynomial_derivative():
    grid = GridSpec((4, 4, 4), DomainSpec((1., 1., 1.)), (0., .01, .04, .1))
    t = torch.tensor(grid.times, dtype=torch.float64)
    u = (t**2).reshape(1, 1, 4, 1, 1, 1).expand(1, 1, 4, 4, 4, 4)
    expected = (2 * t).reshape(1, 1, 4, 1, 1, 1).expand_as(u)
    assert torch.allclose(time_derivative(u, grid), expected, atol=1e-15, rtol=1e-14)
