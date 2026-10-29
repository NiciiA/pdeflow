"""Heuristic split endpoint-gradient steering, distinct from ODE integration."""
from dataclasses import dataclass
from typing import Callable

import torch

from pdeflow.core import ObservationOperator


@dataclass
class ReconstructionEnergy:
    observations: object
    physics: object
    decode: Callable
    observation_weight: float
    physics_weight: float

    def __call__(self, endpoint: torch.Tensor) -> torch.Tensor:
        physical = self.decode(endpoint)
        energy = self.observation_weight * self.observations.per_problem_energy(physical)
        if self.physics_weight:
            energy = energy + .5 * self.physics_weight * self.physics.evaluate(physical).per_problem_mean_square()
        return energy  # one scalar per independent candidate; no population coupling


@dataclass
class SplitGradientGuidance:
    energy: Callable
    strength: float
    max_norm: float

    def correct(self, proposal: torch.Tensor, generation_time: float, step_size: float, endpoint: Callable):
        with torch.enable_grad():
            x = proposal.detach().requires_grad_(True)
            estimate = endpoint(x, generation_time)
            e = self.energy(estimate)
            gradient, = torch.autograd.grad(e.sum(), x)
            if not torch.isfinite(gradient).all() or not torch.isfinite(e).all():
                raise FloatingPointError("Nonfinite guidance energy/gradient")
            norm = gradient.flatten(1).norm(dim=1)
            clip = (self.max_norm / norm.clamp_min(1e-12)).clamp_max(1)
            gradient = gradient * clip.reshape(-1, *([1] * (gradient.ndim - 1)))
            corrected = x - self.strength * step_size * gradient
        return corrected.detach(), {"energy_before": e.detach().cpu().tolist(), "gradient_norm": norm.detach().cpu().tolist()}
