"""Independent Gaussian-source linear conditional flow matching."""
from typing import Protocol

import torch


class Formulation(Protocol):
    name: str

    def training_state(self, target, source_rng, time_rng): ...
    def loss(self, prediction, target): ...
    def endpoint(self, state, time, velocity): ...


class LinearCFM:
    name = "independent_gaussian_linear_cfm"

    def training_state(self, target, source_rng, time_rng):
        z = torch.randn(target.shape, generator=source_rng, dtype=target.dtype, device="cpu").to(target.device)
        s = torch.rand(target.shape[0], generator=time_rng, dtype=target.dtype, device="cpu").to(target.device)
        broadcast = s.reshape(-1, *([1] * (target.ndim - 1)))
        return (1 - broadcast) * z + broadcast * target, target - z, s

    def loss(self, prediction, target):
        return torch.nn.functional.mse_loss(prediction, target)

    def endpoint(self, state, time, velocity):
        return state + (1 - time) * velocity


def make_formulation(name: str) -> LinearCFM:
    if name != LinearCFM.name:
        raise ValueError("Only independent_gaussian_linear_cfm is implemented; diffusion is planned")
    return LinearCFM()
