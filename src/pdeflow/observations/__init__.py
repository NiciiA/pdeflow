"""Differentiable on-grid point sensors, masks and noise metadata."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
import torch

from pdeflow.core import GridSpec


@dataclass(frozen=True)
class PointOperator:
    grid: GridSpec
    indices: torch.Tensor  # [measurement,channel,time,*spatial] integer node indices
    channels: int = 1

    def __post_init__(self):
        if self.indices.ndim != 2 or self.indices.shape[1] != 2 + self.grid.dimension or self.indices.dtype != torch.long:
            raise ValueError("Point indices must be int64 [measurement,channel,time,*space]")
        limits = self.indices.new_tensor([self.channels, len(self.grid.times), *self.grid.shape])
        if self.indices.numel() == 0 or torch.any(self.indices < 0) or torch.any(self.indices >= limits):
            raise ValueError("Observation indices out of bounds or empty")
        if len(torch.unique(self.indices, dim=0)) != len(self.indices):
            raise ValueError("Duplicate or conflicting observation indices")

    def apply(self, physical_state: torch.Tensor) -> torch.Tensor:
        if tuple(physical_state.shape[1:]) != (self.channels, len(self.grid.times), *self.grid.shape):
            raise ValueError("Observation operator grid/channel/time layout mismatch")
        ix = self.indices.to(physical_state.device)
        return physical_state[(slice(None), *ix.unbind(dim=1))]

    def coordinates(self):
        coords = self.grid.coordinates()
        return {"fields": self.indices[:, 0].tolist(),
                "physical_times": None if self.grid.temporal_kind == "static" else [self.grid.times[t] for t in self.indices[:, 1].tolist()],
                "spatial": [[float(coords[j][row[2 + j]]) for j in range(self.grid.dimension)] for row in self.indices]}


@dataclass(frozen=True)
class ObservationSet:
    operator: PointOperator
    values: torch.Tensor  # [problem,measurement]
    mask: torch.Tensor  # same shape, True denotes measured (including observed zero)
    noise_std: float
    noiseless_tolerance: float = .1
    field_noise_std: tuple[float, ...] | None = None
    field_tolerance: tuple[float, ...] | None = None

    measurement_noise_std: tuple[float, ...] | None = None

    def __post_init__(self):
        if self.measurement_noise_std is not None and (len(self.measurement_noise_std)!=len(self.operator.indices) or any(not isfinite(x) or x<0 for x in self.measurement_noise_std)):raise ValueError("Per-observation absolute sigma must match measurement count and be finite nonnegative")
        if self.values.ndim != 2 or self.values.shape[1] != len(self.operator.indices) or self.mask.shape != self.values.shape or self.mask.dtype != torch.bool:
            raise ValueError("Observation value/mask layout mismatch")
        if not torch.isfinite(self.values).all() or not isfinite(self.noise_std) or not isfinite(self.noiseless_tolerance) or self.noise_std < 0 or self.noiseless_tolerance <= 0 or not self.mask.any(1).all():
            raise ValueError("Observation values/scales must be finite with at least one measurement per problem")

        for scales in (self.field_noise_std, self.field_tolerance):
            if scales is not None and (len(scales) != self.operator.channels or any(not isfinite(x) or x < 0 for x in scales)):
                raise ValueError("Per-field scales must be finite, nonnegative and match channels")
        if self.field_tolerance is not None and min(self.field_tolerance) <= 0:
            raise ValueError("Per-field tolerances must be positive")

    def scales(self, tensor):
        if self.measurement_noise_std is not None:
            tolerance=self.field_tolerance or (self.noiseless_tolerance,)*self.operator.channels
            floor=tensor.new_tensor(tolerance)[self.operator.indices[:,0].to(tensor.device)]
            return torch.maximum(tensor.new_tensor(self.measurement_noise_std),floor)
        noise = self.field_noise_std or (self.noise_std,) * self.operator.channels
        tolerance = self.field_tolerance or (self.noiseless_tolerance,) * self.operator.channels
        return tensor.new_tensor([max(a,b) for a,b in zip(noise,tolerance)])[self.operator.indices[:,0].to(tensor.device)]

    def per_problem_energy(self, physical: torch.Tensor) -> torch.Tensor:
        pred = self.operator.apply(physical)
        values, mask = self.values.to(pred), self.mask.to(pred.device)
        if values.shape != pred.shape:
            raise ValueError("Observation problem batch mismatch")
        scale = self.scales(pred)
        return .5 * (((pred - values) / scale).square() * mask).sum(1) / mask.sum(1)

    def to_dict(self):
        return {"schema_version": 1, "indices": self.operator.indices.tolist(), "values": self.values.tolist(),
                "mask": self.mask.tolist(), "noise_std": self.noise_std, "noiseless_tolerance": self.noiseless_tolerance,
                "measurement_noise_std": list(self.measurement_noise_std) if self.measurement_noise_std is not None else None, "field_noise_std": self.field_noise_std, "field_tolerance": self.field_tolerance, "coordinates": self.operator.coordinates(), "operator": "on_grid_point", "channels": self.operator.channels}

    @classmethod
    def from_dict(cls, v, grid):
        if v["schema_version"] != 1 or v["operator"] != "on_grid_point":
            raise ValueError("Unsupported observation schema/operator")
        op = PointOperator(grid, torch.tensor(v["indices"], dtype=torch.long), v["channels"])
        coordinates = v.get("coordinates")
        # Early Darcy exports stored the static internal index as zero in this
        # metadata slot. Accept exactly that legacy encoding without changing
        # any sensor coordinates or introducing a physical-time axis.
        if grid.temporal_kind == "static" and isinstance(coordinates, dict) and coordinates.get("physical_times") == [0.] * len(op.indices):
            coordinates = {**coordinates, "physical_times": None}
        if coordinates != op.coordinates():
            raise ValueError("Observation physical coordinate metadata does not match grid indices")
        return cls(op, torch.tensor(v["values"], dtype=torch.float64), torch.tensor(v["mask"], dtype=torch.bool), v["noise_std"], v["noiseless_tolerance"], tuple(v["field_noise_std"]) if v.get("field_noise_std") else None, tuple(v["field_tolerance"]) if v.get("field_tolerance") else None, tuple(v["measurement_noise_std"]) if v.get("measurement_noise_std") is not None else None)


def sensor_observations(reference: torch.Tensor, grid: GridSpec, sensors: int, seed: int, noise_std: float, tolerance=.1) -> ObservationSet:
    """Sample spatial sensors at all saved times. Sensor choice never inspects truth."""
    sensor_rng, noise_rng = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2)]
    flat = sensor_rng.choice(int(np.prod(grid.shape)), sensors, replace=False)
    positions = np.stack(np.unravel_index(flat, grid.shape), axis=1)
    indices = torch.tensor([[c, t, *p] for c in range(reference.shape[1]) for t in range(len(grid.times)) for p in positions], dtype=torch.long)
    operator = PointOperator(grid, indices, reference.shape[1])
    values = operator.apply(reference).double()
    noise = torch.from_numpy(noise_rng.normal(size=tuple(values.shape))) * noise_std
    return ObservationSet(operator, values + noise, torch.ones_like(values, dtype=torch.bool), noise_std, tolerance)
