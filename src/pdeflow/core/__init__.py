"""Serializable physical contracts. No model or PDE dependencies."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import isfinite, prod
from typing import Any, Protocol

import torch


@dataclass(frozen=True)
class DomainSpec:
    lengths: tuple[float, ...]
    origins: tuple[float, ...] = ()
    boundary: str = "periodic"
    geometry: str = "cartesian"

    def __post_init__(self):
        object.__setattr__(self, "lengths", tuple(self.lengths))
        object.__setattr__(self, "origins", tuple(self.origins) or (0.,) * len(self.lengths))
        if len(self.lengths) not in (1, 2, 3) or len(self.origins) != len(self.lengths):
            raise ValueError("Domain requires 1–3 spatial axes and matching origins")
        if any(not isfinite(x) or x <= 0 for x in self.lengths) or not all(map(isfinite, self.origins)):
            raise ValueError("Domain lengths must be finite positive and origins finite")
        if (self.boundary, self.geometry) != ("periodic", "cartesian"):
            raise ValueError("Only periodic Cartesian domains are implemented")


@dataclass(frozen=True)
class GridSpec:
    shape: tuple[int, ...]
    domain: DomainSpec
    times: tuple[float, ...]
    convention: str = "periodic_nodes_endpoint_excluded"
    temporal_kind: str = "transient"

    def __post_init__(self):
        object.__setattr__(self, "shape", tuple(self.shape))
        object.__setattr__(self, "times", tuple(self.times))
        if len(self.shape) != len(self.domain.lengths) or any(type(n) is not int or n < 4 for n in self.shape):
            raise ValueError("Grid requires matching 1–3 spatial axes, each with >=4 nodes")
        if self.convention not in ("periodic_nodes_endpoint_excluded", "periodic_cell_centers"):
            raise ValueError("Unsupported coordinate convention")
        if not self.times or not all(map(isfinite, self.times)) or any(b <= a for a, b in zip(self.times, self.times[1:])):
            raise ValueError("Physical times must be finite and strictly increasing")
        if self.temporal_kind not in ("transient", "static") or (self.temporal_kind == "static" and self.times != (0.,)):
            raise ValueError("Static T=1 uses only an internal zero index, not physical time")

    @property
    def dimension(self) -> int:
        return len(self.shape)

    @property
    def axis_names(self) -> tuple[str, ...]:
        return ("x", "y", "z")[:self.dimension]

    @property
    def spacing(self) -> tuple[float, ...]:
        return tuple(l / n for l, n in zip(self.domain.lengths, self.shape))

    @property
    def cell_volume(self) -> float:
        return prod(self.spacing)

    def coordinates(self, *, dtype=torch.float64, device="cpu") -> tuple[torch.Tensor, ...]:
        return tuple(o + (torch.arange(n, dtype=dtype, device=device) + (.5 if self.convention == "periodic_cell_centers" else 0)) * h
                     for o, n, h in zip(self.domain.origins, self.shape, self.spacing))

    def time_weights(self, *, dtype=torch.float64, device="cpu") -> torch.Tensor:
        t = torch.tensor(self.times, dtype=dtype, device=device)
        if len(t) == 1:
            return torch.ones_like(t)
        dt = t[1:] - t[:-1]
        return torch.cat((dt[:1] / 2, (dt[:-1] + dt[1:]) / 2, dt[-1:] / 2))

    @classmethod
    def from_dict(cls, value: dict) -> GridSpec:
        return cls(tuple(value["shape"]), DomainSpec(**value["domain"]), tuple(value["times"]), value.get("convention", "periodic_nodes_endpoint_excluded"), value.get("temporal_kind", "transient"))


@dataclass(frozen=True)
class FieldSpec:
    name: str
    units: str = "nondimensional"
    role: str = "unknown"

    def __post_init__(self):
        if not self.name or self.role not in ("known", "unknown"):
            raise ValueError("Fields require a name and explicit known/unknown role")


@dataclass(frozen=True)
class ProblemSpec:
    family: str
    grid: GridSpec
    fields: tuple[FieldSpec, ...]
    known_parameters: dict[str, float]
    operator_spec: dict[str, Any]
    unknown_parameters: tuple[str, ...] = ()
    initial_condition: str = "unknown"
    schema_version: int = 1
    known_fields: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.schema_version != 1:
            raise ValueError("Unsupported problem schema")
        names = [x.name for x in self.fields]
        if len(set(names)) != len(names) or not names:
            raise ValueError("Field names must be nonempty and unique")
        if set(self.known_parameters) & set(self.unknown_parameters):
            raise ValueError("A parameter cannot be both known and unknown")
        if not all(isfinite(v) for v in self.known_parameters.values()):
            raise ValueError("Parameters must be finite")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, v: dict) -> ProblemSpec:
        if set(v) - {"family", "grid", "fields", "known_parameters", "operator_spec", "unknown_parameters", "initial_condition", "schema_version", "known_fields"}:
            raise ValueError("Unknown problem keys; references and hidden fields are not inference inputs")
        return cls(v["family"], GridSpec.from_dict(v["grid"]), tuple(FieldSpec(**f) for f in v["fields"]),
                   v["known_parameters"], v["operator_spec"], tuple(v.get("unknown_parameters", ())),
                   v.get("initial_condition", "unknown"), v.get("schema_version", 1), v.get("known_fields", {}))


@dataclass
class FieldBatch:
    values: torch.Tensor  # physical [problem,C,T,*space]
    grid: GridSpec
    fields: tuple[FieldSpec, ...]

    def __post_init__(self):
        expected = (len(self.fields), len(self.grid.times), *self.grid.shape)
        if tuple(self.values.shape[1:]) != expected:
            raise ValueError(f"Expected physical [problem,{expected}], got {tuple(self.values.shape)}")
        if not torch.isfinite(self.values).all():
            raise ValueError("Physical fields must be finite")


@dataclass
class StateBundle:
    fields: FieldBatch
    scalar_unknowns: dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass(frozen=True)
class TransformBundle:
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    fitted_ids: tuple[str, ...]

    def __post_init__(self):
        if len(self.mean) != len(self.scale) or not self.mean or any(not isfinite(s) or s <= 0 for s in self.scale):
            raise ValueError("Invalid field transform")
        if not all(map(isfinite, self.mean)) or not self.fitted_ids:
            raise ValueError("Transform requires finite means and training provenance")

    @classmethod
    def fit(cls, training: torch.Tensor, ids: list[str]) -> TransformBundle:
        if training.shape[0] != len(ids) or not ids:
            raise ValueError("Transform fit requires exactly the training IDs")
        axes = (0, *range(2, training.ndim))
        mean = training.double().mean(axes)
        scale = training.double().std(axes, correction=0).clamp_min(1e-8)
        return cls(tuple(mean.tolist()), tuple(scale.tolist()), tuple(ids))

    def _stats(self, x):
        if x.shape[1] != len(self.mean):
            raise ValueError("Transform field/channel mismatch")
        shape = (1, len(self.mean), *([1] * (x.ndim - 2)))
        return (x.new_tensor(self.mean).reshape(shape), x.new_tensor(self.scale).reshape(shape))

    def encode(self, physical: torch.Tensor) -> torch.Tensor:
        mean, scale = self._stats(physical)
        return (physical - mean) / scale

    def decode(self, model: torch.Tensor) -> torch.Tensor:
        mean, scale = self._stats(model)
        return model * scale + mean


@dataclass
class Conditioning:
    family: str
    known_parameters: dict[str, torch.Tensor]  # [problem], never hidden fields
    known_fields: dict[str, torch.Tensor] = field(default_factory=dict)

    def select(self, indices) -> Conditioning:
        return Conditioning(self.family, {name: value[indices] for name, value in self.known_parameters.items()}, {name: value[indices] for name, value in self.known_fields.items()})

    def to(self, *, device, dtype) -> Conditioning:
        return Conditioning(self.family, {name: value.to(device=device, dtype=dtype) for name, value in self.known_parameters.items()}, {name: value.to(device=device, dtype=dtype) for name, value in self.known_fields.items()})

    @classmethod
    def from_problem(cls, problem: ProblemSpec, candidates: int, *, device="cpu", dtype=torch.float64) -> Conditioning:
        return cls(problem.family, {name: torch.full((candidates,), value, device=device, dtype=dtype)
                                    for name, value in problem.known_parameters.items()}, {name: torch.tensor(value,device=device,dtype=dtype).expand(candidates,*problem.grid.shape) for name,value in problem.known_fields.items()})


class ObservationOperator(Protocol):
    def apply(self, physical_state: torch.Tensor) -> torch.Tensor: ...


@dataclass
class SamplingState:
    model_state: torch.Tensor  # [problem,particle,C*T,*space]
    generation_time: float
    step_index: int
    source_rng: torch.Generator
    diagnostics: dict[str, Any] = field(default_factory=dict)
    estimated_physical_endpoint: torch.Tensor | None = None


@dataclass
class ReconstructionResult:
    samples: torch.Tensor  # physical [problem,particle,C,T,*space]
    diagnostics: dict[str, Any]
    provenance: dict[str, Any]
    stop_reason: str = "completed_schedule"
    corrected: bool = False
    inference_state: Any | None = None
    weights: torch.Tensor | None = None
    generation_snapshots: dict | None = None


@dataclass(frozen=True)
class Capabilities:
    family: str = "heat"
    dimensions: tuple[int, ...] = (1, 2, 3)
    boundary: str = "periodic"
    formulation: str = "independent_gaussian_linear_cfm"
    condition_keys: tuple[str, ...] = ("diffusivity",)
    fixed_time_window: bool = True
    unknown_parameters: bool = False

    def validate(self, problem: ProblemSpec, formulation: str, stepper: str, schedule: str):
        if problem.family != self.family or problem.grid.dimension not in self.dimensions:
            raise ValueError("Checkpoint family/dimension capability mismatch")
        if problem.grid.domain.boundary != self.boundary or formulation != self.formulation:
            raise ValueError("Boundary/formulation capability mismatch")
        if set(problem.known_parameters) != set(self.condition_keys) or problem.unknown_parameters:
            raise ValueError("Only known diffusivity conditioning is implemented; unknown parameter inference is planned")
        if problem.known_parameters["diffusivity"] <= 0:
            raise ValueError("Heat diffusivity must be positive")
        if len(problem.fields) != 1 or problem.fields[0].name != "u" or problem.fields[0].role != "unknown":
            raise ValueError("This specialist reconstructs the unknown scalar field u")
        if problem.initial_condition != "unknown":
            raise ValueError("Known initial constraints require an explicit observation operator")
        if stepper not in ("euler", "heun") or schedule not in ("uniform", "quadratic"):
            raise ValueError("Supported ODE steppers: euler/heun; schedules: uniform/quadratic")
