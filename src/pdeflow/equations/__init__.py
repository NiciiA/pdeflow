"""A safe scalar expression grammar and heat-family residual evaluator."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import torch

from pdeflow.core import FieldSpec, GridSpec, ProblemSpec
from pdeflow.numerics import fd_laplacian, spectral_laplacian, time_derivative


@dataclass(frozen=True)
class Expr:
    op: str
    args: tuple[Expr, ...] = ()
    name: str = ""
    value: float = 0.

    def __post_init__(self):
        arities = {"field": 0, "constant": 0, "coefficient": 0, "add": 2,
                   "multiply": 2, "power": 1, "time_derivative": 1, "laplacian": 1}
        if self.op not in arities or len(self.args) != arities[self.op]:
            raise ValueError(f"Unsupported scalar operator/arity: {self.op}")
        if not isfinite(self.value) or (self.op in ("field", "coefficient") and not self.name):
            raise ValueError("Invalid expression leaf")
        if self.op == "power" and self.value not in (0., 1., 2., 3.):
            raise ValueError("Only nonnegative integer powers 0–3 are implemented")

    def to_dict(self):
        return {"op": self.op, "args": [a.to_dict() for a in self.args], "name": self.name, "value": self.value}

    @classmethod
    def from_dict(cls, d: dict, depth: int = 0):
        if depth > 16 or set(d) - {"op", "args", "name", "value"}:
            raise ValueError("Expression too deep or contains unknown keys")
        return cls(d["op"], tuple(cls.from_dict(a, depth + 1) for a in d.get("args", [])), d.get("name", ""), d.get("value", 0.))

    def evaluate(self, fields: dict[str, torch.Tensor], coefficients: dict[str, torch.Tensor | float],
                 grid: GridSpec, spatial: str = "fd"):
        if spatial not in ("fd", "spectral"):
            raise ValueError("Unknown derivative backend")
        if self.op == "field":
            if self.name not in fields:
                raise ValueError(f"Unknown field {self.name}")
            return fields[self.name]
        if self.op == "coefficient":
            if self.name not in coefficients:
                raise ValueError(f"Unknown coefficient {self.name}")
            return coefficients[self.name]
        if self.op == "constant":
            return self.value
        a = [x.evaluate(fields, coefficients, grid, spatial) for x in self.args]
        if self.op == "add":
            return a[0] + a[1]
        if self.op == "multiply":
            return a[0] * a[1]
        if self.op == "power":
            return a[0] ** self.value
        if self.op == "time_derivative":
            return time_derivative(a[0], grid)
        return (fd_laplacian if spatial == "fd" else spectral_laplacian)(a[0], grid)


def heat_expression() -> Expr:
    u = Expr("field", name="u")
    return Expr("add", (Expr("time_derivative", (u,)),
                       Expr("multiply", (Expr("constant", value=-1.),
                                         Expr("multiply", (Expr("coefficient", name="diffusivity"), Expr("laplacian", (u,))))))))


def heat_problem(grid: GridSpec, diffusivity: float) -> ProblemSpec:
    if not isfinite(diffusivity) or diffusivity <= 0:
        raise ValueError("Known diffusivity must be finite and positive")
    return ProblemSpec("heat", grid, (FieldSpec("u"),), {"diffusivity": diffusivity}, heat_expression().to_dict())


@dataclass
class ResidualBundle:
    pointwise: dict[str, torch.Tensor]
    units: dict[str, str]
    time_weights: torch.Tensor

    def mean_square(self, name="pde") -> torch.Tensor:
        r = self.pointwise[name]
        w = self.time_weights.reshape(1, 1, -1, *([1] * (r.ndim - 3)))
        return (r.square() * w).sum() / (w.expand_as(r).sum())

    def per_problem_mean_square(self, name="pde") -> torch.Tensor:
        r = self.pointwise[name]
        w = self.time_weights.reshape(1, 1, -1, *([1] * (r.ndim - 3))).expand_as(r)
        axes = tuple(range(1, r.ndim))
        return (r.square() * w).sum(axes) / w.sum(axes)


@dataclass
class HeatPhysics:
    grid: GridSpec
    diffusivity: torch.Tensor
    spatial: str = "fd"

    def evaluate(self, physical: torch.Tensor) -> ResidualBundle:
        d = self.diffusivity.to(physical).reshape(-1, *([1] * (physical.ndim - 1)))
        if d.shape[0] != physical.shape[0] or torch.any(d <= 0):
            raise ValueError("Diffusivity requires one positive known value per problem")
        r = heat_expression().evaluate({"u": physical}, {"diffusivity": d}, self.grid, self.spatial)[:, :, 1:-1]
        w = self.grid.time_weights(dtype=physical.dtype, device=physical.device)[1:-1]
        return ResidualBundle({"pde": r}, {"pde": "u / physical_time (nondimensional)"}, w)


def make_physics(problem: ProblemSpec, known_parameters: dict[str, torch.Tensor], spatial="fd") -> HeatPhysics:
    if problem.family != "heat" or problem.operator_spec != heat_expression().to_dict():
        raise ValueError("No physics factory for this family/operator specification")
    if set(known_parameters) != {"diffusivity"}:
        raise ValueError("Heat physics requires exactly the known diffusivity")
    return HeatPhysics(problem.grid, known_parameters["diffusivity"], spatial)
