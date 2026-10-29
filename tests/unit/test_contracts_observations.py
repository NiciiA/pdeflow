from dataclasses import replace

import pytest
import torch

from pdeflow.core import Capabilities, DomainSpec, FieldBatch, FieldSpec, GridSpec, TransformBundle
from pdeflow.equations import Expr, heat_problem
from pdeflow.models import TimeChannelLayout
from pdeflow.observations import ObservationSet, PointOperator, sensor_observations
from pdeflow.runtime import Config, RuntimeSpec


def grid(d=2):
    return GridSpec((8,) * d, DomainSpec((1.,) * d), (0., .05, .1, .15))


def test_transform_uses_only_explicit_training_and_physical_roundtrip():
    g = grid()
    train = torch.arange(2 * 4 * 64, dtype=torch.float64).reshape(2, 1, 4, 8, 8)
    transform = TransformBundle.fit(train, ["train-a", "train-b"])
    test = torch.full((1, 1, 4, 8, 8), 100000., dtype=torch.float64)
    mean_before = transform.mean
    assert torch.allclose(transform.decode(transform.encode(test)), test)
    assert transform.mean == mean_before == (train.mean().item(),)
    assert abs(transform.encode(train).mean().item()) < 1e-14
    assert transform.fitted_ids == ("train-a", "train-b")
    FieldBatch(train, g, (FieldSpec("u"),))
    with pytest.raises(ValueError, match="Expected physical"):
        FieldBatch(train[:, :, :2], g, (FieldSpec("u"),))


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_time_packing_and_periodic_coordinates(dimension):
    g = grid(dimension)
    x = torch.randn(2, 2, 4, *g.shape)
    layout = TimeChannelLayout(2, g)
    assert torch.equal(layout.unpack(layout.pack(x)), x)
    assert layout.pack(x).shape[1] == 8
    for coordinates in g.coordinates():
        assert coordinates[0] == 0 and coordinates[-1] == 7 / 8
    assert g.cell_volume == (1 / 8)**dimension
    assert abs(g.time_weights().sum().item() - .15) < 1e-14


def test_observed_zero_mask_and_sensor_gradient():
    g = grid(1)
    op = PointOperator(g, torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long))
    x = torch.ones(1, 1, 4, 8, dtype=torch.float64, requires_grad=True)
    obs = ObservationSet(op, torch.zeros(1, 2, dtype=torch.float64), torch.tensor([[True, False]]), 0., .1)
    energy = obs.per_problem_energy(x).sum()
    gradient, = torch.autograd.grad(energy, x)
    assert energy.item() == pytest.approx(50.)
    assert gradient[0, 0, 1, 2] == pytest.approx(100.)
    assert torch.count_nonzero(gradient) == 1
    eps = 1e-6
    direction = torch.zeros_like(x); direction[0, 0, 1, 2] = 1
    finite_diff = (obs.per_problem_energy(x + eps * direction) - obs.per_problem_energy(x - eps * direction)) / (2 * eps)
    assert finite_diff.item() == pytest.approx(100., rel=1e-8)
    assert ObservationSet.from_dict(obs.to_dict(), g).to_dict() == obs.to_dict()
    # Sensor placement depends on RNG/grid, never on target amplitude.
    a = sensor_observations(x.detach(), g, 2, 17, .03)
    b = sensor_observations(99 * x.detach(), g, 2, 17, .03)
    assert torch.equal(a.operator.indices, b.operator.indices)


def test_safe_expression_serialization_and_invalid_combinations():
    g = grid()
    p = heat_problem(g, .1)
    assert Expr.from_dict(p.operator_spec).to_dict() == p.operator_spec
    with pytest.raises(ValueError, match="operator"):
        Expr.from_dict({"op": "eval", "name": "print(42)"})
    with pytest.raises(ValueError, match="known and unknown"):
        replace(p, unknown_parameters=("diffusivity",))
    with pytest.raises(ValueError, match="positive"):
        heat_problem(g, 0.)
    with pytest.raises(ValueError, match="periodic"):
        DomainSpec((1.,), boundary="neumann")
    with pytest.raises(ValueError, match="capability"):
        Capabilities().validate(replace(p, family="darcy"), "independent_gaussian_linear_cfm", "euler", "uniform")
    with pytest.raises(ValueError, match="ODE steppers"):
        Capabilities().validate(p, "independent_gaussian_linear_cfm", "euler_maruyama", "uniform")
    with pytest.raises(ValueError, match="full precision"):
        RuntimeSpec(precision="amp").resolve()
    with pytest.raises(ValueError, match="Unknown configuration"):
        Config.from_dict({"grid": {"shape": [8], "domain": {"lengths": [1.]}, "times": [0., .1, .2]}, "typo": 1})


def test_poisson_expression_is_evaluation_not_neural_conditioning():
    g = grid(1)
    x = g.coordinates()[0]
    field = torch.cos(2 * torch.pi * x)[None, None, None]
    forcing = (2 * torch.pi)**2 * field
    residual = Expr("add", (Expr("multiply", (Expr("constant", value=-1), Expr("laplacian", (Expr("field", name="u"),)))),
                            Expr("multiply", (Expr("constant", value=-1), Expr("field", name="f")))))
    out = residual.evaluate({"u": field, "f": forcing}, {}, g, "spectral")
    assert out.abs().max() < 1e-12
