import math

import pytest
import torch

from pdeflow.core import Conditioning, DomainSpec, GridSpec
from pdeflow.equations import HeatPhysics
from pdeflow.generative import LinearCFM
from pdeflow.models import TimeChannelLayout, make_model
from pdeflow.observations import ObservationSet, PointOperator
from pdeflow.runtime import ModelConfig
from pdeflow.sampling import generation_schedule, make_stepper
from pdeflow.steering import ReconstructionEnergy, SplitGradientGuidance


@pytest.mark.parametrize("name,order,calls", [("euler", 1, 1), ("heun", 2, 2)])
def test_ode_order_and_model_counts(name, order, calls):
    errors = []
    for n in (10, 20, 40):
        state = torch.ones(1, dtype=torch.float64)
        counter = [0]

        def dynamics(x, s):
            counter[0] += 1
            return x

        reported = 0
        stepper = make_stepper(name)
        for j in range(n):
            result = stepper.advance(dynamics, state, j / n, 1 / n)
            state = result.state
            reported += result.evaluations
        assert counter[0] == reported == n * calls
        errors.append(abs(state.item() - math.e))
    assert errors[-2] / errors[-1] == pytest.approx(2**order, rel=.07)


@pytest.mark.parametrize("name", ["uniform", "quadratic"])
def test_schedule_endpoints_and_monotonicity(name):
    s = generation_schedule(name, 12)
    assert s[0] == 0 and s[-1] == 1
    assert all(b > a for a, b in zip(s, s[1:]))


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_real_backbone_and_endpoint_model_jacobian(dimension):
    torch.manual_seed(8)
    grid = GridSpec((4,) * dimension, DomainSpec((1.,) * dimension), (0., .05, .1))
    model = make_model(ModelConfig(width=4, blocks=1), TimeChannelLayout(1, grid)).double()
    for p in model.parameters():
        p.requires_grad_(False)
    x = torch.randn(1, 3, *grid.shape, dtype=torch.float64, requires_grad=True)
    cond = Conditioning("heat", {"diffusivity": torch.tensor([.1], dtype=torch.float64)})
    op = PointOperator(grid, torch.tensor([[0, 1, *([1] * dimension)]], dtype=torch.long))
    obs = ObservationSet(op, torch.tensor([[.3]], dtype=torch.float64), torch.ones(1, 1, dtype=torch.bool), .1)
    layout = model.layout
    energy = ReconstructionEnergy(obs, HeatPhysics(grid, cond.known_parameters["diffusivity"]), layout.unpack, 1., .0001)

    def endpoint(z, s):
        return LinearCFM().endpoint(z, s, model(z, z.new_full((1,), s), cond))

    loss = energy(endpoint(x, .4)).sum()
    gradient, = torch.autograd.grad(loss, x)
    direction = torch.randn_like(x); direction = direction / direction.norm()
    eps = 1e-6
    finite_difference = (energy(endpoint(x + eps * direction, .4)).sum() - energy(endpoint(x - eps * direction, .4)).sum()) / (2 * eps)
    assert (gradient * direction).sum().item() == pytest.approx(finite_difference.item(), rel=2e-6, abs=1e-7)
    # Check that detaching the model output would remove a nonzero Jacobian term.
    z = x.detach().clone().requires_grad_(True)
    detached_endpoint = z + .6 * model(z, z.new_full((1,), .4), cond).detach()
    wrong_grad, = torch.autograd.grad(energy(detached_endpoint).sum(), z)
    assert (gradient - wrong_grad).norm() > 1e-5
    steering = SplitGradientGuidance(energy, .001, 1e6)
    new, diagnostic = steering.correct(x, .4, 1., endpoint)
    assert energy(endpoint(new, .4)).item() < loss.item()
    assert diagnostic["gradient_norm"][0] > 0


def test_cfm_source_and_generation_endpoints():
    target = torch.full((3, 2, 4), 2., dtype=torch.float64)
    formulation = LinearCFM()
    source = torch.Generator().manual_seed(7)
    time = torch.Generator().manual_seed(9)
    x, velocity, s = formulation.training_state(target, source, time)
    assert s.shape == (3,) and torch.all((s >= 0) & (s < 1))
    # Along a conditional straight line the endpoint estimate is exactly the supplied endpoint.
    assert torch.allclose(formulation.endpoint(x, s.reshape(3, 1, 1), velocity), target)
    assert torch.equal(formulation.endpoint(target, 1., velocity), target)


def test_real_configuration_substitutions_preserve_physical_problem(tmp_path):
    from dataclasses import replace
    from pdeflow.runtime import Config,RuntimeSpec
    from pdeflow.training import load_inference,train
    from pdeflow.data import generate
    from pdeflow.core import ProblemSpec
    from pdeflow.observations import ObservationSet
    from pdeflow.runtime import read_json
    from pdeflow.sampling import sample
    config=Config.load('configs/experiments/cpu_2d.json')
    config=replace(config,data=replace(config.data,count=6),model=replace(config.model,width=4,blocks=1),training=replace(config.training,steps=1))
    generate(config,tmp_path/'data');train(config,tmp_path/'data',tmp_path/'train')
    bundle=load_inference(tmp_path/'train/inference.pt',runtime=RuntimeSpec())
    problem=ProblemSpec.from_dict(read_json(tmp_path/'data/test_problem.json'))
    obs=ObservationSet.from_dict(read_json(tmp_path/'data/test_observations.json'),problem.grid)
    before=problem.to_dict()
    for stepper in ('euler','heun'):
        for schedule in ('uniform','quadratic'):
            for mode in ('unguided','observation','observation_pde'):
                inference=replace(config.inference,steps=2,particles=1,stepper=stepper,schedule=schedule)
                result=sample(bundle,problem,obs,inference,mode)
                expected=2*(1 if stepper=='euler' else 2)+(0 if mode=='unguided' else 1)
                assert result.diagnostics['model_evaluations']==expected
                assert problem.to_dict()==before
                assert result.samples.shape==(1,1,1,4,8,8)
