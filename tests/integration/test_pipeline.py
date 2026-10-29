from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from pdeflow.cli import evaluate, reconstruct
from pdeflow.core import DomainSpec, GridSpec
from pdeflow.data import generate, load_dataset
from pdeflow.evaluation.baseline import fourier_heat_basis, inverse_baseline
from pdeflow.observations import sensor_observations
from pdeflow.runtime import (Config, DataConfig, InferenceConfig, ModelConfig, RuntimeSpec,
                             TrainingConfig, atomic_write, read_json)
from pdeflow.training import load_inference, train
from pdeflow.core import ProblemSpec
from pdeflow.observations import ObservationSet
from pdeflow.sampling import sample


def tiny_config(dimension=2):
    return Config(GridSpec((4,) * dimension, DomainSpec((1.,) * dimension), (0., .03, .06)),
                  RuntimeSpec(threads=1), DataConfig(count=9), ModelConfig(width=4, blocks=1),
                  TrainingConfig(steps=6, batch_size=2, checkpoint_every=2),
                  InferenceConfig(steps=3, particles=2, sensors=4)).validate()


def test_resume_is_exact_and_export_is_dataset_independent(tmp_path):
    config = tiny_config()
    data_path = tmp_path / "data"
    manifest = generate(config, data_path)
    data = load_dataset(data_path, config)
    ix = [*data.indices("train"), *data.indices("validation"), *data.indices("test")]
    assert len(ix) == len(set(ix)) == 9
    train(config, data_path, tmp_path / "full")
    train(config, data_path, tmp_path / "resumed", stop_after=3)
    train(config, data_path, tmp_path / "resumed", resume=tmp_path / "resumed/checkpoint.pt")
    full = torch.load(tmp_path / "full/checkpoint.pt", weights_only=True)
    resumed = torch.load(tmp_path / "resumed/checkpoint.pt", weights_only=True)
    assert full["progress"] == resumed["progress"] == 6
    assert [r["training_loss"] for r in full["history"]] == [r["training_loss"] for r in resumed["history"]]
    for name in full["model"]:
        assert torch.equal(full["model"][name], resumed["model"][name])
    for name in full["rng"]:
        assert torch.equal(full["rng"][name], resumed["rng"][name])
    assert full["scheduler"] is None and full["scaler"] is None
    train_ids = {manifest["ids"][i] for i in manifest["splits"]["train"]}
    assert set(full["transform"]["fitted_ids"]) == train_ids
    assert not train_ids.intersection(manifest["ids"][i] for i in manifest["splits"]["test"])
    first = next(iter(full["model"].values()))
    init = torch.load(tmp_path / "resumed/inference.pt", weights_only=True)
    assert "optimizer" not in init and "rng" not in init
    # Remove all full training arrays before inference; standalone measurements remain.
    (data_path / "trajectories.npz").unlink()
    (data_path / "manifest.json").unlink()
    result = reconstruct(config, tmp_path / "resumed/inference.pt", data_path / "test_problem.json",
                         data_path / "test_observations.json", tmp_path / "reconstruction")
    assert set(result) == {"unguided", "observation", "observation_pde", "fourier_inverse"}
    for mode, calls in (("unguided", 6), ("observation", 8), ("observation_pde", 8)):
        report = evaluate(tmp_path / "reconstruction" / mode)
        assert report["cost"]["model_evaluations"] == calls
        assert report["cost"]["gradient_evaluations"] == (0 if mode == "unguided" else 3)
        assert "relative_l2_unobserved_mean" not in report
        assert np.isfinite(report["pde_fd_rmse"])
        assert (tmp_path / "reconstruction" / mode / "reconstruction.png").stat().st_size > 1000


@pytest.mark.parametrize("dimension", [1, 3])
def test_training_changes_weights_and_guided_reconstruction_in_spatial_adapters(tmp_path, dimension):
    config = tiny_config(dimension)
    generate(config, tmp_path / "data")
    train(config, tmp_path / "data", tmp_path / "train", stop_after=1)
    step_one = torch.load(tmp_path / "train/inference.pt", weights_only=True)
    train(config, tmp_path / "data", tmp_path / "train", resume=tmp_path / "train/checkpoint.pt")
    step_six = torch.load(tmp_path / "train/inference.pt", weights_only=True)
    assert any(not torch.equal(step_one["model"][k], step_six["model"][k]) for k in step_one["model"])
    result = reconstruct(config, tmp_path / "train/inference.pt", tmp_path / "data/test_problem.json",
                         tmp_path / "data/test_observations.json", tmp_path / "reconstruction", tmp_path / "data/test_reference.npz")
    if dimension == 1:  # four sensors cover all four nodes at every physical time
        assert all(m["relative_l2_unobserved_mean"] is None for m in result.values())
    else:
        assert all(np.isfinite(m["relative_l2_unobserved_mean"]) for m in result.values())
    with np.load(tmp_path / "reconstruction/observation_pde/result.npz") as a:
        assert a["samples"].shape == (1, 2, 1, 3, *([4] * dimension))


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_fourier_inverse_recovers_identifiable_basis_coefficients(dimension):
    # Full spatial sensors make this a well-determined inverse test; no hidden initial supplied.
    grid = GridSpec((6,) * dimension, DomainSpec((1.,) * dimension), (0., .03, .06))
    basis, _ = fourier_heat_basis(grid, .07, 1)
    coefficients = torch.linspace(-.4, .8, len(basis), dtype=torch.float64)
    truth = torch.einsum("k,kct...->ct...", coefficients, basis).unsqueeze(0)
    obs = sensor_observations(truth, grid, 6**dimension, 5, 0.)
    result = inverse_baseline(grid, .07, obs, 1, 1e-10)
    assert torch.max(torch.abs(result.samples[0] - truth)) < 1e-8


def test_bad_resume_layout_and_data_integrity_fail_early(tmp_path):
    config = tiny_config()
    generate(config, tmp_path / "data")
    train(config, tmp_path / "data", tmp_path / "train", stop_after=2)
    changed = replace(config, training=replace(config.training, learning_rate=.1))
    with pytest.raises(ValueError, match="same resolved configuration"):
        train(changed, tmp_path / "data", tmp_path / "train", resume=tmp_path / "train/checkpoint.pt")
    changed_grid = replace(config, grid=replace(config.grid, times=(0., .02, .04)))
    with pytest.raises(ValueError, match="grid mismatch"):
        load_inference(tmp_path / "train/inference.pt", expected=changed_grid)
    # Simulate a saved accelerator device tag: tests metadata portability, not accelerator execution.
    ckpt = torch.load(tmp_path / "train/inference.pt", weights_only=True)
    ckpt["config"]["runtime"]["device"] = "cuda"
    torch.save(ckpt, tmp_path / "accelerator_tag.pt")
    portable = load_inference(tmp_path / "accelerator_tag.pt", runtime=RuntimeSpec(device="cpu"))
    assert next(portable.model.parameters()).device.type == "cpu"
    with open(tmp_path / "data/trajectories.npz", "ab") as f:
        f.write(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_dataset(tmp_path / "data", config)


def test_atomic_failure_preserves_last_complete_artifact(tmp_path):
    path = tmp_path / "artifact"
    path.write_bytes(b"complete")

    def fail(stream):
        stream.write(b"partial")
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        atomic_write(path, fail)
    assert path.read_bytes() == b"complete"
    assert list(tmp_path.iterdir()) == [path]


def test_candidate_population_does_not_couple_guidance(tmp_path):
    config = tiny_config()
    generate(config, tmp_path / "data")
    train(config, tmp_path / "data", tmp_path / "train", stop_after=1)
    bundle = load_inference(tmp_path / "train/inference.pt", runtime=config.runtime)
    problem = ProblemSpec.from_dict(read_json(tmp_path / "data/test_problem.json"))
    obs = ObservationSet.from_dict(read_json(tmp_path / "data/test_observations.json"), problem.grid)
    one = sample(bundle, problem, obs, replace(config.inference, particles=1), "observation_pde")
    two = sample(bundle, problem, obs, replace(config.inference, particles=2), "observation_pde")
    assert torch.allclose(one.samples[:, 0], two.samples[:, 0], atol=1e-6, rtol=1e-6)
    assert one.diagnostics["model_evaluations"] == two.diagnostics["model_evaluations"]
