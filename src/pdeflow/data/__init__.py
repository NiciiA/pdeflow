"""Complete-trajectory splits and simple safe dataset storage."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from pdeflow.core import Conditioning, GridSpec
from pdeflow.equations import heat_problem
from pdeflow.numerics import heat_reference, smooth_initial
from pdeflow.observations import sensor_observations
from pdeflow.runtime import Config, atomic_write, digest, file_digest, read_json, write_json


@dataclass
class TrajectoryData:
    values: torch.Tensor
    diffusivity: torch.Tensor
    metadata: dict

    def indices(self, split):
        return self.metadata["splits"][split]

    def conditioning(self, indices) -> Conditioning:
        # The heat data adapter owns parameter-array meanings, not the trainer.
        return Conditioning(self.metadata["family"], {"diffusivity": self.diffusivity[indices]})


def _generate_heat(config: Config, output: str | Path):
    config.validate(check_device=False)
    if config.data.storage == "sharded_npz":
        from .shards import generate_shards
        return generate_shards(config, output)
    output = Path(output)
    if (output / "manifest.json").exists() or (output / "trajectories.npz").exists():
        raise FileExistsError(f"Dataset already exists at {output}; select a new output directory")
    start = time.perf_counter()
    grid, dc = config.grid, config.data
    seeds = np.random.SeedSequence(dc.seed).spawn(dc.count + 1)
    values, diffusivities, ids, record_seeds = [], [], [], []
    for j in range(dc.count):
        record_seed = int(seeds[j].generate_state(1)[0])
        rng = np.random.default_rng(record_seed)
        d = float(rng.uniform(*dc.diffusivity_range))
        initial = smooth_initial(rng, grid, dc.bandwidth)
        values.append(heat_reference(initial, d, grid)[None])
        diffusivities.append(d)
        ids.append(digest({"generator": "fourier_heat_v1", "grid": asdict(grid), "data": asdict(dc), "index": j})[:20])
        record_seeds.append(record_seed)
    order = np.random.default_rng(seeds[-1]).permutation(dc.count).tolist()
    train_n, val_n = int(.6 * dc.count), max(1, int(.2 * dc.count))
    splits = {"train": order[:train_n], "validation": order[train_n:train_n + val_n], "test": order[train_n + val_n:]}
    arrays = {"values": np.stack(values), "diffusivity": np.array(diffusivities)}
    atomic_write(output / "trajectories.npz", lambda f: np.savez_compressed(f, **arrays))
    metadata = {"schema_version": 1, "family": "heat", "generator": "fourier_heat_v1", "grid": asdict(grid), "data_config": asdict(dc),
                "ids": ids, "record_seeds": record_seeds, "splits": splits, "units": "nondimensional",
                "fields": [{"name": "u", "role": "unknown", "meaning": "signed scalar diffusion field"}],
                "sha256": file_digest(output / "trajectories.npz"), "generation_seconds": time.perf_counter() - start}
    # Standalone public reconstruction inputs contain only known metadata and observations.
    k = splits["test"][0]
    truth = torch.from_numpy(arrays["values"][k:k + 1])
    obs = sensor_observations(truth, grid, config.inference.sensors, config.inference.seed, config.inference.noise_std, config.inference.noise_floor)
    write_json(output / "test_problem.json", heat_problem(grid, diffusivities[k]).to_dict())
    write_json(output / "test_observations.json", obs.to_dict())
    atomic_write(output / "test_reference.npz", lambda f: np.savez_compressed(f, values=truth.numpy()))
    # Publish the completion manifest after all dataset and public-input artifacts.
    write_json(output / "manifest.json", metadata)
    return metadata


def _load_heat(path: str | Path, config: Config) -> TrajectoryData:
    path = Path(path)
    m = read_json(path / "manifest.json")
    if m["schema_version"] == 2:
        from .shards import load_shards
        return load_shards(path, config)
    from pdeflow.runtime import DataConfig
    if m["schema_version"] != 1 or m.get("family") != "heat" or GridSpec.from_dict(m["grid"]) != config.grid or digest(asdict(DataConfig(**m["data_config"]))) != digest(asdict(config.data)):
        raise ValueError("Dataset schema/grid/generator configuration mismatch")
    if file_digest(path / "trajectories.npz") != m["sha256"]:
        raise ValueError("Dataset integrity hash mismatch")
    flat = [i for split in m["splits"].values() for i in split]
    if sorted(flat) != list(range(len(m["ids"]))) or len(set(m["ids"])) != len(m["ids"]) or any(not m["splits"][s] for s in ("train", "validation", "test")):
        raise ValueError("Invalid or overlapping trajectory splits")
    with np.load(path / "trajectories.npz", allow_pickle=False) as a:
        values, d = torch.from_numpy(a["values"].copy()), torch.from_numpy(a["diffusivity"].copy())
    if tuple(values.shape) != (config.data.count, 1, len(config.grid.times), *config.grid.shape) or tuple(d.shape) != (config.data.count,) or not torch.isfinite(values).all() or not torch.isfinite(d).all() or torch.any(d <= 0):
        raise ValueError("Invalid dataset values/layout/parameters")
    return TrajectoryData(values, d, m)


def generate(config: Config, output):
    from pdeflow.specialists import get_specialist
    config.validate(check_device=False)
    return get_specialist(config.specialist).generate(config, output)


def load_dataset(path, config: Config):
    from pdeflow.specialists import get_specialist
    return get_specialist(config.specialist).load_data(path, config)
