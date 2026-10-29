"""Validated runtime/configuration and atomic persistence."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import sys
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from pdeflow.core import Capabilities, DomainSpec, GridSpec
from pdeflow.equations import heat_problem


@dataclass(frozen=True)
class RuntimeSpec:
    device: str = "cpu"
    dtype: str = "float32"
    precision: str = "full"
    threads: int = 2

    def resolve(self) -> tuple[torch.device, torch.dtype]:
        if self.device not in ("auto", "cpu", "cuda", "mps") or self.dtype not in ("float32", "float64"):
            raise ValueError("Unsupported device/dtype")
        if self.precision != "full" or self.threads < 1:
            raise ValueError("Only full precision and positive thread counts are implemented")
        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable; choose cpu explicitly")
        if device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS requested but unavailable; choose cpu explicitly")
        if device == "mps" and self.dtype == "float64":
            raise ValueError("MPS float64 is unsupported")
        torch.set_num_threads(self.threads)
        return torch.device(device), getattr(torch, self.dtype)


@dataclass(frozen=True)
class DataConfig:
    count: int = 30
    seed: int = 11
    diffusivity_range: tuple[float, float] = (.03, .12)
    bandwidth: float = 1.2
    storage: str = "npz"
    split_counts: tuple[int, int, int] | None = None
    shard_size: int = 32
    storage_dtype: str = "float64"
    bandwidth_range: tuple[float, float] | None = None
    amplitude_range: tuple[float, float] = (1., 1.)
    offset_range: tuple[float, float] = (0., 0.)


@dataclass(frozen=True)
class ModelConfig:
    width: int = 12
    blocks: int = 2
    name: str = "residual_cnn"
    levels: int = 1
    conditioning_width: int | None = None
    coordinate_features: bool = False
    activation_checkpointing: bool = False
    temporal_handling: str = "fixed_channels"


@dataclass(frozen=True)
class TrainingConfig:
    steps: int = 30
    batch_size: int = 4
    learning_rate: float = .002
    seed: int = 23
    checkpoint_every: int = 10
    optimizer: str = "adam"


@dataclass(frozen=True)
class InferenceConfig:
    steps: int = 12
    stepper: str = "heun"
    schedule: str = "uniform"
    particles: int = 2
    sensors: int = 10
    noise_std: float = .03
    noise_floor: float = .1
    seed: int = 37
    observation_strength: float = .3
    physics_strength: float = .001
    correction_strength: float = 1.
    gradient_max_norm: float = 10.
    basis_degree: int = 1
    ridge: float = .001
    controller: dict | None = None
    fk: dict | None = None
    snapshots: dict | None = None

    def validate(self):
        if self.snapshots is not None:
            from pdeflow.sampling.snapshots import SnapshotSpec
            SnapshotSpec(**self.snapshots).validate()
        if self.fk is not None:
            from pdeflow.fk import FKConfig
            FKConfig(**self.fk).validate()
        if self.controller is not None:
            from pdeflow.process import ProcessConfig
            ProcessConfig(**self.controller).validate()
        for name in ("steps", "particles", "sensors", "basis_degree"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("Sampling seed must be an integer in [0,2^63)")
        if self.stepper not in ("euler", "heun") or self.schedule not in ("uniform", "quadratic"):
            raise ValueError("Incompatible ODE stepper or schedule")
        if min(self.noise_std, self.observation_strength, self.physics_strength, self.correction_strength) < 0 or min(self.noise_floor, self.gradient_max_norm, self.ridge) <= 0:
            raise ValueError("Invalid noise, guidance or regularization scales")
        json.dumps(asdict(self), allow_nan=False)
        return self


@dataclass(frozen=True)
class ProviderConfig:
    mode: str = "online"
    distribution_version: str = "physical_v1"
    calibration_count: int = 8
    validation_count: int = 2
    test_count: int = 2
    offline_count: int = 32
    capacity: int = 8
    reuse_limit: int = 2
    max_attempts: int = 2
    workers: int = 0
    solver_threads: int = 1
    solver_rtol: float = 1e-10
    solver_max_iterations: int = 4000

    def validate(self):
        if self.mode not in ("offline","online","hybrid") or self.distribution_version != "physical_v1":
            raise ValueError("Unsupported provider mode/distribution version")
        if torch.distributed.is_available() and torch.distributed.is_initialized():raise ValueError("Distributed reference ID allocation is not implemented; use one process/rank")
        if self.workers != 0 or self.solver_threads != 1:
            raise ValueError("Deterministic reference provider currently requires workers=0, solver_threads=1; no distributed allocation implemented")
        if any(type(v) is not int or v<1 for v in [self.calibration_count,self.validation_count,self.test_count,self.offline_count,self.capacity,self.reuse_limit,self.max_attempts,self.solver_max_iterations]):
            raise ValueError("Provider counts/capacity/reuse/attempt limits must be positive integers")
        if not 0<self.solver_rtol<1:raise ValueError("Invalid reference tolerance")
        return self


@dataclass(frozen=True)
class ChemistryConfig:
    diffusivity_range: tuple[float,float] = (.01,.04)
    forward_rate_range: tuple[float,float] = (.3,1.)
    reverse_rate_range: tuple[float,float] = (.1,.5)
    initial_floor: float = .05
    initial_amplitude: float = .5
    reference_step: float = .0025

    def validate(self):
        for bounds in (self.diffusivity_range,self.forward_rate_range,self.reverse_rate_range):
            if len(bounds)!=2 or bounds[0]<0 or bounds[1]<bounds[0]:raise ValueError("Invalid chemistry parameter ranges")
        if self.diffusivity_range[0]<=0 or self.initial_floor<0 or self.initial_amplitude<=0 or self.reference_step<=0:
            raise ValueError("Positive diffusion/step/amplitude and nonnegative initial floor required")
        return self


@dataclass(frozen=True)
class Config:
    grid: GridSpec
    runtime: RuntimeSpec = field(default_factory=RuntimeSpec)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    formulation: str = "independent_gaussian_linear_cfm"
    specialist: str = "heat"
    provider: ProviderConfig | None = None
    chemistry: ChemistryConfig | None = None

    def validate(self, check_device=True):
        self.inference.validate()
        if self.provider:self.provider.validate()
        if self.chemistry:self.chemistry.validate()
        from pdeflow.specialists import get_specialist
        get_specialist(self.specialist).validate_config(self)
        if self.data.count < 6 or len(self.data.diffusivity_range) != 2 or not 0 < self.data.diffusivity_range[0] <= self.data.diffusivity_range[1] or self.data.bandwidth <= 0:
            raise ValueError("Need >=6 trajectories, positive diffusivity range and bandwidth")
        d = self.data
        if d.storage not in ("npz", "sharded_npz") or d.storage_dtype not in ("float32", "float64") or d.shard_size < 1:
            raise ValueError("Invalid data storage/dtype/shard size")
        if d.split_counts is not None and (len(d.split_counts) != 3 or any(type(n) is not int or n < 1 for n in d.split_counts) or sum(d.split_counts) != d.count):
            raise ValueError("split_counts must be positive train/validation/test counts summing to count")
        for bounds in (d.amplitude_range, d.offset_range, d.bandwidth_range or (d.bandwidth, d.bandwidth)):
            if len(bounds) != 2 or bounds[0] > bounds[1]: raise ValueError("Invalid sampling interval")
        if d.amplitude_range[0] <= 0 or (d.bandwidth_range and d.bandwidth_range[0] <= 0):
            raise ValueError("Amplitude and bandwidth must be positive")
        if d.storage == "npz" and (d.storage_dtype != "float64" or d.split_counts is not None or d.bandwidth_range is not None or tuple(d.amplitude_range) != (1., 1.) or tuple(d.offset_range) != (0., 0.)):
            raise ValueError("Legacy NPZ smoke storage is float64 with its original distribution/splits; use sharded_npz for configurable profiles")
        if self.model.name not in ("residual_cnn","multiscale_unet") or self.model.width < 2 or self.model.blocks < 1 or not 1<=self.model.levels<=5 or (self.model.conditioning_width is not None and self.model.conditioning_width<2) or self.model.temporal_handling!="fixed_channels":
            raise ValueError("Unsupported backbone or invalid CNN size")
        if self.model.name=='residual_cnn' and (self.model.levels!=1 or self.model.conditioning_width is not None or self.model.coordinate_features or self.model.activation_checkpointing):raise ValueError('Extended capacity/coordinate options require multiscale_unet; legacy CNN weights are unchanged')
        if self.model.name=='multiscale_unet' and any(n%(2**(self.model.levels-1)) for n in self.grid.shape):raise ValueError('Multiscale grid axes must be divisible by the declared pooling factor')
        t, i = self.training, self.inference
        if min(t.steps, t.batch_size, t.checkpoint_every, i.steps, i.particles, i.sensors) < 1 or t.learning_rate <= 0 or t.optimizer != "adam":
            raise ValueError("Positive budgets and Adam optimizer are required")
        if not 1 <= i.basis_degree < min(self.grid.shape) / 2:
            raise ValueError("Fourier baseline degree must be >=1 and below Nyquist")
        if i.sensors > int(np.prod(self.grid.shape)) or i.noise_std < 0 or min(i.noise_floor, i.gradient_max_norm, i.ridge) <= 0:
            raise ValueError("Invalid sensors, observation scales, gradient limit or ridge")
        if min(i.observation_strength, i.physics_strength, i.correction_strength) < 0:
            raise ValueError("Guidance strengths must be nonnegative")
        # JSON permits NaN in Python by default; reject it recursively here.
        json.dumps(asdict(self), allow_nan=False)
        if check_device:
            self.runtime.resolve()
        else:
            # Metadata may describe an accelerator absent from the loading host.
            if self.runtime.device not in ("auto", "cpu", "cuda", "mps"):
                raise ValueError("Unsupported saved runtime device")
            RuntimeSpec("cpu", self.runtime.dtype, self.runtime.precision, self.runtime.threads).resolve()
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, v: dict, check_device=True):
        unknown = set(v) - {"grid", "runtime", "data", "model", "training", "inference", "formulation", "specialist", "provider", "chemistry"}
        if unknown:
            raise ValueError(f"Unknown configuration keys: {unknown}")
        return cls(GridSpec.from_dict(v["grid"]), RuntimeSpec(**v.get("runtime", {})), DataConfig(**v.get("data", {})),
                   ModelConfig(**v.get("model", {})), TrainingConfig(**v.get("training", {})), InferenceConfig(**v.get("inference", {})),
                   v.get("formulation", "independent_gaussian_linear_cfm"), v.get("specialist", "heat"), ProviderConfig(**v["provider"]) if v.get("provider") else None, ChemistryConfig(**v["chemistry"]) if v.get("chemistry") else None).validate(check_device=check_device)

    @classmethod
    def load(cls, path, device=None, check_device=True):
        v = read_json(path)
        if device is not None:
            v.setdefault("runtime", {})["device"] = device
        return cls.from_dict(v, check_device=check_device)


def atomic_write(path: Path | str, writer: Callable):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_json(path, value):
    content = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode()
    atomic_write(path, lambda f: f.write(content))


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_digest(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def environment(runtime: RuntimeSpec) -> dict:
    device, dtype = runtime.resolve()
    return {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(),
            "torch": str(torch.__version__), "numpy": np.__version__, "device": str(device), "dtype": str(dtype),
            "threads": runtime.threads, "cuda_available": torch.cuda.is_available(),
            "mps_available": torch.backends.mps.is_available()}


def process_peak_memory_bytes() -> int:
    """Process lifetime resident high-water mark, not an isolated phase peak."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def code_identity() -> dict:
    package = Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        h.update(str(path.relative_to(package)).encode())
        h.update(path.read_bytes())
    try:
        result = subprocess.run(["git", "-C", str(package.parents[1]), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=False)
        revision = result.stdout.strip() if result.returncode == 0 else None
    except FileNotFoundError:
        revision = None
    return {"version": "pdeflow-0.1.0", "source_sha256": h.hexdigest(), "git_revision": revision}
