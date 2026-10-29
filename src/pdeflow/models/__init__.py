"""Fixed physical-time packing and residual spatial CNN adapters."""
from dataclasses import dataclass

import torch
from torch import nn

from pdeflow.conditioning import make_condition_encoder
from pdeflow.core import Conditioning, GridSpec
from pdeflow.runtime import ModelConfig


@dataclass(frozen=True)
class TimeChannelLayout:
    channels: int
    grid: GridSpec

    def pack(self, field):
        if tuple(field.shape[1:]) != (self.channels, len(self.grid.times), *self.grid.shape):
            raise ValueError("Fixed physical-time/channel/grid layout mismatch")
        return field.reshape(field.shape[0], self.channels * len(self.grid.times), *self.grid.shape)

    def unpack(self, state):
        if tuple(state.shape[1:]) != (self.channels * len(self.grid.times), *self.grid.shape):
            raise ValueError("Packed model layout mismatch")
        return state.reshape(state.shape[0], self.channels, len(self.grid.times), *self.grid.shape)


def convolution(dimension: int):
    try:
        return {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[dimension]
    except KeyError as e:
        raise ValueError("Only 1D/2D/3D spatial convolutions supported") from e


class ResidualBlock(nn.Module):
    def __init__(self, conv, width):
        super().__init__()
        self.net = nn.Sequential(nn.SiLU(), conv(width, width, 3, padding=1, padding_mode="circular"),
                                 nn.SiLU(), conv(width, width, 3, padding=1, padding_mode="circular"))

    def forward(self, h):
        return h + self.net(h) / 2**.5


class ResidualCNN(nn.Module):
    def __init__(self, layout: TimeChannelLayout, config: ModelConfig, condition_kind="heat_scalars"):
        super().__init__()
        self.layout = layout
        conv = convolution(layout.grid.dimension)
        channels = layout.channels * len(layout.grid.times)
        self.condition_encoder = make_condition_encoder(condition_kind, config.width, layout.grid.dimension)
        self.time_encoder = nn.Sequential(nn.Linear(3, config.width), nn.SiLU(), nn.Linear(config.width, config.width))
        self.input = conv(channels, config.width, 3, padding=1, padding_mode="circular")
        self.blocks = nn.Sequential(*(ResidualBlock(conv, config.width) for _ in range(config.blocks)))
        self.output = conv(config.width, channels, 3, padding=1, padding_mode="circular")

    def forward(self, state: torch.Tensor, generation_time: torch.Tensor, condition: Conditioning):
        self.layout.unpack(state)  # cheap strict compatibility check
        if generation_time.shape != (state.shape[0],):
            raise ValueError("Generation time requires one value per model batch item")
        s = generation_time
        embedding = self.time_encoder(torch.stack((s, torch.sin(torch.pi * s), torch.cos(torch.pi * s)), dim=1))
        features = self.condition_encoder(condition)
        embedding = embedding + features if features.ndim == 2 else embedding.reshape(state.shape[0], -1, *([1] * self.layout.grid.dimension)) + features
        if embedding.shape[0] != state.shape[0]:
            raise ValueError("Condition batch mismatch")
        h = self.input(state) + (embedding.reshape(state.shape[0], -1, *([1] * self.layout.grid.dimension)) if embedding.ndim == 2 else embedding)
        return self.output(torch.nn.functional.silu(self.blocks(h)))


def make_model(config: ModelConfig, layout: TimeChannelLayout, condition_kind="heat_scalars"):
    if config.name == "multiscale_unet":
        from .multiscale import MultiscaleUNet
        return MultiscaleUNet(layout,config,condition_kind)
    if config.name != "residual_cnn":
        raise ValueError(f"Unsupported backbone {config.name}")
    return ResidualCNN(layout, config, condition_kind)


def architecture_metadata(model,config,layout):
    from dataclasses import asdict
    return {'configuration':asdict(config),'parameters':sum(p.numel() for p in model.parameters()),'spatial_dimension':layout.grid.dimension,'packed_channels':layout.channels*len(layout.grid.times),'deepest_path_3x3_layers':2+2*config.blocks*(2*config.levels-1),'temporal_handling':'fixed physical-time frames packed as channels; generation s separately embedded','resolution_policy':'exact checkpoint grid/domain/convention/saved times; no resolution-generalization claim','receptive_field':'finite periodic convolutions with multiscale pooling/skips' if config.name=='multiscale_unet' else 'finite periodic convolutions; no pooling'}
