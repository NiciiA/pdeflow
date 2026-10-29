"""Known family/scalar encoding, independent of observations and equation trees."""
import torch
from torch import nn

from pdeflow.core import Conditioning


class ScalarConditionEncoder(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, width), nn.SiLU(), nn.Linear(width, width))

    def forward(self, condition: Conditioning):
        if condition.family != "heat" or set(condition.known_parameters) != {"diffusivity"}:
            raise ValueError("Encoder supports heat family plus known diffusivity only")
        d = condition.known_parameters["diffusivity"]
        if d.ndim != 1 or not torch.isfinite(d).all() or torch.any(d <= 0):
            raise ValueError("Encoder requires positive [problem] diffusivity")
        # A family feature (heat=1) and nondimensional log diffusivity.
        return self.net(torch.stack((torch.ones_like(d), d.log()), dim=1))


class ForcingFieldEncoder(nn.Module):
    """Known forcing map features, with no unknown permeability/pressure inputs."""
    def __init__(self,width,dimension):
        super().__init__();conv={2:nn.Conv2d,3:nn.Conv3d}.get(dimension)
        if conv is None:raise ValueError('Forcing encoder requires 2D/3D')
        self.net=nn.Sequential(conv(1,width,1),nn.SiLU(),conv(width,width,1))
    def forward(self,condition):
        if condition.known_parameters or set(condition.known_fields)!={'f'}:raise ValueError('Forcing condition requires exactly known field f')
        f=condition.known_fields['f']
        if not torch.isfinite(f).all():raise ValueError('Nonfinite forcing condition')
        return self.net(f[:,None])


def make_condition_encoder(kind,width,dimension):
    if kind=='heat_scalars':return ScalarConditionEncoder(width)
    if kind=='forcing_field':return ForcingFieldEncoder(width,dimension)
    if kind=='chemistry_scalars':return ChemistryConditionEncoder(width)
    raise ValueError(f'Unsupported condition encoder {kind}')


class ChemistryConditionEncoder(nn.Module):
    def __init__(self,width):
        super().__init__();self.net=nn.Sequential(nn.Linear(5,width),nn.SiLU(),nn.Linear(width,width))
    def forward(self,condition):
        from pdeflow.chemistry import PARAMETERS
        if condition.family!='chemistry' or condition.known_fields or set(condition.known_parameters)!=set(PARAMETERS):raise ValueError('Chemistry encoder requires exactly known D_A/D_B/D_C/k_f/k_r')
        p=condition.known_parameters
        if any(not torch.isfinite(x).all() for x in p.values()) or any(torch.any(p[n]<=0) for n in PARAMETERS[:3]) or any(torch.any(p[n]<0) for n in PARAMETERS[3:]):raise ValueError('Invalid chemistry conditioning parameters')
        return self.net(torch.stack([p[n].log() if n in PARAMETERS[:3] else p[n].log1p() for n in PARAMETERS],dim=1))
