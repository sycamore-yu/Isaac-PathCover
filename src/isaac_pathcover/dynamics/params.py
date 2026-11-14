"""The L2F airframe fields used by rotor actuation; all tensors are batched."""
from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
import json

import torch


@dataclass
class Params:
    rotor_positions: torch.Tensor
    rotor_thrust_directions: torch.Tensor
    rotor_torque_directions: torch.Tensor
    rotor_thrust_coefficients: torch.Tensor
    rotor_torque_constants: torch.Tensor
    rotor_time_constants_rising: torch.Tensor
    rotor_time_constants_falling: torch.Tensor
    mass: torch.Tensor
    gravity: torch.Tensor
    J: torch.Tensor
    J_inv: torch.Tensor
    action_min: torch.Tensor
    action_max: torch.Tensor

    @property
    def device(self):
        return self.mass.device

    @property
    def dtype(self):
        return self.mass.dtype


def load_params(device="cpu", dtype=torch.float32, num_envs=1):
    data = json.loads(files(__package__).joinpath("acmpc_quad.json").read_text())["dynamics"]
    values = {}
    for name in Params.__dataclass_fields__:
        value = data["action_limit"][name.removeprefix("action_")] if name.startswith("action_") else data[name]
        tensor = torch.tensor([value], device=device, dtype=dtype)
        values[name] = tensor.repeat(num_envs, *([1] * (tensor.ndim - 1)))
    return Params(**values)
