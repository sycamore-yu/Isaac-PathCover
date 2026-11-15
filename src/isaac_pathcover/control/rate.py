"""Isaac-L2F CTBR body-rate feedback before the existing rotor actuator.

Rate law copied from Isaac-L2F e36743e control/ctbr.py::CTBRController.
The caller samples actual body rates at every physics step.
"""
from __future__ import annotations

from importlib.resources import files
import json

import torch


class BodyRateController:
    def __init__(self, params, total_thrust_max):
        self.params = params
        model = json.loads(files("isaac_pathcover.dynamics").joinpath("acmpc_quad.json").read_text())["dynamics"]
        self.rate_gain = params.mass.new_tensor(model["body_rate_gain"])
        self.rate_limit = params.mass.new_tensor(model["body_rate_limit"])
        self.total_thrust_max = total_thrust_max

    def wrench(self, ctbr, omega_body):
        """Convert thrust acceleration (m/s²) and body rates (rad/s) to N and N m."""
        thrust = (ctbr[:, 0] * self.params.mass).clamp_min(0).minimum(self.total_thrust_max)
        target = ctbr[:, 1:].maximum(-self.rate_limit).minimum(self.rate_limit)
        angular_momentum = torch.einsum("nij,nj->ni", self.params.J, omega_body)
        acceleration = self.rate_gain * (target - omega_body)
        moment = torch.einsum("nij,nj->ni", self.params.J, acceleration)
        moment += torch.cross(omega_body, angular_momentum, dim=-1)
        return torch.cat((thrust[:, None], moment), dim=-1)
