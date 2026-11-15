"""Isaac-L2F CTBR body-rate feedback before the existing rotor actuator.

Based on Isaac-L2F e36743e control/ctbr.py::CTBRController, with measured
angular-acceleration damping for the existing first-order motor response.
"""
from __future__ import annotations

from importlib.resources import files
import json

import torch


class BodyRateController:
    def __init__(self, params, total_thrust_max, natural_frequency=40., damping_ratio=.9):
        self.params = params
        model = json.loads(files("isaac_pathcover.dynamics").joinpath("acmpc_quad.json").read_text())["dynamics"]
        self.natural_frequency, self.damping_ratio = natural_frequency, damping_ratio
        motor_tau = params.rotor_time_constants_rising[0, 0]
        self.rate_gain = params.mass.new_full((3,), natural_frequency**2) * motor_tau
        self.rate_damping = params.mass.new_full((3,), 2 * damping_ratio * natural_frequency) * motor_tau - 1.
        self.rate_limit = params.mass.new_tensor(model["body_rate_limit"])
        self.total_thrust_max = total_thrust_max
        self.previous_omega = params.mass.new_zeros((len(params.mass), 3))

    def reset(self, ids=None):
        self.previous_omega[slice(None) if ids is None else ids] = 0.

    def wrench(self, ctbr, omega_body, dt):
        """Convert thrust acceleration (m/s²) and body rates (rad/s) to N and N m."""
        thrust = (ctbr[:, 0] * self.params.mass).clamp_min(0).minimum(self.total_thrust_max)
        target = ctbr[:, 1:].maximum(-self.rate_limit).minimum(self.rate_limit)
        angular_momentum = torch.einsum("nij,nj->ni", self.params.J, omega_body)
        measured_acceleration = (omega_body - self.previous_omega) / dt
        self.previous_omega.copy_(omega_body)
        acceleration = self.rate_gain * (target - omega_body) - self.rate_damping * measured_acceleration
        moment = torch.einsum("nij,nj->ni", self.params.J, acceleration)
        moment += torch.cross(omega_body, angular_momentum, dim=-1)
        return torch.cat((thrust[:, None], moment), dim=-1)

    def metadata(self):
        return {"natural_frequency_rad_s": self.natural_frequency, "damping_ratio": self.damping_ratio,
                "rate_gain_per_s": self.rate_gain.cpu().tolist(),
                "rate_damping": self.rate_damping.cpu().tolist(),
                "rate_limit_rad_s": self.rate_limit.cpu().tolist()}
