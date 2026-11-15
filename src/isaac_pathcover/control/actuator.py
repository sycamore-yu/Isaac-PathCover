"""Physical wrench to L2F rotor allocation and 400 Hz motor response."""
from __future__ import annotations

import torch

from isaac_pathcover.dynamics import motor_step_closed_form, rotor_thrust, rotor_thrust_mean, wrench_from_thrust


class RotorActuator:
    def __init__(self, params):
        self.params = params
        direction = params.rotor_thrust_directions
        moments = torch.cross(params.rotor_positions, direction, dim=-1)
        moments += params.rotor_torque_constants[..., None] * params.rotor_torque_directions
        self.allocation = torch.cat((direction[..., 2:3], moments), dim=-1).transpose(1, 2)
        self.allocation_inverse = torch.linalg.inv(self.allocation)
        self.lower = rotor_thrust(params, params.action_min[:, None].expand(-1, 4))
        self.upper = rotor_thrust(params, params.action_max[:, None].expand(-1, 4))
        hover = (-params.mass * params.gravity[:, 2])[:, None].expand(-1, 4) / 4
        self.hover_state = self._inverse_thrust(hover)
        self.motor_state = self.hover_state.clone()
        self.motor_command = self.hover_state.clone()
        self.requested_wrench = torch.zeros_like(self.motor_state)
        self.allocated_wrench = torch.zeros_like(self.motor_state)
        self.allocated_thrust = hover.clone()

    def _inverse_thrust(self, thrust):
        c0, c1, c2 = self.params.rotor_thrust_coefficients.unbind(-1)
        command = (torch.sqrt(c1.square() + 4 * c2 * (thrust - c0)) - c1) / (2 * c2)
        return command.maximum(self.params.action_min[:, None]).minimum(self.params.action_max[:, None])

    def command(self, wrench):
        self.requested_wrench.copy_(wrench)
        differential = torch.einsum("nij,nj->ni", self.allocation_inverse[:, :, 1:], wrench[:, 1:])
        span = differential.amax(dim=-1, keepdim=True) - differential.amin(dim=-1, keepdim=True)
        available = self.upper[:, :1] - self.lower[:, :1]
        differential *= available / span.maximum(available)
        collective_min = (self.lower - differential).amax(dim=-1, keepdim=True)
        collective_max = (self.upper - differential).amin(dim=-1, keepdim=True)
        collective = (wrench[:, :1] / 4).maximum(collective_min).minimum(collective_max)
        thrust = collective + differential
        # The common collective shift and uniform torque scale preserve torque direction.
        self.allocated_thrust.copy_(thrust.maximum(self.lower).minimum(self.upper))
        self.allocated_wrench.copy_(torch.einsum("nij,nj->ni", self.allocation, self.allocated_thrust))
        self.motor_command.copy_(self._inverse_thrust(self.allocated_thrust))

    def step(self, dt):
        thrust = rotor_thrust_mean(self.params, self.motor_state, self.motor_command, dt)
        self.motor_state.copy_(motor_step_closed_form(self.params, self.motor_state, self.motor_command, dt))
        return wrench_from_thrust(self.params, thrust)

    def reset(self, ids=None):
        ids = slice(None) if ids is None else ids
        self.motor_state[ids] = self.hover_state[ids]
        self.motor_command[ids] = self.hover_state[ids]
