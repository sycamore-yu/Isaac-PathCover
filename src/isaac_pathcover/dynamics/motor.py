"""L2F rotor force and closed-form motor response, extracted without equation changes."""
from __future__ import annotations
import torch
from torch import Tensor
from .params import Params


def rotor_thrust(params: Params, u: Tensor) -> Tensor:
    """Per-rotor thrust ``[N,4]`` (N): ``c0 + c1 u + c2 u^2``."""
    c = params.rotor_thrust_coefficients
    return c[..., 0] + c[..., 1] * u + c[..., 2] * u * u


def wrench_from_thrust(params: Params, thrust: Tensor) -> tuple[Tensor, Tensor]:
    """Body-frame force ``[N,3]`` (N) and torque ``[N,3]`` (N m) about the CoM from per-rotor thrust ``[N,4]`` (N)."""
    f_i = thrust[..., None] * params.rotor_thrust_directions  # [N,4,3]
    drag = params.rotor_torque_directions * (params.rotor_torque_constants * thrust)[..., None]
    tau_i = drag + torch.cross(params.rotor_positions, f_i, dim=-1)
    return f_i.sum(1), tau_i.sum(1)


def _tau(params: Params, u: Tensor, u_cmd: Tensor) -> Tensor:
    return torch.where(u_cmd >= u, params.rotor_time_constants_rising, params.rotor_time_constants_falling)


def rotor_thrust_mean(params: Params, u: Tensor, u_cmd: Tensor, dt: Tensor | float) -> Tensor:
    """Time average of the thrust polynomial over a hold of ``dt`` (s) ``[N,4]`` (N).

    The motor state follows the exact first-order lag from ``u`` towards ``u_cmd`` with the
    rising/falling time constant, ``u(t) = u_cmd + (u - u_cmd) exp(-t / tau)``; the average of
    ``c0 + c1 u + c2 u^2`` over ``[0, dt]`` is closed-form. A simulator that holds the wrench
    constant over ``dt`` then applies the exact impulse of the lagging thrust.
    """
    dt_t = torch.as_tensor(dt, dtype=u.dtype, device=u.device)
    if dt_t.dim() == 1:
        dt_t = dt_t[:, None]
    a = dt_t / _tau(params, u, u_cmd)
    e1 = -torch.expm1(-a) / a  # mean of exp(-t/tau)
    e2 = -torch.expm1(-2 * a) / (2 * a)  # mean of exp(-2t/tau)
    d = u - u_cmd
    mean_u = u_cmd + d * e1
    mean_u2 = u_cmd * u_cmd + 2 * u_cmd * d * e1 + d * d * e2
    c = params.rotor_thrust_coefficients
    return c[..., 0] + c[..., 1] * mean_u + c[..., 2] * mean_u2


def motor_step_closed_form(params: Params, u: Tensor, u_cmd: Tensor, dt: Tensor | float) -> Tensor:
    """Exact first-order lag over ``dt`` with rising/falling rule, clamped to the action limit."""
    dt_t = torch.as_tensor(dt, dtype=u.dtype, device=u.device)
    if dt_t.dim() == 1:
        dt_t = dt_t[:, None]
    out = u + (u_cmd - u) * (1 - torch.exp(-dt_t / _tau(params, u, u_cmd)))
    return torch.clamp(out, params.action_min[:, None], params.action_max[:, None])
