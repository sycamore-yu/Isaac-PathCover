"""CA-AC-MPC fast-backend prediction model, in physical units.

The reference is differentialMPCPerformance/cpp/drone_kernels.cu at
14aa349bb1f509c529bf3662c7feb72db68f1cb7.  That backend normalizes its
quaternion, unlike the unused Python DroneDx.forward path.  Its published
Euler Jacobian is preserved separately from the derivative of this rollout.
This module does not integrate the Newton execution model.
"""

from __future__ import annotations

import torch
from torch import Tensor

MASS = 0.752
DT = 0.02
GRAVITY = 9.8066
LOWER = (0.0, -10.0, -10.0, -4.0)
UPPER = (34.0, 10.0, 10.0, 4.0)


def ca_step(x: Tensor, u: Tensor) -> Tensor:
    """Euler p/v, body-rate quaternion exponential, and CA normalization.

    A small-angle series removes the norm's undefined second derivative at
    zero.  Below 1e-8 rad/s its difference from the CUDA epsilon convention
    is below float32 resolution.  The validation compares the actual kernel.
    """
    p, q, v = x[..., :3], x[..., 3:7], x[..., 7:]
    w = u[..., 1:]
    r2 = w.square().sum(-1)
    r = r2.clamp_min(1e-24).sqrt() + 1e-8
    a = DT * 0.5
    small = r2 < 1e-16
    scalar = torch.where(small, 1 - a*a*r2/2 + a**4*r2.square()/24, (a*r).cos())
    scale = torch.where(small, a - a**3*r2/6 + a**5*r2.square()/120, (a*r).sin()/r)
    ew, ex, ey, ez = scalar, *(w * scale[..., None]).unbind(-1)
    qw, qx, qy, qz = q.unbind(-1)
    q_next = torch.stack((qw*ew-qx*ex-qy*ey-qz*ez,
                          qw*ex+qx*ew+qy*ez-qz*ey,
                          qw*ey-qx*ez+qy*ew+qz*ex,
                          qw*ez+qx*ey-qy*ex+qz*ew), -1)
    q_next = q_next / (q_next.norm(dim=-1, keepdim=True) + 1e-8)
    direction = torch.stack((2*(qx*qz+qw*qy), 2*(qy*qz-qw*qx),
                             qw.square()-qx.square()-qy.square()+qz.square()), -1)
    acceleration = direction * (u[..., :1]/MASS) + x.new_tensor((0.0, 0.0, -GRAVITY))
    return torch.cat((p+DT*v, q_next, v+DT*acceleration), -1)


def ca_jacobian(x: Tensor, u: Tensor) -> tuple[Tensor, Tensor]:
    """The approximate analytic Jacobian used by the original CUDA solver."""
    shape = x.shape[:-1]
    A = torch.eye(10, device=x.device, dtype=x.dtype).expand(*shape, 10, 10).clone()
    B = x.new_zeros((*shape, 10, 4))
    eye = torch.eye(3, device=x.device, dtype=x.dtype)
    A[..., :3, 7:] = DT*eye
    qw, qx, qy, qz = x[..., 3:7].unbind(-1)
    wx, wy, wz = u[..., 1:].unbind(-1)
    zero = torch.zeros_like(wx)
    G = torch.stack((zero, -wx, -wy, -wz,
                     wx, zero, wz, -wy,
                     wy, -wz, zero, wx,
                     wz, wy, -wx, zero), -1).reshape(*shape, 4, 4)
    A[..., 3:7, 3:7] += DT*0.5*G
    B[..., 3:7, 1:] = DT*0.5*torch.stack((-qx, -qy, -qz,
                                         qw, -qz, qy,
                                         qz, qw, -qx,
                                         -qy, qx, qw), -1).reshape(*shape, 4, 3)
    V = torch.stack((qy, qz, qw, qx,
                     -qx, -qw, qz, qy,
                     qw, -qx, -qy, qz), -1).reshape(*shape, 3, 4)
    A[..., 7:, 3:7] = (2*DT/MASS)*u[..., 0, None, None]*V
    B[..., 7:, 0] = (DT/MASS)*torch.stack((2*(qx*qz+qw*qy), 2*(qy*qz-qw*qx),
                                         qw.square()-qx.square()-qy.square()+qz.square()), -1)
    return A, B


def exact_jacobian(x: Tensor, u: Tensor) -> tuple[Tensor, Tensor]:
    """Derivative of the same discrete forward map, including normalization."""
    shape = x.shape[:-1]
    z = torch.cat((x, u), -1).reshape(-1, 14)
    F = torch.vmap(torch.func.jacrev(lambda zz: ca_step(zz[:10], zz[10:])))(z)
    F = F.reshape(*shape, 10, 14)
    return F[..., :10], F[..., 10:]


def rollout(x0: Tensor, u: Tensor) -> Tensor:
    """Return H+1 states for H controls, with the initial state at index zero."""
    states = [x0]
    for k in range(u.shape[1]):
        states.append(ca_step(states[-1], u[:, k]))
    return torch.stack(states, dim=1)


def objective(x: Tensor, u: Tensor, C: Tensor, c: Tensor) -> Tensor:
    """Original running objective; the returned terminal state has zero cost."""
    z = torch.cat((x[:, :u.shape[1]], u), -1)
    return (0.5*z*(C@z.unsqueeze(-1)).squeeze(-1)+c*z).sum((1, 2))


def hover_initialization(x0: Tensor, horizon: int) -> Tensor:
    result = x0.new_zeros((x0.shape[0], horizon, 4))
    result[..., 0] = MASS*GRAVITY
    return result


def projected_stationarity(x: Tensor, u: Tensor, C: Tensor, c: Tensor, *, jacobian: str) -> Tensor:
    """Maximum reduced-gradient residual in original, unscaled cost units."""
    z = torch.cat((x[:, :u.shape[1]], u), -1)
    gradient = (C@z.unsqueeze(-1)).squeeze(-1)+c
    reduced = gradient[..., 10:].clone()
    derivative = ca_jacobian if jacobian == "ca" else exact_jacobian
    A, B = derivative(x[:, :-2], u[:, :-1])
    costate = gradient[:, -1, :10]
    for k in range(u.shape[1]-2, -1, -1):
        reduced[:, k] += (B[:, k].transpose(-1, -2)@costate.unsqueeze(-1)).squeeze(-1)
        costate = gradient[:, k, :10] + (A[:, k].transpose(-1, -2)@costate.unsqueeze(-1)).squeeze(-1)
    lo, hi = u.new_tensor(LOWER), u.new_tensor(UPPER)
    at_lower, at_upper = (u-lo).abs() <= 1e-8, (u-hi).abs() <= 1e-8
    reduced = torch.where(at_lower, reduced.clamp_max(0),
                          torch.where(at_upper, reduced.clamp_min(0), reduced))
    return reduced.abs().amax((1, 2))
