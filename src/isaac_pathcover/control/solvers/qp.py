"""Structured QP data and published TurboMPC ADMM settings."""

from dataclasses import dataclass, fields

import torch


@dataclass(frozen=True)
class QPData:
    """Batched stage coordinates, with lower cost off-diagonal blocks E."""

    D: torch.Tensor
    E: torch.Tensor
    q: torch.Tensor
    A0: torch.Tensor
    A_minus: torch.Tensor
    A_plus: torch.Tensor
    c0: torch.Tensor
    c_dyn: torch.Tensor
    G: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor

    def contiguous(self):
        return QPData(**{f.name: getattr(self, f.name).contiguous() for f in fields(self)})


@dataclass(frozen=True)
class ADMMConfig:
    """The upstream projection ADMM settings, with an explicit iteration budget."""

    max_iter: int = 4096
    check_termination_every: int = 25
    check_infeasibility_every: int = 25
    adapt_rho_every: int = 25
    eps_abs: float = 1e-5
    eps_rel: float = 1e-5
    eps_prim_inf: float = 1e-4
    eps_dual_inf: float = 1e-4
    rho: float = 0.1
    sigma: float = 1e-6
    rho_f_factor: float = 1000.0
    alpha: float = 1.6
    adaptive_rho_tolerance: float = 5.0
    rho_update_max_factor: float = 1e6
    rho_min: float = 1e-6
    rho_max: float = 1e6
    constraint_handling: int = 0
    log_barrier_kappa: float = 0.0
    use_slack: bool = False
    slack_weight: float = 0.0


@dataclass(frozen=True)
class ADMMResult:
    """Independent output tensors, including all upstream warm-start buffers."""

    iterations: torch.Tensor
    x: torch.Tensor
    z: torch.Tensor
    y: torch.Tensor
    lambda0: torch.Tensor
    lambda_dyn: torch.Tensor
    xi: torch.Tensor
    rho: torch.Tensor
    kernel_ns: torch.Tensor
    status: torch.Tensor
    residuals: torch.Tensor


def schur_matrix(qp: QPData, rho: torch.Tensor, config: ADMMConfig) -> torch.Tensor:
    """Build P + sigma I + rho_f C.T C + rho G.T G in banded blocks."""
    b, t, n, _ = qp.D.shape
    rf = (rho * config.rho_f_factor)[:, None, None, None]
    diagonal = qp.D + config.sigma * torch.eye(n, dtype=qp.D.dtype, device=qp.D.device)
    diagonal = diagonal + rho[:, None, None, None] * (qp.G.transpose(-1, -2) @ qp.G)
    diagonal[:, 0] += rf[:, 0] * (qp.A0.transpose(-1, -2) @ qp.A0)
    diagonal[:, :-1] += rf * (qp.A_minus.transpose(-1, -2) @ qp.A_minus)
    diagonal[:, 1:] += rf * (qp.A_plus.transpose(-1, -2) @ qp.A_plus)
    lower = qp.E + rf * (qp.A_plus.transpose(-1, -2) @ qp.A_minus)
    matrix = qp.D.new_zeros((b, t, n, 3 * n))
    matrix[..., n : 2 * n] = diagonal
    matrix[:, 1:, :, :n] = lower
    matrix[:, :-1, :, 2 * n :] = lower.transpose(-1, -2)
    return matrix
