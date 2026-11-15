"""Isaac-L2F TurboMPC SQP for the source ten-state CA flight model.

SQP linearization, structured ADMM, four-step line search and warm controls
are copied from Isaac-L2F e36743e8076bfe12b3e11911e2f1f866d61ae158.
The source prediction and physical input bounds are unchanged. The learning
and implicit-adjoint path is omitted for this trajectory tracking controller.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

from .admm import ADMM
from .model import LOWER, UPPER, ca_step, exact_jacobian, objective, rollout
from .qp import ADMMConfig, QPData


NX, NU = 10, 4


def mv(matrix, vector):
    return (matrix @ vector.unsqueeze(-1)).squeeze(-1)


class TurboMPC:
    def __init__(self, horizon=16, max_iter=3, *, compile_kernels=True):
        self.horizon, self.max_iter = horizon, max_iter
        self.cost_scale, self.tolerance = 100., 1e-5
        self.config = ADMMConfig(max_iter=500, eps_abs=1e-5, eps_rel=1e-4)
        self.admm = ADMM(self.config, compile_kernels=compile_kernels)
        self.last_info = None
        self.compile_kernels = compile_kernels
        self._rollout = rollout
        if compile_kernels:
            os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(Path.cwd()/".runtime"/"inductor"))
            os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "2")
            self.linearize = torch.compile(self.linearize, fullgraph=True, dynamic=False)
            self._line_search = torch.compile(self._line_search, fullgraph=True, dynamic=False)
            self._rollout = torch.compile(rollout, fullgraph=True, dynamic=False)

    def linearize(self, x, u):
        A, B = exact_jacobian(x[:, :-2], u[:, :-1])
        F = torch.cat((A, B), -1)
        z = torch.cat((x[:, :-2], u[:, :-1]), -1)
        return F, ca_step(x[:, :-2], u[:, :-1]) - mv(F, z)

    def qp_data(self, x0, C, c, F, f):
        b, h, width = c.shape
        eye = torch.eye(NX, width, device=x0.device, dtype=x0.dtype)
        selector = torch.cat((eye.new_zeros((NU, NX)), torch.eye(NU, device=x0.device, dtype=x0.dtype)), -1)
        return QPData(C / self.cost_scale, C.new_zeros((b, h-1, width, width)), c / self.cost_scale,
                      eye.expand(b, NX, width), F, -eye.expand(b, h-1, NX, width), x0, -f,
                      selector.expand(b, h, NU, width),
                      x0.new_tensor(LOWER).expand(b, h, NU), x0.new_tensor(UPPER).expand(b, h, NU))

    def _line_search(self, x0, C, c, x, u, proposed):
        b, h = u.shape[:2]
        width = NX + NU
        alphas = x0.new_tensor((1., .5, .25, .1))
        trial_u = u[:, None] + alphas[None, :, None, None] * (proposed-u)[:, None]
        trial_x = self._rollout(x0[:, None].expand(b, 4, NX).reshape(-1, NX),
                               trial_u.reshape(-1, h, NU))
        trial_cost = objective(trial_x, trial_u.reshape(-1, h, NU),
                               C[:, None].expand(b, 4, h, width, width).reshape(-1, h, width, width),
                               c[:, None].expand(b, 4, h, width).reshape(-1, h, width)).reshape(b, 4)
        best_cost, best_index = trial_cost.min(1)
        accepted = best_cost < objective(x, u, C, c)
        batch = torch.arange(b, device=x0.device)
        next_u = torch.where(accepted[:, None, None], trial_u[batch, best_index], u)
        next_x = torch.where(accepted[:, None, None], trial_x.reshape(b, 4, h+1, NX)[batch, best_index], x)
        return next_x, next_u, (next_u-u).abs().amax((1, 2))

    @torch.no_grad()
    def solve(self, x0, C, c, u_init):
        x0, C, c, u = x0.double(), C.double(), c.double(), u_init.detach().double().clone()
        lower, upper = u.new_tensor(LOWER), u.new_tensor(UPPER)
        u = u.clamp(lower, upper)
        x = self._rollout(x0, u)
        warm = None
        for iteration in range(self.max_iter):
            F, f = self.linearize(x, u)
            qp = self.qp_data(x0, C, c, F, f)
            result = self.admm.solve(qp, warm)
            proposed = result.x[..., NX:].clamp(lower, upper)
            next_x, next_u, change = self._line_search(x0, C, c, x, u, proposed)
            failed, converged = torch.stack(((result.status >= 2).any() | (~torch.isfinite(result.x)).any(),
                                             (change <= self.tolerance).all())).tolist()
            if failed:
                raise RuntimeError(f"TurboMPC QP failure: {result.status.tolist()}")
            x, u, warm = next_x, next_u, result
            if converged:
                break
        self.last_info = {"iterations": iteration+1, "qp_status": result.status.detach(),
                          "qp_iterations": result.iterations.detach(), "qp_residuals": result.residuals.detach(),
                          "last_control_change": change.detach()}
        return x, u, self.last_info

    def close(self):
        self.admm.close()
