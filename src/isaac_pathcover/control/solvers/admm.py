"""Python structured ADMM with PyTorch/Triton updates and nvmath linear solves.

Equations, check cadence and status precedence follow ToyotaResearchInstitute/
turbompc@7050a4f (MIT, copyright 2026 The TurboMPC Project Contributors;
license retained in third_party/turbompc/LICENSE).
Persistent tiny-system kernels or CUDA Graphs advance between host decisions;
neither removes adaptive rho, early termination or infeasibility checks.
"""

from dataclasses import fields
from functools import partial
from threading import Lock

import torch

from .cudss import BatchedLU, linear_system
from .qp import ADMMConfig, ADMMResult, QPData, schur_matrix


def mv(a, x):
    return (a * x.unsqueeze(-2)).sum(-1)


def transpose_mv(a, x):
    return (a * x.unsqueeze(-1)).sum(-2)


def constraints(qp, x):
    return (mv(qp.A0, x[:, 0]),
            mv(qp.A_minus, x[:, :-1]) + mv(qp.A_plus, x[:, 1:]),
            mv(qp.G, x))


def equality_transpose(qp, y0, yd):
    first = transpose_mv(qp.A0, y0)
    minus, plus = transpose_mv(qp.A_minus, yd), transpose_mv(qp.A_plus, yd)
    zero = first.new_zeros((first.shape[0], 1, first.shape[1]))
    return (torch.cat((first[:, None], torch.zeros_like(plus)), 1)
            + torch.cat((minus, zero), 1) + torch.cat((zero, plus), 1))


def hessian_mv(qp, x):
    base = mv(qp.D, x)
    zero = base[:, :1] * 0
    return (base + torch.cat((zero, mv(qp.E, x[:, :-1])), 1)
            + torch.cat((transpose_mv(qp.E, x[:, 1:]), zero), 1))


def batch_norm(*values):
    norms = [v.flatten(1).abs().amax(1) for v in values if v.numel()]
    return torch.stack(norms).amax(0)


def _gamma(qp, state, rhs, status, *, sigma, rho_factor):
    x, z, y, y0, yd, xi, rho = state
    rf = rho * rho_factor
    gamma = (sigma * x - qp.q
             + equality_transpose(qp, rf[:, None] * qp.c0 - y0,
                                  rf[:, None, None] * qp.c_dyn - yd)
             + transpose_mv(qp.G, rho[:, None, None] * z - y))
    rhs.copy_(torch.where((status == -1)[:, None, None], gamma, rhs))


def _positive_root(offset, kappa):
    root = torch.sqrt(offset.square() + 4 * kappa)
    return torch.where(offset >= 0, .5 * (offset + root), 2 * kappa / (root - offset))


def _barrier_prox(value, lower, upper, kappa):
    finite_l, finite_u = lower.isfinite(), upper.isfinite()
    # Sanitized operands keep unused infinite-bound branches out of Newton divisions.
    lo, hi = torch.where(finite_l, lower, 0.), torch.where(finite_u, upper, 1.)
    mid = lo + .5 * (hi - lo)
    lo_inside = torch.nextafter(lo, hi)
    hi_inside = torch.nextafter(hi, lo)
    estimate = torch.where(value <= mid, torch.minimum(lo + _positive_root(value - lo, kappa), mid),
                            torch.maximum(hi - _positive_root(hi - value, kappa), mid))
    estimate = estimate.clamp(lo_inside, hi_inside)
    for _ in range(7):
        dl, du = estimate - lo, hi - estimate
        residual = estimate - value + kappa * (1 / du - 1 / dl)
        derivative = 1 + kappa * (1 / du.square() + 1 / dl.square())
        step = -residual / derivative
        fraction = torch.where(step < 0, .99 * dl / (-step),
                               torch.where(step > 0, .99 * du / step, 1.))
        estimate = (estimate + fraction.clamp_max(1) * step).clamp(lo_inside, hi_inside)
    left = torch.maximum(lo + _positive_root(value - lo, kappa),
                         torch.nextafter(lo, torch.full_like(lo, float("inf"))))
    right = torch.minimum(hi - _positive_root(hi - value, kappa),
                          torch.nextafter(hi, torch.full_like(hi, -float("inf"))))
    return torch.where(finite_l & finite_u, estimate,
                       torch.where(finite_l, left, torch.where(finite_u, right, value)))


def _post_solve(qp, state, solved_x, status, *, alpha, rho_factor,
                use_slack, slack_weight, barrier, kappa):
    x, z, y, y0, yd, xi, rho = state
    oldx = x
    c0, cd, gx = constraints(qp, solved_x)
    rf, rg = rho[:, None] * rho_factor, rho[:, None, None]
    z_relaxed = alpha * gx + (1 - alpha) * z
    z_tilde = z_relaxed + y / rg
    if barrier:
        scale = kappa * (1 / rg + 1 / slack_weight) if use_slack else kappa / rg
        projected = _barrier_prox(z_tilde, qp.lower, qp.upper, scale)
    else:
        projected = z_tilde.clamp(qp.lower, qp.upper)
    if use_slack:
        newxi = rg / (slack_weight + rg) * (projected - z_tilde)
        newz = projected - newxi
    else:
        newxi, newz = torch.zeros_like(xi), projected
    active = (status == -1)[:, None, None]
    x.copy_(torch.where(active, alpha * solved_x + (1 - alpha) * oldx, oldx))
    y.copy_(torch.where(active, y + rg * (z_relaxed - newz), y))
    y0.copy_(torch.where(active[:, :, 0], y0 + alpha * rf * (c0 - qp.c0), y0))
    yd.copy_(torch.where(active, yd + alpha * rf[:, :, None] * (cd - qp.c_dyn), yd))
    z.copy_(torch.where(active, newz, z))
    xi.copy_(torch.where(active, newxi, xi))


def _measure(qp, state, previous, *, use_slack, slack_weight):
    x, z, y, y0, yd, xi, rho = state
    oldx, oldy, oldy0, oldyd, oldxi = previous
    c0, cd, gx = constraints(qp, x)
    px = hessian_mv(qp, x)
    aty = equality_transpose(qp, y0, yd) + transpose_mv(qp.G, y)
    primal = batch_norm(c0 - qp.c0, cd - qp.c_dyn, gx - z)
    primal_norm = batch_norm(c0, cd, qp.c0, qp.c_dyn, gx, z)
    dual = batch_norm(px + qp.q + aty)
    dual_norm = batch_norm(px, qp.q, aty)
    if use_slack and xi.numel():
        dual = torch.maximum(dual, batch_norm(slack_weight * xi + y))
        dual_norm = torch.maximum(dual_norm, batch_norm(slack_weight * xi, y))
    finite = torch.stack([a.flatten(1).isfinite().all(1) for a in state[:-1]]).all(0) & rho.isfinite()

    # The certificate direction is the immediately preceding iterate, as in CUDA.
    dx, dy, dy0, dyd, dxi = x - oldx, y - oldy, y0 - oldy0, yd - oldyd, xi - oldxi
    dy_norm = batch_norm(dy, dy0, dyd)
    stationarity = batch_norm(equality_transpose(qp, dy0, dyd) + transpose_mv(qp.G, dy))
    if use_slack and dy.numel():
        stationarity = torch.maximum(stationarity, batch_norm(dy))
    support = ((qp.c0 * dy0).flatten(1).sum(1) + (qp.c_dyn * dyd).flatten(1).sum(1)
               + torch.where(dy > 0, qp.upper * dy,
                             torch.where(dy < 0, qp.lower * dy, 0.)).flatten(1).sum(1))
    dx_norm = batch_norm(dx, dxi) if use_slack else batch_norm(dx)
    pdx = hessian_mv(qp, dx)
    hessian = batch_norm(pdx, slack_weight * dxi) if use_slack else batch_norm(pdx)
    objective = (qp.q * dx).flatten(1).sum(1)
    d0, dd, dg = constraints(qp, dx)
    if use_slack:
        dg = dg + dxi
    lf, uf = qp.lower.isfinite(), qp.upper.isfinite()
    violation = torch.where(lf & uf, dg.abs(),
                            torch.where(lf, (-dg).clamp_min(0),
                                        torch.where(uf, dg.clamp_min(0), 0.)))
    direction = batch_norm(d0, dd, violation)
    return (torch.stack((primal, dual, primal_norm, dual_norm), 1), finite,
            torch.stack((dy_norm, stationarity, support, dx_norm, hessian, objective, direction), 1))


def _checkpoint(qp, state, previous, status, iterations, terminal_residuals,
                iteration, *, cfg):
    metrics, finite, cert = _measure(qp, state, previous,
                                     use_slack=cfg.use_slack, slack_weight=cfg.slack_weight)
    p, d, pn, dn = metrics.unbind(1)
    pt, dt = cfg.eps_abs + cfg.eps_rel * pn, cfg.eps_abs + cfg.eps_rel * dn
    last = iteration == cfg.max_iter
    termination = last | (iteration % cfg.check_termination_every == 0)
    infeasibility = (last | (iteration % cfg.check_infeasibility_every == 0)) & (cfg.constraint_handling == 0)
    checking = termination | infeasibility
    active = status == -1
    finite = finite & metrics.isfinite().all(1)
    floor = 10 * torch.finfo(torch.float64).eps
    solved = finite & (p <= pt) & (d <= dt) & termination
    pi = ((cert[:, 0] > floor) & (cert[:, 1] <= cfg.eps_prim_inf * cert[:, 0])
          & (cert[:, 2] <= -cfg.eps_prim_inf * cert[:, 0]) & infeasibility)
    di = ((cert[:, 3] > floor) & (cert[:, 4] <= cfg.eps_dual_inf * cert[:, 3])
          & (cert[:, 5] <= -cfg.eps_dual_inf * cert[:, 3])
          & (cert[:, 6] <= cfg.eps_dual_inf * cert[:, 3]) & infeasibility)
    new_status = torch.where(last, 1, -1).expand_as(status)
    new_status = torch.where(di, 3, new_status)
    new_status = torch.where(pi, 2, new_status)
    new_status = torch.where(solved, 0, new_status)
    new_status = torch.where(~finite, 4, new_status)
    terminal = active & checking & (new_status != -1)
    iterations.copy_(torch.where(terminal, iteration, iterations))
    status.copy_(torch.where(active & checking, new_status, status))
    terminal_residuals.copy_(torch.where((active & checking)[:, None],
                                         torch.stack((p, d, pt, dt), 1), terminal_residuals))
    changed = torch.zeros_like(status, dtype=torch.bool)
    if cfg.adapt_rho_every:
        rho_check = iteration % cfg.adapt_rho_every == 0
        rho = state[-1]
        ratio = torch.sqrt((p / (1e-10 + pn)) / (d / (1e-10 + dn) + 1e-30))
        candidate = (rho * ratio.clamp(1 / cfg.rho_update_max_factor,
                                       cfg.rho_update_max_factor)).clamp(cfg.rho_min, cfg.rho_max)
        scale = candidate / rho
        changed = ((torch.maximum(scale, 1 / scale) >= cfg.adaptive_rho_tolerance)
                   & (status == -1) & rho_check)
        rho.copy_(torch.where(changed, candidate, rho))
    return torch.cat((status, changed.to(torch.int32)))


class _Workspace:
    def __init__(self, qp, config, compile_kernels, capture_graph):
        self.config = config
        self.qp = QPData(**{f.name: getattr(qp, f.name).clone().contiguous() for f in fields(qp)})
        b, t, n = qp.q.shape
        self.linear = linear_system(b, t, n, qp.q.device)
        self.persistent = (compile_kernels and isinstance(self.linear, BatchedLU)
                           and max(t*n, qp.A0.shape[-2], (t-1)*qp.A_minus.shape[-2],
                                   t*qp.G.shape[-2]) <= 32)
        self.stream = torch.cuda.current_stream(qp.q.device)
        self.state = [torch.zeros_like(a) for a in
                      (qp.q, qp.lower, qp.lower, qp.c0, qp.c_dyn, qp.lower)]
        self.state.append(qp.q.new_full((b,), config.rho))
        self.previous = [torch.zeros_like(self.state[i]) for i in (0, 2, 3, 4, 5)]
        self.status = torch.full((b,), -1, dtype=torch.int32, device=qp.q.device)
        self.iterations = torch.full_like(self.status, config.max_iter)
        self.current_iteration = torch.zeros((), dtype=torch.int32, device=qp.q.device)
        self.terminal_residuals = qp.q.new_zeros((b, 4))
        self.graphs = {}
        self.capture_graph = capture_graph
        pre, post = _gamma, _post_solve
        if compile_kernels:
            # Per-QP programs skip completed rows before reading the matrices.
            # A batched torch.where only masks the result after doing the work.
            from .admm_kernels import gamma, post_solve
            pre, post = gamma, post_solve
        self.pre = partial(pre, sigma=config.sigma, rho_factor=config.rho_f_factor)
        self.post = partial(post, alpha=config.alpha, rho_factor=config.rho_f_factor,
                            use_slack=config.use_slack, slack_weight=config.slack_weight,
                            barrier=config.constraint_handling == 1, kappa=config.log_barrier_kappa)
        checkpoint, schur = _checkpoint, schur_matrix
        if compile_kernels:
            # Compile the functions, not functools.partial's shared dispatcher:
            # otherwise unrelated kernels/shapes exhaust one recompile cache.
            checkpoint = torch.compile(checkpoint, fullgraph=True, dynamic=False)
            schur = torch.compile(schur, fullgraph=True, dynamic=False)
        self.checkpoint = partial(checkpoint, cfg=config)
        self.schur = partial(schur, config=config)

    def reset(self, qp, warm):
        for f in fields(qp):
            getattr(self.qp, f.name).copy_(getattr(qp, f.name))
        if warm is None:
            for x in self.state[:-1]:
                x.zero_()
            self.state[1].clamp_(qp.lower, qp.upper)
            self.state[-1].fill_(self.config.rho)
        else:
            for dst, name in zip(self.state, ("x", "z", "y", "lambda0", "lambda_dyn", "xi", "rho")):
                dst.copy_(getattr(warm, name))
        self.status.fill_(-1)
        self.iterations.fill_(self.config.max_iter)
        self.terminal_residuals.zero_()
        self.linear.rhs.zero_()
        self.linear.factorize(self.schur(self.qp, self.state[-1]))

    def step(self, snapshot=False):
        if snapshot:
            for target, index in zip(self.previous, (0, 2, 3, 4, 5)):
                target.copy_(self.state[index])
        self.pre(self.qp, self.state, self.linear.rhs, self.status)
        self.linear.solve()
        self.post(self.qp, self.state, self.linear.solution, self.status)

    def run_segment(self, length, save_previous):
        if self.persistent:
            from .admm_kernels import run_segment
            run_segment(self, length, save_previous)
            return
        if not self.capture_graph:
            for i in range(length):
                self.step(save_previous and i == length - 1)
            return
        key = (length, save_previous)
        if key not in self.graphs:
            # Capture only after kernels and cuDSS's solve workspace are initialized.
            snapshot = [x.clone() for x in self.state]
            self.step(False)
            if save_previous:
                self.step(True)
            self.stream.synchronize()
            for target, source in zip(self.state, snapshot):
                target.copy_(source)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=self.stream):
                for i in range(length):
                    self.step(save_previous and i == length - 1)
            self.graphs[key] = graph
        self.graphs[key].replay()

    def check(self, iteration):
        self.current_iteration.fill_(iteration)
        flags = self.checkpoint(self.qp, self.state, self.previous, self.status,
                                self.iterations, self.terminal_residuals, self.current_iteration)
        host = flags.cpu().numpy()
        b = len(self.status)
        mask = host[:b] == -1
        if not mask.any():
            return True
        if host[b:].any():
            self.linear.factorize(self.schur(self.qp, self.state[-1]))
        return False

    def solve(self, qp, warm):
        self.reset(qp, warm)
        iteration = 0
        cfg = self.config
        while iteration < cfg.max_iter:
            periods = [cfg.check_termination_every]
            if cfg.constraint_handling == 0:
                periods.append(cfg.check_infeasibility_every)
            if cfg.adapt_rho_every:
                periods.append(cfg.adapt_rho_every)
            until = min([cfg.max_iter - iteration]
                        + [p - iteration % p for p in periods if p > 0])
            next_iteration = iteration + until
            infeasibility = cfg.constraint_handling == 0 and (
                next_iteration == cfg.max_iter or next_iteration % cfg.check_infeasibility_every == 0)
            self.run_segment(until, infeasibility)
            iteration = next_iteration
            if self.check(iteration):
                break
        x, z, y, y0, yd, xi, rho = (a.clone() for a in self.state)
        return ADMMResult(self.iterations.clone(), x, z, y, y0, yd, xi, rho,
                          torch.full_like(rho, float("nan")), self.status.clone(), self.terminal_residuals.clone())

    def close(self):
        self.stream.synchronize()
        self.graphs.clear()
        self.linear.close()


class ADMM:
    """Structured ADMM with Python GPU kernels and official NVIDIA bindings.

    Graph segments preserve adaptive rho, check cadence and early termination.
    Warm starts and returned results never alias the workspace buffers.
    """

    def __init__(self, config=None, *, compile_kernels=True, capture_graph=True):
        self.config = config or ADMMConfig()
        cfg = self.config
        if cfg.max_iter < 1 or cfg.check_termination_every < 1 or cfg.check_infeasibility_every < 1:
            raise ValueError("ADMM iteration budgets/check periods must be positive")
        if cfg.adapt_rho_every < 0 or cfg.sigma <= 0 or cfg.rho <= 0 or not 0 < cfg.alpha < 2:
            raise ValueError("Invalid ADMM regularization or relaxation")
        if cfg.constraint_handling not in (0, 1):
            raise ValueError("Unknown constraint handling")
        self.compile_kernels, self.capture_graph = compile_kernels, capture_graph
        self._workspaces = {}
        self._device = None
        self._closed = False
        self._lock = Lock()

    # Cached graph buffers cross PPO inference and gradient-enabled minibatches.
    @torch.inference_mode(False)
    @torch.no_grad()
    def solve(self, qp, warm_start=None):
        with self._lock:
            if self._closed:
                raise RuntimeError("ADMM solver is closed")
            if not qp.q.is_cuda or qp.q.dtype != torch.float64:
                raise ValueError("ADMM requires float64 CUDA QP tensors")
            if self._device is None:
                self._device = qp.q.device
            if self._device != qp.q.device:
                raise ValueError("Use one ADMM instance per CUDA device")
            if qp.q.ndim != 3 or qp.A0.ndim != 3 or qp.A_minus.ndim != 4 or qp.G.ndim != 4:
                raise ValueError("Invalid QP tensor contract: tensor ranks")
            b, t, n = qp.q.shape
            n0, nx, m = qp.A0.shape[-2], qp.A_minus.shape[-2], qp.G.shape[-2]
            if min(b, n, n0, nx) <= 0 or t < 2 or b * t * max(n, m) > 2**31 - 1:
                raise ValueError("Invalid QP tensor contract: dimensions")
            shapes = ((b,t,n,n), (b,t-1,n,n), (b,t,n), (b,n0,n),
                      (b,t-1,nx,n), (b,t-1,nx,n), (b,n0), (b,t-1,nx),
                      (b,t,m,n), (b,t,m), (b,t,m))
            for field, shape in zip(fields(qp), shapes):
                value = getattr(qp, field.name)
                if value.shape != shape or value.dtype != torch.float64 or value.device != self._device:
                    raise ValueError(f"Invalid QP tensor contract: {field.name}")
            if warm_start is not None:
                for name, shape in zip(("x", "z", "y", "lambda0", "lambda_dyn", "xi", "rho"),
                                       ((b,t,n),(b,t,m),(b,t,m),(b,n0),(b,t-1,nx),(b,t,m),(b,))):
                    value = getattr(warm_start, name)
                    if value.shape != shape or value.dtype != torch.float64 or value.device != self._device:
                        raise ValueError(f"Invalid warm-start tensor contract: {name}")
            qp = qp.contiguous()
            caller = torch.cuda.current_stream(qp.q.device)
            key = (caller.cuda_stream, tuple((f.name, tuple(getattr(qp, f.name).shape)) for f in fields(qp)))
            if key not in self._workspaces:
                # cuDSS analysis, factors, solves and capture share one stream for
                # their entire lifetime. Moving the handle after analysis leaves
                # cuDSS 0.8's internal solve stream outside the captured graph.
                stream = torch.cuda.Stream(device=qp.q.device) if self.capture_graph else caller
                stream.wait_stream(caller)
                with torch.cuda.stream(stream):
                    self._workspaces[key] = _Workspace(qp, self.config, self.compile_kernels, self.capture_graph)
            workspace = self._workspaces[key]
            workspace.stream.wait_stream(caller)
            # QP temporaries belong to the caller's allocator, not this solve stream.
            for field in fields(qp):
                getattr(qp, field.name).record_stream(workspace.stream)
            with torch.cuda.stream(workspace.stream):
                result = workspace.solve(qp, warm_start)
            caller.wait_stream(workspace.stream)
            for field in fields(result):
                getattr(result, field.name).record_stream(caller)
            return result

    def close(self):
        with self._lock:
            self._closed = True
            for workspace in self._workspaces.values():
                workspace.close()
            self._workspaces.clear()
