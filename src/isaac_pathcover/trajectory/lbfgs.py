"""Python port of GCOPTER's cautious L-BFGS and Lewis–Overton line search.

Source: PathCover 8723a05d, trajectory_server/include/gcopter/lbfgs.hpp.
Copyright (c) 2021 Zhepei Wang. Distributed under the upstream MIT license.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class LBFGSParameters:
    mem_size: int = 8
    g_epsilon: float = 1.0e-5
    past: int = 3
    delta: float = 1.0e-6
    max_iterations: int = 0
    max_linesearch: int = 64
    min_step: float = 1.0e-20
    max_step: float = 1.0e20
    f_dec_coeff: float = 1.0e-4
    s_curv_coeff: float = 0.9
    cautious_factor: float = 1.0e-6
    machine_prec: float = 1.0e-16


@dataclass
class LBFGSResult:
    x: np.ndarray
    cost: float
    gradient: np.ndarray
    status: int
    iterations: int
    evaluations: int

    @property
    def success(self):
        return self.status >= 0

    @property
    def message(self):
        return {
            0: "gradient convergence", 1: "relative cost convergence",
            -1012: "nonfinite objective", -1011: "minimum line-search step",
            -1010: "maximum line-search step", -1009: "line-search limit",
            -1008: "iteration limit", -1007: "line-search interval too small",
            -1006: "nonpositive line-search step", -1005: "non-descent direction",
        }.get(self.status, str(self.status))


def _line_search(fun, xp, gp, finit, direction, step, p):
    if not step > 0:
        return xp, finit, gp, step, -1006, 0
    dginit = float(gp @ direction)
    if dginit > 0:
        return xp, finit, gp, step, -1005, 0
    mu, nu = 0.0, p.max_step
    bracketed, touched = False, False
    for count in range(1, p.max_linesearch + 1):
        x = xp + step * direction
        f, g = fun(x)
        if not np.isfinite(f):
            return x, f, g, step, -1012, count
        if f > finit + step * p.f_dec_coeff * dginit:
            nu, bracketed = step, True
        elif g @ direction < p.s_curv_coeff * dginit:
            mu = step
        else:
            return x, f, g, step, count, count
        if count == p.max_linesearch:
            return x, f, g, step, -1009, count
        if bracketed and nu - mu < p.machine_prec * nu:
            return x, f, g, step, -1007, count
        step = 0.5 * (mu + nu) if bracketed else 2.0 * step
        if step < p.min_step:
            return x, f, g, step, -1011, count
        if step > p.max_step:
            if touched:
                return x, f, g, step, -1010, count
            touched, step = True, p.max_step


def minimize(fun, initial, parameters=None):
    """Minimize an objective returning (cost, analytic gradient)."""
    p = parameters or LBFGSParameters()
    x = np.asarray(initial, dtype=np.float64).copy()
    if x.ndim != 1 or x.size == 0:
        raise ValueError("L-BFGS requires a nonempty vector")
    if p.mem_size <= 0 or p.max_linesearch <= 0 or p.max_iterations < 0:
        raise ValueError("invalid L-BFGS iteration/memory parameters")
    f, g = fun(x)
    g = np.asarray(g, dtype=np.float64)
    calls = 1
    if not np.isfinite(f):
        return LBFGSResult(x, f, g, -1012, 0, calls)
    if np.max(np.abs(g)) / max(1.0, np.max(np.abs(x))) < p.g_epsilon or not np.any(g):
        return LBFGSResult(x, f, g, 0, 0, calls)
    direction = -g
    step = 1.0 / np.linalg.norm(direction)
    pf = np.empty(max(1, p.past))
    pf[0] = f
    s_mem = np.zeros((p.mem_size, x.size))
    y_mem = np.zeros_like(s_mem)
    ys_mem = np.zeros(p.mem_size)
    alpha = np.zeros(p.mem_size)
    end, bound, k = 0, 0, 1
    while True:
        xp, gp, fp = x, g, f
        x, f, g, step, status, count = _line_search(
            fun, xp, gp, fp, direction, step, p)
        calls += count
        if status < 0:
            return LBFGSResult(xp, fp, gp, status, k, calls)
        if np.max(np.abs(g)) / max(1.0, np.max(np.abs(x))) < p.g_epsilon or not np.any(g):
            return LBFGSResult(x, f, g, 0, k, calls)
        if p.past:
            if k >= p.past and abs(pf[k % p.past] - f) / max(1.0, abs(f)) < p.delta:
                return LBFGSResult(x, f, g, 1, k, calls)
            pf[k % p.past] = f
        if p.max_iterations and k >= p.max_iterations:
            return LBFGSResult(x, f, g, -1008, k, calls)
        k += 1
        s_mem[end], y_mem[end] = x - xp, g - gp
        ys = float(y_mem[end] @ s_mem[end])
        yy = float(y_mem[end] @ y_mem[end])
        ys_mem[end] = ys
        direction = -g.copy()
        cautious = float(s_mem[end] @ s_mem[end]) * np.linalg.norm(gp) * p.cautious_factor
        if ys > cautious:
            bound = min(p.mem_size, bound + 1)
            end = (end + 1) % p.mem_size
            j = end
            for _ in range(bound):
                j = (j + p.mem_size - 1) % p.mem_size
                alpha[j] = s_mem[j] @ direction / ys_mem[j]
                direction -= alpha[j] * y_mem[j]
            direction *= ys / yy
            for _ in range(bound):
                beta = y_mem[j] @ direction / ys_mem[j]
                direction += (alpha[j] - beta) * s_mem[j]
                j = (j + 1) % p.mem_size
        step = 1.0
