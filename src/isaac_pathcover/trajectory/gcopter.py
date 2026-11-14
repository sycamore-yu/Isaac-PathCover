"""GCOPTER 3-D corridor optimizer, ported from PathCover 8723a05d.

The time transform, normalized squared vertex coordinates, shortest-path
initialization, capped penalty margins, trapezoidal penalty integral and
L-BFGS settings follow trajectory_server/include/gcopter_solver.hpp.
Copyright (c) 2021 Zhepei Wang and PathCover contributors (MIT/BSD-3-Clause).
"""
from dataclasses import dataclass
from time import perf_counter
import numpy as np

from ..planning.geometry import GeometryError, chebyshev_center, vertices
from .flatness import FlatnessMap
from .lbfgs import LBFGSParameters, minimize
from .minco import MINCO, PiecewisePolynomial, derivative_bases


@dataclass(frozen=True)
class TrajectoryConfig:
    cost_order: int = 3
    time_weight: float = 100.0
    safety_margin: float = 0.15
    length_per_piece: float = 2.0
    pieces_per_polytope: int = 1
    smoothing_eps: float = 0.01
    integral_resolution: int = 16
    magnitude_bounds: tuple = (3.0, 3.0, 0.7854, 2.0, 10.0)
    penalty_weights: tuple = (10000.0, 100.0, 10000.0, 10000.0, 10000.0)
    physical_parameters: tuple = (0.716, 9.81, 0.0, 0.0, 0.0, 0.001)
    relative_cost_tolerance: float = 1.0e-4
    max_iterations: int = 1000


@dataclass
class OptimizationResult:
    trajectory: PiecewisePolynomial | None
    success: bool
    status: str
    status_code: int
    cost: float
    iterations: int
    evaluations: int
    elapsed_s: float


def forward_t(tau):
    tau = np.asarray(tau)
    return np.where(tau > 0, (0.5 * tau + 1) * tau + 1,
                    1 / ((0.5 * tau - 1) * tau + 1))


def inverse_t(times):
    times = np.asarray(times)
    out = np.empty_like(times)
    mask = times > 1
    out[mask] = np.sqrt(2 * times[mask] - 1) - 1
    out[~mask] = 1 - np.sqrt(2 / times[~mask] - 1)
    return out


def backward_grad_t(tau, gradient):
    den = (0.5 * tau - 1) * tau + 1
    return gradient * np.where(tau > 0, tau + 1, (1 - tau) / den ** 2)


def smoothed_l1(value, mu):
    """Original C2 positive-part approximation and its derivative."""
    value = np.asarray(value)
    z = np.clip(value / mu, 0.0, 1.0)
    f = np.where(value > mu, value - 0.5 * mu, (mu - 0.5 * value) * z ** 3)
    df = np.where(value > mu, 1.0, z ** 2 * (3.0 - 2.0 * z))
    return f, df


def _forward_points(xi, vpolys):
    points, offset = [], 0
    for poly in vpolys:
        k = len(poly)
        q = xi[offset:offset + k]
        r = q[:-1] / np.linalg.norm(q)
        points.append(poly[0] + r ** 2 @ poly[1:])
        offset += k
    return np.asarray(points).reshape(-1, 3)


def _backward_points(xi, vpolys, grad_points, norm_penalty=True):
    grad, cost, offset = np.empty_like(xi), 0.0, 0
    for poly, gp in zip(vpolys, grad_points):
        k = len(poly)
        q = xi[offset:offset + k]
        norm = np.linalg.norm(q)
        u = q / norm
        g = np.r_[2 * u[:-1] * (poly[1:] @ gp), 0.0]
        g = (g - u * (u @ g)) / norm
        violation = float(q @ q) - 1.0
        if norm_penalty and violation > 0:
            cost += violation ** 3
            g += 6.0 * violation ** 2 * q
        grad[offset:offset + k] = g
        offset += k
    return cost, grad


def _constant_xi(vpolys):
    return np.concatenate([np.full(len(poly), np.sqrt(1.0 / len(poly))) for poly in vpolys])


def _inverse_points(points, vpolys):
    all_xi = []
    params = LBFGSParameters(past=0, delta=1e-5,
                             g_epsilon=np.finfo(np.float32).eps, max_iterations=128)
    for point, poly in zip(points, vpolys):
        def objective(q):
            current = _forward_points(q, [poly])[0]
            delta = current - point
            c, g = _backward_points(q, [poly], np.array([2 * delta]))
            return float(delta @ delta) + c, g
        initial = np.full(len(poly), np.sqrt(1.0 / len(poly)))
        all_xi.append(minimize(objective, initial, params).x)
    return np.concatenate(all_xi) if all_xi else np.empty(0)


def _shortest_path(start, end, vpolys, smooth):
    overlaps = vpolys[1::2]
    if not overlaps:
        return np.array([start, end])
    def objective(x):
        points = np.vstack([start, _forward_points(x, overlaps), end])
        delta = np.diff(points, axis=0)
        distance = np.sqrt(np.sum(delta * delta, axis=1) + smooth)
        directions = delta / distance[:, None]
        gp = directions[:-1] - directions[1:]
        c, g = _backward_points(x, overlaps, gp)
        return float(distance.sum()) + c, g
    result = minimize(objective, _constant_xi(overlaps),
                      LBFGSParameters(past=3, delta=1e-3, g_epsilon=1e-5))
    return np.vstack([start, _forward_points(result.x, overlaps), end])


class GCopter:
    def __init__(self, corridor, start_state, end_state, config=None):
        self.config = config or TrajectoryConfig()
        cfg = self.config
        self.hpolys = []
        self.vpolys = []
        self.applied_margins = []
        for a, b in corridor:
            a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
            norm = np.linalg.norm(a, axis=1)
            self.hpolys.append((a / norm[:, None], b / norm))
        if not self.hpolys:
            raise ValueError("GCOPTER requires a nonempty corridor")
        self.poly_count = len(self.hpolys)
        radii, overlap_radii = [], []
        for i, (a, b) in enumerate(self.hpolys):
            center, radius = chebyshev_center(a, b)
            v = vertices(a, b, center)
            self.vpolys.append(np.vstack([v[0], v[1:] - v[0]]))
            radii.append(radius)
            if i < self.poly_count - 1:
                a1, b1 = self.hpolys[i + 1]
                ai, bi = np.vstack([a, a1]), np.r_[b, b1]
                center, radius = chebyshev_center(ai, bi)
                v = vertices(ai, bi, center)
                self.vpolys.append(np.vstack([v[0], v[1:] - v[0]]))
                overlap_radii.append(radius)
        self.applied_margins = np.minimum(max(0.0, cfg.safety_margin), np.maximum(0, 0.8 * np.array(radii)))
        for i, radius in enumerate(overlap_radii):
            cap = max(0.0, 0.8 * radius)
            self.applied_margins[i:i + 2] = np.minimum(self.applied_margins[i:i + 2], cap)
        self.penalty_polys = [(a, b - margin) for (a, b), margin in zip(
            self.hpolys, self.applied_margins)]
        self.start_state = self._clamp_boundary(start_state)
        self.end_state = self._clamp_boundary(end_state)
        self.short_path = _shortest_path(
            self.start_state[:, 0], self.end_state[:, 0], self.vpolys, cfg.smoothing_eps)
        if cfg.pieces_per_polytope > 0:
            self.piece_counts = np.full(self.poly_count, cfg.pieces_per_polytope, dtype=int)
        else:
            self.piece_counts = (np.linalg.norm(np.diff(self.short_path, axis=0), axis=1)
                                 / cfg.length_per_piece).astype(int) + 1
        self.piece_count = int(self.piece_counts.sum())
        self.hpoly_indices, self.vpoly_indices = [], []
        points, times = [], []
        speed = cfg.magnitude_bounds[0] * 3.0
        for i, count in enumerate(self.piece_counts):
            a, b = self.short_path[i:i + 2]
            step = (b - a) / count
            for j in range(count):
                self.hpoly_indices.append(i)
                times.append(np.linalg.norm(step) / speed)
                if i > 0 or j > 0:
                    points.append(a + j * step)
                if j < count - 1:
                    self.vpoly_indices.append(2 * i)
                elif i < self.poly_count - 1:
                    self.vpoly_indices.append(2 * i + 1)
        times = np.array(times)
        if np.any(times <= 0):
            raise GeometryError("corridor initialization contains a zero-length piece")
        self.point_polys = [self.vpolys[i] for i in self.vpoly_indices]
        self.initial_x = np.r_[inverse_t(times), _inverse_points(
            np.array(points).reshape(-1, 3), self.point_polys)]
        self.minco = MINCO(self.start_state, self.end_state, self.piece_count, cfg.cost_order)
        self.flatmap = FlatnessMap(cfg.physical_parameters)
        self.nodes = np.linspace(0.0, 1.0, cfg.integral_resolution + 1)
        self.quadrature = np.ones(cfg.integral_resolution + 1) / cfg.integral_resolution
        self.quadrature[[0, -1]] *= 0.5

    def _clamp_boundary(self, boundary):
        cfg = self.config
        state = np.asarray(boundary, dtype=np.float64).copy()
        if cfg.cost_order == 4 and state.shape == (3, 3):
            state = np.column_stack([state, np.zeros(3)])
        if state.shape != (3, cfg.cost_order):
            raise ValueError("boundary columns must be p,v,a[,jerk]")
        vmax, _, theta_max, thrust_min, thrust_max = cfg.magnitude_bounds
        mass, grav = cfg.physical_parameters[:2]
        norm = np.linalg.norm(state[:, 1])
        if norm > vmax:
            state[:, 1] *= vmax / norm
        f = state[:, 2].copy()
        f[2] += grav
        if f[2] <= 0.0:
            f[:] = [0.0, 0.0, thrust_min / mass]
        else:
            horizontal = np.linalg.norm(f[:2])
            maximum = f[2] * np.tan(theta_max)
            if theta_max < np.pi / 2 and horizontal > maximum:
                f[:2] *= maximum / horizontal
            norm = np.linalg.norm(f)
            if norm < thrust_min / mass:
                f *= thrust_min / (mass * norm)
            elif norm > thrust_max / mass:
                f *= thrust_max / (mass * norm)
        f[2] -= grav
        state[:, 2] = f
        return state

    def _penalty(self, times, coefficients):
        cfg = self.config
        local_t = times[:, None] * self.nodes[None]
        all_basis = derivative_bases(local_t, coefficients.shape[1] - 1, range(5))
        state = all_basis @ coefficients[:, None]
        basis = [all_basis[:, :, d] for d in range(5)]
        pos, vel, acc, jer, snap = [state[:, :, d] for d in range(5)]
        thrust, quat, omega = self.flatmap.forward(vel, acc, jer)
        gp, gv, go = np.zeros_like(pos), np.zeros_like(pos), np.zeros_like(pos)
        gq, gt = np.zeros_like(quat), np.zeros_like(thrust)
        penalty = np.zeros_like(thrust)
        wp, wv, wo, wtheta, wt = cfg.penalty_weights
        vmax, omax, theta_max, tmin, tmax = cfg.magnitude_bounds
        for i, hidx in enumerate(self.hpoly_indices):
            a, b = self.penalty_polys[hidx]
            value, deriv = smoothed_l1(pos[i] @ a.T - b, cfg.smoothing_eps)
            gp[i] = wp * deriv @ a
            penalty[i] += wp * np.sum(value, axis=1)
        value, deriv = smoothed_l1(np.sum(vel * vel, axis=-1) - vmax ** 2, cfg.smoothing_eps)
        penalty += wv * value
        gv += (2 * wv * deriv)[..., None] * vel
        value, deriv = smoothed_l1(np.sum(omega * omega, axis=-1) - omax ** 2, cfg.smoothing_eps)
        penalty += wo * value
        go += (2 * wo * deriv)[..., None] * omega
        cosine = 1 - 2 * np.sum(quat[..., 1:3] ** 2, axis=-1)
        value, deriv = smoothed_l1(np.arccos(cosine) - theta_max,
                                   cfg.smoothing_eps)
        penalty += wtheta * value
        active = deriv > 0
        theta_scale = np.zeros_like(deriv)
        theta_scale[active] = 4 * wtheta * deriv[active] / np.sqrt(1 - cosine[active] ** 2)
        gq[..., 1:3] = theta_scale[..., None] * quat[..., 1:3]
        mean, radius = (tmin + tmax) / 2, abs(tmax - tmin) / 2
        value, deriv = smoothed_l1((thrust - mean) ** 2 - radius ** 2, cfg.smoothing_eps)
        penalty += wt * value
        gt += 2 * wt * deriv * (thrust - mean)
        gp, gv, ga, gj, _, _ = self.flatmap.backward(gp, gv, gt, gq, go)
        weight = times[:, None] * self.quadrature[None]
        grad_c = sum(np.einsum("nkm,nkd,nk->nmd", beta, gradient, weight)
                     for beta, gradient in zip(basis[:4], [gp, gv, ga, gj]))
        gradient_t = np.sum((gp * vel + gv * acc + ga * jer + gj * snap), axis=-1)
        grad_t = np.sum(gradient_t * self.nodes[None] * weight
                        + penalty * self.quadrature[None], axis=1)
        return float(np.sum(penalty * weight)), grad_c, grad_t

    def cost_gradient(self, x):
        tau, xi = x[:self.piece_count], x[self.piece_count:]
        times = forward_t(tau)
        points = _forward_points(xi, self.point_polys)
        self.minco.set_parameters(points, times)
        cost, grad_c, grad_t = self.minco.energy_with_partials()
        penalty, pc, pt = self._penalty(times, self.minco.coefficients)
        grad_p, grad_t = self.minco.propagate_gradient(grad_c + pc, grad_t + pt)
        cost += penalty + self.config.time_weight * times.sum()
        grad_t += self.config.time_weight
        norm_cost, grad_xi = _backward_points(xi, self.point_polys, grad_p)
        return float(cost + norm_cost), np.r_[backward_grad_t(tau, grad_t), grad_xi]

    def optimize(self):
        started = perf_counter()
        result = minimize(self.cost_gradient, self.initial_x,
                          LBFGSParameters(mem_size=256, past=3, min_step=1e-32,
                                          g_epsilon=0, delta=self.config.relative_cost_tolerance,
                                          max_iterations=self.config.max_iterations))
        trajectory = None
        if result.success:
            times = forward_t(result.x[:self.piece_count])
            points = _forward_points(result.x[self.piece_count:], self.point_polys)
            self.minco.set_parameters(points, times)
            trajectory = self.minco.trajectory()
        self.last_result = result
        return OptimizationResult(trajectory, result.success, result.message, result.status,
                                  result.cost if result.success else float("inf"),
                                  result.iterations, result.evaluations, perf_counter() - started)
