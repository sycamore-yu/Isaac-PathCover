"""MINCO S3NU/S4NU with the original band system and analytic adjoint.

Source: PathCover 8723a05d, trajectory_server/include/gcopter/minco.hpp.
Copyright (c) 2021 Zhepei Wang. Distributed under the upstream MIT license.
"""
from dataclasses import dataclass
from functools import lru_cache
from math import factorial
import numpy as np
from scipy.linalg.lapack import dgbtrf, dgbtrs


@lru_cache(maxsize=32)
def _basis_table(degree, derivatives):
    factors = np.zeros((len(derivatives), degree + 1))
    powers = np.zeros_like(factors, dtype=int)
    for row, d in enumerate(derivatives):
        for k in range(d, degree + 1):
            factors[row, k] = factorial(k) / factorial(k - d)
            powers[row, k] = k - d
    return factors, powers


def derivative_bases(t, degree, derivatives):
    factors, powers = _basis_table(degree, tuple(derivatives))
    return np.asarray(t)[..., None, None] ** powers * factors


def derivative_basis(t, degree, derivative=0):
    return derivative_bases(t, degree, (derivative,))[..., 0, :]


@dataclass
class PiecewisePolynomial:
    durations: np.ndarray
    coefficients: np.ndarray

    def __post_init__(self):
        self.durations = np.asarray(self.durations, dtype=np.float64).copy()
        self.coefficients = np.asarray(self.coefficients, dtype=np.float64).copy()
        self.knots = np.concatenate(([0.0], np.cumsum(self.durations)))

    @property
    def duration(self):
        return float(self.knots[-1])

    def evaluate(self, t, derivative=0):
        t = np.clip(np.asarray(t, dtype=np.float64), 0.0, self.duration)
        index = np.minimum(np.searchsorted(self.knots[1:], t, side="left"), len(self.durations) - 1)
        local = t - self.knots[index]
        basis = derivative_basis(local, self.coefficients.shape[1] - 1, derivative)
        return np.einsum("...k,...kd->...d", basis, self.coefficients[index])


class MINCO:
    """Minimum integrated squared jerk (3) or snap (4), in three dimensions.

    Boundary columns are position, velocity, acceleration, and (for S4) jerk.
    Inner points have shape (N-1, 3); polynomial coefficients are ascending.
    """
    def __init__(self, start_state, end_state, pieces, order=3):
        if order not in (3, 4) or pieces < 1:
            raise ValueError("MINCO needs order 3/4 and at least one piece")
        self.order, self.pieces, self.width = order, pieces, 2 * order
        self.start = np.asarray(start_state, dtype=np.float64).copy()
        self.end = np.asarray(end_state, dtype=np.float64).copy()
        if self.start.shape != (3, order) or self.end.shape != (3, order):
            raise ValueError("boundary shape must be (3, order)")
        m, n, s = self.width, pieces, order
        rows, cols, ti, powers, factors = [], [], [], [], []
        fixed_rows, fixed_cols, fixed_values = [], [], []
        self.endpoint_rows = []
        self.endpoint_derivatives = list(range(s, m - 1)) + [0] + list(range(s))
        for k in range(s):
            fixed_rows.append(k)
            fixed_cols.append(k)
            fixed_values.append(float(factorial(k)))
        for i in range(n):
            ders = self.endpoint_derivatives if i < n - 1 else list(range(s))
            first_row = m * i + s
            erows = []
            for r, d in enumerate(ders):
                row = first_row + r
                erows.append(row)
                for k in range(d, m):
                    rows.append(row)
                    cols.append(m * i + k)
                    ti.append(i)
                    powers.append(k - d)
                    factors.append(factorial(k) / factorial(k - d))
                if i < n - 1 and r != s - 1:
                    fixed_rows.append(row)
                    fixed_cols.append(m * (i + 1) + d)
                    fixed_values.append(-float(factorial(d)))
            self.endpoint_rows.append(np.array(erows))
        self._rows, self._cols = np.array(rows), np.array(cols)
        self._ti, self._powers = np.array(ti), np.array(powers)
        self._factors = np.array(factors)
        self._fixed_rows, self._fixed_cols = np.array(fixed_rows), np.array(fixed_cols)
        self._fixed_values = np.array(fixed_values)
        self._waypoint_rows = np.arange(n - 1) * m + m - 1
        self._rhs = np.zeros((m * n, 3))
        self._rhs[:s] = self.start.T
        self._rhs[-s:] = self.end.T
        ks = np.arange(s, m)
        self._energy_factors = np.array([factorial(k) / factorial(k - s) for k in ks])
        self._energy_powers = ks[:, None] + ks[None, :] - 2 * s + 1

    def set_parameters(self, inner_points, durations):
        self.durations = np.asarray(durations, dtype=np.float64).copy()
        if self.durations.shape != (self.pieces,) or np.any(self.durations <= 0):
            raise ValueError("piece durations must be positive")
        points = np.asarray(inner_points, dtype=np.float64).reshape(self.pieces - 1, 3)
        m, n = self.width, self.pieces
        # LAPACK general band layout includes m extra rows for pivot fill-in.
        band = np.zeros((3 * m + 1, m * n), order="F")
        band[2 * m + self._rows - self._cols, self._cols] = (
            self._factors * self.durations[self._ti] ** self._powers)
        band[2 * m + self._fixed_rows - self._fixed_cols, self._fixed_cols] = self._fixed_values
        self._lu, self._pivots, info = dgbtrf(band, m, m, overwrite_ab=1)
        if info:
            raise np.linalg.LinAlgError(f"MINCO band factorization failed: {info}")
        rhs = self._rhs.copy()
        rhs[self._waypoint_rows] = points
        coeffs, info = dgbtrs(self._lu, m, m, rhs, self._pivots)
        if info:
            raise np.linalg.LinAlgError(f"MINCO band solve failed: {info}")
        self.coefficients = coeffs.reshape(n, m, 3)
        return self

    def energy_with_partials(self):
        s = self.order
        factors = self._energy_factors
        powers = self._energy_powers
        q = (factors[:, None] * factors[None, :])[None] * (
            self.durations[:, None, None] ** powers[None]) / powers[None]
        c = self.coefficients[:, s:]
        qc = np.einsum("nij,njd->nid", q, c)
        energy = float(np.sum(c * qc))
        grad_c = np.zeros_like(self.coefficients)
        grad_c[:, s:] = 2 * qc
        end_derivative = np.einsum(
            "nk,nkd->nd", derivative_basis(self.durations, self.width - 1, s),
            self.coefficients)
        grad_t = np.sum(end_derivative * end_derivative, axis=1)
        return energy, grad_c, grad_t

    def propagate_gradient(self, grad_coefficients, grad_durations):
        m, n, s = self.width, self.pieces, self.order
        adjoint, info = dgbtrs(
            self._lu, m, m, np.asarray(grad_coefficients).reshape(m * n, 3),
            self._pivots, trans=1)
        if info:
            raise np.linalg.LinAlgError(f"MINCO adjoint solve failed: {info}")
        grad_points = adjoint[self._waypoint_rows].copy()
        grad_time = np.asarray(grad_durations).copy()
        endpoint_basis = derivative_bases(self.durations, m - 1, range(1, m))
        endpoint_states = endpoint_basis @ self.coefficients
        for i in range(n):
            ders = self.endpoint_derivatives if i < n - 1 else list(range(s))
            dstate = endpoint_states[i, ders]
            grad_time[i] -= np.sum(adjoint[self.endpoint_rows[i]] * dstate)
        return grad_points, grad_time

    def trajectory(self):
        return PiecewisePolynomial(self.durations, self.coefficients)
