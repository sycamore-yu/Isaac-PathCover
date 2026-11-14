"""Trajectory constraint diagnostics; these measurements do not select controls."""
import numpy as np
from numpy.polynomial import polynomial as poly

from .flatness import FlatnessMap
from .minco import derivative_basis


def _extrema(coefficients):
    """Scalar extrema on normalized time [0,1], using derivative roots."""
    coefficients = poly.polytrim(np.asarray(coefficients, dtype=float))
    derivative = poly.polyder(coefficients)
    roots = poly.polyroots(derivative) if len(derivative) > 1 else np.empty(0)
    real = roots.real[(np.abs(roots.imag) <= 1e-7) & (roots.real >= 0) & (roots.real <= 1)]
    values = poly.polyval(np.r_[0.0, real, 1.0], coefficients)
    return float(values.min()), float(values.max())


def _squared_norm_max(coefficients):
    squared = sum(np.convolve(coefficients[:, d], coefficients[:, d]) for d in range(3))
    return np.sqrt(max(0.0, _extrema(squared)[1]))


def trajectory_diagnostics(trajectory, corridor, config, *, hpoly_indices=None,
                           samples_per_piece=101):
    """Report exact polynomial extrema and sampled flatness constraints.

    Speed and acceleration maxima and signed corridor residuals use stationary
    roots in each piece. Body rates, tilt, and thrust use evenly spaced samples
    because the drag-aware flatness map is not polynomial.
    """
    if hpoly_indices is None:
        if len(corridor) != len(trajectory.durations):
            raise ValueError("hpoly_indices are required when pieces and polytopes differ")
        hpoly_indices = range(len(corridor))
    speed_max, acceleration_max, residuals = 0.0, 0.0, []
    velocities, accelerations, jerks, positions = [], [], [], []
    for c, duration, index in zip(trajectory.coefficients, trajectory.durations, hpoly_indices):
        degree = len(c) - 1
        # Derivative coefficients are expressed in u=t/T to condition root finding.
        velocity_coeffs = (np.arange(1, degree + 1)[:, None] * c[1:]
                           * duration ** np.arange(degree)[:, None])
        acceleration_coeffs = (np.arange(2, degree + 1)[:, None]
                               * np.arange(1, degree)[:, None] * c[2:]
                               * duration ** np.arange(degree - 1)[:, None])
        speed_max = max(speed_max, _squared_norm_max(velocity_coeffs))
        acceleration_max = max(acceleration_max, _squared_norm_max(acceleration_coeffs))
        position_coeffs = c * duration ** np.arange(degree + 1)[:, None]
        A, b = corridor[index]
        A, b = np.asarray(A), np.asarray(b)
        normals = np.linalg.norm(A, axis=1)
        plane_coefficients = position_coeffs @ (A / normals[:, None]).T
        plane_coefficients[0] -= b / normals
        residuals.append(max(_extrema(plane_coefficients[:, i])[1] for i in range(len(b))))
        local = np.linspace(0.0, duration, samples_per_piece)
        for values, d in [(positions, 0), (velocities, 1), (accelerations, 2), (jerks, 3)]:
            values.append(derivative_basis(local, degree, d) @ c)
    p, v, a, j = [np.concatenate(value) for value in (positions, velocities, accelerations, jerks)]
    thrust, quat, omega = FlatnessMap(config.physical_parameters).forward(v, a, j)
    cosine = np.clip(1 - 2 * np.sum(quat[:, 1:3] ** 2, axis=1), -1, 1)
    body_rate_max = float(np.max(np.linalg.norm(omega, axis=1)))
    tilt_max = float(np.max(np.arccos(cosine)))
    thrust_min, thrust_max = float(thrust.min()), float(thrust.max())
    vmax, wmax, tmax, fmin, fmax = config.magnitude_bounds
    return {
        "max_speed": float(speed_max),
        "max_acceleration": float(acceleration_max),
        "max_body_rate": body_rate_max,
        "max_tilt": tilt_max,
        "min_thrust": thrust_min,
        "max_thrust": thrust_max,
        "max_corridor_residual_m": max(residuals),
        "piece_corridor_residual_m": residuals,
        "speed_limit_residual": float(speed_max - vmax),
        "body_rate_limit_residual": body_rate_max - wmax,
        "tilt_limit_residual": tilt_max - tmax,
        "thrust_lower_limit_residual": fmin - thrust_min,
        "thrust_upper_limit_residual": thrust_max - fmax,
        "position_sample_min": p.min(axis=0).tolist(),
        "position_sample_max": p.max(axis=0).tolist(),
        "polynomial_extrema": "stationary roots in normalized piece time",
        "flatness_samples_per_piece": samples_per_piece,
    }
