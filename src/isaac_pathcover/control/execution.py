"""Trajectory-server replacement and failure semantics in simulation time."""
from __future__ import annotations

import numpy as np


def execution_check(diagnostics, config):
    """Check a solved candidate before handing it to the Newton controller.

    This execution policy is additional to upstream GCOPTER. Its soft-residual
    allowance is the existing penalty transition width, not an upstream
    feasibility guarantee. Position uses the original corridor without the
    soft safety-margin penalty. Attitude quantities remain sampled checks.
    """
    vmax, wmax, tiltmax, fmin, fmax = config.magnitude_bounds
    midpoint, radius = (fmin + fmax) / 2, (fmax - fmin) / 2
    residuals = {
        "speed_squared": diagnostics["max_speed"] ** 2 - vmax ** 2,
        "body_rate_squared": diagnostics["max_body_rate"] ** 2 - wmax ** 2,
        "tilt": diagnostics["max_tilt"] - tiltmax,
        "thrust_squared": np.maximum((diagnostics["min_thrust"] - midpoint) ** 2,
                                     (diagnostics["max_thrust"] - midpoint) ** 2) - radius ** 2,
    }
    violations = [key for key, value in residuals.items()
                  if not np.isfinite(value) or value > config.smoothing_eps]
    corridor_residual = diagnostics["max_corridor_residual_m"]
    if not np.isfinite(corridor_residual) or corridor_residual > 1e-6:
        violations.append("corridor")
    return {"accepted": not violations, "violations": violations,
            "soft_residual_allowance": config.smoothing_eps,
            "corridor_roundoff_allowance_m": 1e-6,
            "residuals": {key: float(value) if np.isfinite(value) else None
                          for key, value in residuals.items()},
            "flatness_samples_per_piece": diagnostics["flatness_samples_per_piece"]}


class TrajectoryExecution:
    def __init__(self, initial_position):
        self.trajectory = None
        self.start_time = 0.
        self.hover_position = np.asarray(initial_position).copy()

    def reference(self, now):
        if self.trajectory is None:
            return np.column_stack((self.hover_position, np.zeros((3, 3))))
        elapsed = np.clip(now - self.start_time, 0., self.trajectory.duration)
        return np.column_stack([self.trajectory.evaluate(elapsed, derivative=k) for k in range(4)])

    def replace(self, result, now, measured_state, *, validation):
        """Keep an active reference on failure; start successful plans at now."""
        previous = self.reference(now)
        if not result.success:
            return {"accepted": False, "retained_previous": self.trajectory is not None}
        if not validation["accepted"]:
            return {"accepted": False, "retained_previous": self.trajectory is not None,
                    "rejection_reason": validation["violations"]}
        self.trajectory = result.trajectory
        self.start_time = now
        new = self.reference(now)
        return {
            "accepted": True,
            "initial_state_error": np.linalg.norm(new[:, :3] - measured_state, axis=0).tolist(),
            "reference_jump": np.linalg.norm(new[:, :3] - previous[:, :3], axis=0).tolist(),
        }
