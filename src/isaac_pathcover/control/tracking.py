"""Existing Newton geometric controller, driven by SUPER position and yaw."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


class GeometricTracker:
    """Port of the active GeometricController::Controller overload.

    The upstream quadrotor plugin calls the p/v/a overload, with heading-rate
    feedforward. Motor allocation and motor lag belong to the L2F plant.
    """
    def __init__(self, mass=.752, gravity=9.8066, inertia=(.0025, .0021, .0043),
                 kp=(14.5, 14.5, 14.5), kd=(5.8, 5.8, 5.8),
                 kr=(2.2, 2.2, 1.7), kw=(.21, .21, .25)):
        self.mass, self.gravity = mass, gravity
        self.inertia = np.diag(inertia)
        self.kp, self.kd, self.kr, self.kw = map(np.asarray, (kp, kd, kr, kw))

    def command(self, state, position, velocity, acceleration, *, yaw, yaw_rate):
        """State is p(3), quaternion wxyz(4), world v(3), body omega(3)."""
        rotation = Rotation.from_quat(np.asarray(state)[[4, 5, 6, 3]]).as_matrix()
        force = -self.kp * (state[:3] - position) - self.kd * (state[7:10] - velocity)
        force = force + self.mass * (acceleration + np.array([0., 0., self.gravity]))
        thrust = force @ rotation[:, 2]
        z = force / np.linalg.norm(force)
        x_heading = np.array([np.cos(yaw), np.sin(yaw), 0.])
        y = np.cross(z, x_heading)
        y /= np.linalg.norm(y)
        x = np.cross(y, z)
        x /= np.linalg.norm(x)
        desired = np.column_stack((x, y, z))
        skew_error = .5 * (desired.T @ rotation - rotation.T @ desired)
        error_r = skew_error[[2, 0, 1], [1, 2, 0]]
        omega = state[10:13]
        error_w = omega - rotation.T @ desired @ np.array([0., 0., yaw_rate])
        torque = -self.kr * error_r - self.kw * error_w + np.cross(omega, self.inertia @ omega)
        return np.r_[thrust, torque], desired
