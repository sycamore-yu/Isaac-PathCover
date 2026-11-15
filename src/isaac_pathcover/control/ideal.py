"""SUPER perfect_drone_sim command execution without tracking dynamics.

Flatness attitude follows SUPER 2ad3419 ros2_perfect_drone_model.hpp.
Copyright 2025 Yunfan REN, MaRS Lab, HKU; LGPL-3.0-or-later.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def ideal_state(sample):
    """Return p, quaternion wxyz, world velocity and zero body angular velocity."""
    z_body = sample.state[:, 2] + np.array([0., 0., 9.80])
    z_body /= np.linalg.norm(z_body)
    heading = np.array([np.cos(sample.yaw), np.sin(sample.yaw), 0.])
    y_body = np.cross(z_body, heading)
    y_body /= np.linalg.norm(y_body)
    x_body = np.cross(y_body, z_body)
    quaternion = Rotation.from_matrix(np.column_stack((x_body, y_body, z_body))).as_quat()
    return np.r_[sample.state[:, 0], quaternion[[3, 0, 1, 2]], sample.state[:, 1], np.zeros(3)]


class IdealController:
    name = "ideal"

    def __init__(self, env):
        self.control_dt = env.step_dt
        self.solve_times = []

    def step(self, env, state, sample, trajectory, now):
        env.step_ideal(ideal_state(sample))
        return np.zeros(4)

    def metadata(self):
        return {
            "name": self.name,
            "source": "SUPER 2ad3419 perfect_drone_sim updateFlatness",
            "attitude_gravity_m_s2": 9.80,
            "control_hz": 1. / self.control_dt,
            "execution": "direct last-published position, velocity and acceleration/yaw attitude",
            "angular_velocity": "zero, as in source odometry",
            "collision": "original body geometry against scene solids",
        }
