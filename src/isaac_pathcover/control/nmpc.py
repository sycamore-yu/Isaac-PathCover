"""Ten-state TurboMPC to CTBR, then the shared rate controller/Newton plant."""
from __future__ import annotations

from copy import copy
from time import perf_counter

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from .solvers import TurboMPC
from .solvers.model import DT, GRAVITY, LOWER, MASS, UPPER


def reference_attitude(acceleration, yaw, gravity):
    direction = np.asarray(acceleration, dtype=float) - np.array([0., 0., -gravity])
    z = direction / np.linalg.norm(direction)
    heading = np.array([np.cos(yaw), np.sin(yaw), 0.])
    y = np.cross(z, heading)
    y /= np.linalg.norm(y)
    return np.column_stack((np.cross(y, z), y, z))


class NMPCController:
    name = "nmpc"

    def __init__(self, env):
        self.solver = TurboMPC(horizon=16, max_iter=3)
        self.device = env.device
        self.period, self.next_solve = DT, 0.
        self.solve_times = []
        self.command = np.array([MASS*GRAVITY, 0., 0., 0.])
        self.ctbr_command = np.array([GRAVITY, 0., 0., 0.])
        self.controls = torch.as_tensor(self.command, device=self.device, dtype=torch.float64).expand(1, self.solver.horizon, 4).clone()
        weights = [80., 80., 100., 50., 50., 50., 30., 6., 6., 8., .015, .8, .8, .4]
        self.C = torch.diag(torch.tensor(weights, device=self.device, dtype=torch.float64)).expand(1, self.solver.horizon, 14, 14).clone()
        self.C[:, -1, :10, :10] *= 3.
        self.prewarm_seconds = 0.
        initial = env.physical_state()[:, :10].double()
        target = torch.cat((initial[:, None].expand(-1, self.solver.horizon, -1), self.controls), -1)
        c = -(self.C @ target.unsqueeze(-1)).squeeze(-1)
        torch.cuda.synchronize(self.device)
        started = perf_counter()
        self.solver.solve(initial, self.C, c, self.controls)
        torch.cuda.synchronize(self.device)
        self.prewarm_seconds = perf_counter() - started

    def _reference(self, trajectory, sample, now, quaternion):
        command = copy(trajectory) if trajectory is not None else None
        refs = [command.sample(now + k*DT) if command is not None else sample
                for k in range(self.solver.horizon)]
        rotations = [reference_attitude(ref.state[:, 2], ref.yaw, GRAVITY)
                     for ref in refs]
        quaternions = Rotation.from_matrix(np.asarray(rotations)).as_quat()[:, [3, 0, 1, 2]]
        previous = np.asarray(quaternion)
        for k in range(len(quaternions)):
            if np.dot(quaternions[k], previous) < 0:
                quaternions[k] *= -1
            previous = quaternions[k]
        angular_velocity = np.zeros((len(refs), 3))
        for k in range(len(refs)-1):
            angular_velocity[k] = Rotation.from_matrix(rotations[k].T @ rotations[k+1]).as_rotvec()/DT
        angular_velocity[-1] = angular_velocity[-2]
        targets = np.zeros((len(refs), 14))
        targets[:, :3] = [ref.state[:, 0] for ref in refs]
        targets[:, 3:7] = quaternions
        targets[:, 7:10] = [ref.state[:, 1] for ref in refs]
        targets[:, 11:14] = angular_velocity
        gravity = np.array([0., 0., -GRAVITY])
        targets[:, 10] = [MASS*np.linalg.norm(ref.state[:, 2]-gravity) for ref in refs]
        return torch.as_tensor(targets, device=self.device, dtype=torch.float64)[None]

    def step(self, env, state, sample, trajectory, now):
        if now + 1e-9 >= self.next_solve:
            x0 = torch.as_tensor(state[:10], device=self.device, dtype=torch.float64)[None]
            target = self._reference(trajectory, sample, now, state[3:7])
            c = -(self.C @ target.unsqueeze(-1)).squeeze(-1)
            torch.cuda.synchronize(self.device)
            started = perf_counter()
            _, controls, _ = self.solver.solve(x0, self.C, c, self.controls)
            torch.cuda.synchronize(self.device)
            self.solve_times.append(perf_counter() - started)
            self.command = controls[0, 0].cpu().numpy()
            self.ctbr_command = self.command.copy()
            # The source solver uses N; the execution interface uses m/s².
            self.ctbr_command[0] /= MASS
            self.controls = torch.cat((controls[:, 1:], controls[:, -1:]), 1)
            self.next_solve = now + self.period
        env.step_ctbr(torch.as_tensor(self.ctbr_command, dtype=env.command.dtype, device=env.device)[None])
        return env.actuator.requested_wrench[0].cpu().numpy().copy()

    def metadata(self):
        return {"name": self.name, "control_hz": 1./self.period, "prediction_dt_s": DT,
                "horizon_steps": self.solver.horizon, "horizon_s": self.solver.horizon*DT,
                "sqp_max_iter": self.solver.max_iter,
                "admm_max_iter": self.solver.config.max_iter, "admm_eps_abs": self.solver.config.eps_abs,
                "admm_eps_rel": self.solver.config.eps_rel, "state_dim": 10, "control_dim": 4,
                "state_order": "position, quaternion_wxyz, world_velocity",
                "input": "collective_thrust_acceleration_mps2, body_rate_radps",
                "solver_input": "total_thrust_N, body_rate_radps", "mass_kg": MASS,
                "ctbr_lower": [LOWER[0]/MASS, *LOWER[1:]], "ctbr_upper": [UPPER[0]/MASS, *UPPER[1:]],
                "prewarm_seconds": self.prewarm_seconds,
                "source_commit": "e36743e8076bfe12b3e11911e2f1f866d61ae158",
                "solver": "Isaac-L2F Python SQP/ADMM/nvmath-cuDSS"}

    def close(self):
        self.solver.close()
