"""Replay frozen SUPER commands with 100 or 400 Hz Newton wrench updates.

The reference remains at its recorded 100 Hz zero-order hold. Both experiments
call the existing GeometricTracker and use the unchanged FlightEnv plant.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from isaac_pathcover.control.tracking import GeometricTracker
from isaac_pathcover.simulation.environment import FlightEnv, make_env_cfg
from isaac_pathcover.simulation.metrics import body_collisions
from isaac_pathcover.simulation.scene import Scene


class ReplayFlightEnv(FlightEnv):
    """Change only when the existing tracker refreshes the allocated wrench."""

    def _apply_action(self):
        if self.control_hz == 400:
            state = self.physical_state()[0].detach().cpu().numpy().astype(float)
            self.held_wrench, self.held_desired = self.tracker.command(
                state, self.reference[:, 0], self.reference[:, 1], self.reference[:, 2],
                yaw=self.yaw_reference[0], yaw_rate=self.yaw_reference[1])
            self.held_force = feedback_force(self.tracker, state, self.reference)
            self.command.copy_(torch.as_tensor(self.held_wrench[None], dtype=torch.float32,
                                               device=self.device))
            self.actuator.command(self.command)
        self.physics_wrench.append(self.held_wrench.copy())
        self.physics_desired.append(self.held_desired.copy())
        self.physics_force.append(self.held_force.copy())
        self.physics_allocated.append(self.actuator.allocated_wrench[0].detach().cpu().numpy().copy())
        self.physics_motor_command.append(self.actuator.motor_command[0].detach().cpu().numpy().copy())
        super()._apply_action()


def feedback_force(tracker, state, reference):
    """Diagnostic copy of the force already used by GeometricTracker.command."""
    force = -tracker.kp * (state[:3] - reference[:, 0])
    force -= tracker.kd * (state[7:10] - reference[:, 1])
    force += tracker.mass * (reference[:, 2] + np.array([0., 0., tracker.gravity]))
    return force


def first_time(mask, times):
    indices = np.flatnonzero(mask)
    return float(times[indices[0]]) if len(indices) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--rate", type=int, choices=(100, 400), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = np.load(args.input_dir / "input_flight.npz")
    original_report = json.loads((args.input_dir / "input_result.json").read_text())
    scene = Scene(original_report["scene"])
    tracker = GeometricTracker()
    torch.set_num_threads(1)
    started = time.perf_counter()
    env = ReplayFlightEnv(make_env_cfg(scene, device=args.device, seed=original_report["seed"]))
    startup = time.perf_counter() - started
    env.control_hz = args.rate
    env.tracker = tracker
    env.physics_wrench, env.physics_desired, env.physics_force = [], [], []
    env.physics_allocated, env.physics_motor_command = [], []
    states, substeps, contacts, motors, requested, allocated = [], [], [], [], [], []
    started = time.perf_counter()
    try:
        env.reset()
        for tick, (reference, yaw_reference) in enumerate(zip(source["references"], source["yaw_references"])):
            state = env.physical_state()[0].detach().cpu().numpy().astype(float)
            env.reference, env.yaw_reference = reference, yaw_reference
            env.held_wrench, env.held_desired = tracker.command(
                state, reference[:, 0], reference[:, 1], reference[:, 2],
                yaw=yaw_reference[0], yaw_rate=yaw_reference[1])
            env.held_force = feedback_force(tracker, state, reference)
            requested.append(env.held_wrench.copy())
            env.step(torch.as_tensor(env.held_wrench[None], dtype=torch.float32, device=env.device))
            after = env.physical_state()[0].detach().cpu().numpy().astype(float)
            states.append(after)
            substeps.extend(torch.stack(env.substep_states).detach().cpu().numpy())
            contacts.append(float(env.contact_force()))
            motors.append(env.actuator.motor_state[0].detach().cpu().numpy().copy())
            allocated.append(env.actuator.allocated_wrench[0].detach().cpu().numpy().copy())
            if (tick + 1) % 500 == 0:
                print(json.dumps({"rate_hz": args.rate, "time_s": (tick + 1) * env.step_dt,
                                  "tracking_error_m": float(np.linalg.norm(after[:3] - reference[:, 0])),
                                  "contact_force_N": contacts[-1]}), flush=True)
        wall = time.perf_counter() - started
        physics_dt = float(env.physics_dt)
        arrays = dict(time=source["time"], states=np.asarray(states),
                      references=source["references"], yaw_references=source["yaw_references"],
                      command_time=source["command_time"], physics_states=np.asarray(substeps),
                      contact_force=np.asarray(contacts), motor_states=np.asarray(motors),
                      requested_wrench=np.asarray(requested), allocated_wrench=np.asarray(allocated),
                      physics_requested_wrench=np.asarray(env.physics_wrench),
                      physics_allocated_wrench=np.asarray(env.physics_allocated),
                      physics_motor_command=np.asarray(env.physics_motor_command),
                      physics_desired_rotation=np.asarray(env.physics_desired),
                      physics_feedback_force=np.asarray(env.physics_force))
    finally:
        env.close()

    physics_times = np.arange(len(arrays["physics_states"])) * physics_dt
    physics_states = arrays["physics_states"]
    rotations = Rotation.from_quat(physics_states[:, [4, 5, 6, 3]]).as_matrix()
    desired = arrays["physics_desired_rotation"]
    attitude_error = Rotation.from_matrix(np.swapaxes(desired, 1, 2) @ rotations).magnitude()
    desired_increment = Rotation.from_matrix(np.swapaxes(desired[:-1], 1, 2) @ desired[1:]).magnitude()
    tracking_error = np.linalg.norm(arrays["states"][:, :3] - arrays["references"][:, :, 0], axis=1)
    collision_states = np.concatenate((physics_states, arrays["states"][-1:]))
    collision_times = np.r_[physics_times, arrays["time"][-1]]
    collision = body_collisions(scene, collision_states)
    negative_force = arrays["physics_feedback_force"][:, 2] < 0.
    negative_thrust = arrays["physics_requested_wrench"][:, 0] < 0.
    source_delta = arrays["states"] - source["states"]
    source_collision = body_collisions(scene, source["physics_states"])
    source_first_collision = first_time(source_collision, physics_times)
    precontact = arrays["time"] < source_first_collision
    first_collision = first_time(collision, collision_times)
    before_collision = arrays["time"] < (first_collision if first_collision is not None else np.inf)
    arrays.update(physics_time=physics_times, physics_attitude_error=attitude_error,
                  physics_desired_increment=np.r_[0., desired_increment], tracking_error=tracking_error,
                  collision_time=collision_times, body_collision=collision)
    np.savez_compressed(args.output / "replay.npz", **arrays)
    np.savetxt(args.output / "tracking.csv", np.column_stack((arrays["time"], tracking_error,
               np.linalg.norm(arrays["states"][:, 7:10], axis=1),
               np.linalg.norm(arrays["references"][:, :, 1], axis=1), arrays["contact_force"])),
               delimiter=",", header="time_s,tracking_error_m,actual_speed_m_s,reference_speed_m_s,contact_force_N",
               comments="")
    np.savetxt(args.output / "attitude.csv", np.column_stack((physics_times, attitude_error,
               np.r_[0., desired_increment], arrays["physics_feedback_force"][:, 2],
               arrays["physics_requested_wrench"][:, 0],
               np.linalg.norm(physics_states[:, 10:13], axis=1), collision[:-1])), delimiter=",",
               header="time_s,attitude_error_rad,desired_increment_rad,feedback_force_z_N,requested_thrust_N,body_rate_rad_s,body_collision",
               comments="")
    root = Path(__file__).resolve().parents[3]
    source_files = ["src/isaac_pathcover/control/tracking.py", "src/isaac_pathcover/control/actuator.py",
                    "src/isaac_pathcover/simulation/environment.py",
                    "src/isaac_pathcover/simulation/scene.py", str(Path(__file__).resolve().relative_to(root))]
    source_files += [str(path.relative_to(root)) for path in sorted((root / "src/isaac_pathcover/dynamics").glob("*.py"))]
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in source_files}
    summary = {
        "scene": scene.name, "seed": original_report["seed"], "controller_hz": args.rate,
        "reference_hz": 100, "physics_hz": round(1. / physics_dt),
        "duration_sim_s": float(arrays["time"][-1]), "startup_s": startup, "wall_time_s": wall,
        "input_sha256": hashlib.sha256((args.input_dir / "input_flight.npz").read_bytes()).hexdigest(),
        "source_hashes": hashes,
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "torch", "isaaclab", "newton", "warp-lang")},
        "gpu": torch.cuda.get_device_name(), "threads": torch.get_num_threads(),
        "finite": bool(all(np.isfinite(v).all() for v in arrays.values())),
        "first_body_collision_s": first_collision, "body_collision_samples": int(collision.sum()),
        "first_contact_force_s": first_time(arrays["contact_force"] > 1e-6, arrays["time"]),
        "max_contact_force_N": float(arrays["contact_force"].max()),
        "first_negative_force_z_s": first_time(negative_force, physics_times),
        "negative_force_z_duration_s": float(negative_force.sum() * physics_dt),
        "min_force_z_N": float(arrays["physics_feedback_force"][:, 2].min()),
        "first_negative_requested_thrust_s": first_time(negative_thrust, physics_times),
        "negative_requested_thrust_duration_s": float(negative_thrust.sum() * physics_dt),
        "min_requested_thrust_N": float(arrays["physics_requested_wrench"][:, 0].min()),
        "max_attitude_error_rad": float(attitude_error.max()),
        "max_desired_attitude_increment_rad": float(desired_increment.max()),
        "max_body_rate_rad_s": float(np.linalg.norm(physics_states[:, 10:13], axis=1).max()),
        "tracking_rmse_m": float(np.sqrt(np.mean(tracking_error ** 2))),
        "tracking_max_m": float(tracking_error.max()),
        "pre_collision_tracking_rmse_m": float(np.sqrt(np.mean(tracking_error[before_collision] ** 2))),
        "common_pre_collision_tracking_rmse_m": float(np.sqrt(np.mean(tracking_error[precontact] ** 2))),
        "final_goal_error_m": float(np.linalg.norm(arrays["states"][-1, :3] - scene.goal)),
        "source_replay_max_state_error": np.max(np.abs(source_delta), axis=0).tolist(),
        "source_replay_pre_contact_max_state_error": np.max(np.abs(source_delta[precontact]), axis=0).tolist(),
        "source_replay_max_requested_wrench_error": np.max(np.abs(arrays["requested_wrench"] - source["requested_wrench"]), axis=0).tolist(),
        "comparison_window_s": source_first_collision,
    }
    (args.output / "result.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print("RESULT " + json.dumps(summary, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
