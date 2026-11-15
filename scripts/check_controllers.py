"""Check direct Ideal execution or NMPC hover in the existing Newton scene."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from isaac_pathcover.control.execution import CommandSample
from isaac_pathcover.control.ideal import IdealController, ideal_state
from isaac_pathcover.simulation.environment import FlightEnv, make_env_cfg
from isaac_pathcover.simulation.metrics import body_collisions
from isaac_pathcover.simulation.scene import Scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", choices=("ideal", "nmpc"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    scene = Scene("medium")
    env = FlightEnv(make_env_cfg(scene))
    try:
        env.reset()
        if args.controller == "ideal":
            controller = IdealController(env)
            sample = CommandSample(np.column_stack((scene.start + [.15, .2, .1],
                                   [.5, .2, -.1], [.8, -.4, .3], [.1, .2, .3])),
                                   .6, .2, False, False)
            controller.step(env, env.physical_state()[0].cpu().numpy(), sample, None, 0.)
            state = env.physical_state()[0].cpu().numpy()
            expected = ideal_state(sample).astype(np.float32)
            np.testing.assert_allclose(state, expected, atol=2e-6, rtol=0)
            samples = torch.stack(env.substep_states).cpu().numpy()
            np.testing.assert_allclose(samples, np.broadcast_to(expected, samples.shape), atol=2e-6, rtol=0)
            assert not body_collisions(scene, state[None]).any()
            overlap = CommandSample(np.column_stack(((scene.box_lower[0]+scene.box_upper[0])/2,
                                                      np.zeros((3, 3)))), 0., 0., False, False)
            controller.step(env, state, overlap, None, env.step_dt)
            contact_state = env.physical_state()[0].cpu().numpy()
            assert body_collisions(scene, contact_state[None]).all()
            report = {"controller": "ideal", "direct_state_max_error": float(np.max(np.abs(state-expected))),
                      "free_pose_collision": False, "occupied_pose_collision": True,
                      "physics_substeps": len(samples), "passed": True}
            np.savez_compressed(args.output/"states.npz", state=state, expected=expected,
                                substep_states=samples, collision_state=contact_state)
        else:
            from isaac_pathcover.control.nmpc import NMPCController

            pose = torch.zeros((1, 7), device=env.device)
            pose[0, :3] = torch.as_tensor(scene.start + [.1, -.1, .05], device=env.device)
            pose[0, 3:] = torch.as_tensor(Rotation.from_euler("x", .05).as_quat(), device=env.device)
            env.robot.write_root_com_pose_to_sim_index(root_pose=pose)
            controller = NMPCController(env)
            sample = CommandSample(np.column_stack((scene.start, np.zeros((3, 3)))), 0., 0., False, False)
            states, thrusts, requested, allocated = [], [], [], []
            try:
                for tick in range(round(5./env.step_dt)):
                    state = env.physical_state()[0].cpu().numpy()
                    controller.step(env, state, sample, None, tick*env.step_dt)
                    states.append(env.physical_state()[0].cpu().numpy().copy())
                    thrusts.append(env.actuator.allocated_thrust[0].cpu().numpy().copy())
                    requested.append(env.actuator.requested_wrench[0].cpu().numpy().copy())
                    allocated.append(env.actuator.allocated_wrench[0].cpu().numpy().copy())
            finally:
                controller.close()
            states, thrusts = np.asarray(states), np.asarray(thrusts)
            error = np.linalg.norm(states[:, :3]-scene.start, axis=1)
            tail = states[-round(1./env.step_dt):]
            lower, upper = env.actuator.lower.cpu().numpy(), env.actuator.upper.cpu().numpy()
            requested_thrusts = np.asarray(requested) @ env.actuator.allocation_inverse[0].cpu().numpy().T
            bounded = bool(np.all(thrusts >= lower-1e-5) and np.all(thrusts <= upper+1e-5))
            stable = bool(np.isfinite(states).all() and np.max(np.linalg.norm(tail[:, :3]-scene.start, axis=1)) < .05
                          and np.max(np.linalg.norm(tail[:, 7:10], axis=1)) < .1)
            report = {"controller": "nmpc", "duration_s": 5., "initial_position_offset_m": [.1, -.1, .05],
                      "initial_roll_rad": .05, "tracking_max_m": float(error.max()),
                      "final_position_error_m": float(error[-1]), "final_speed_m_s": float(np.linalg.norm(states[-1, 7:10])),
                      "rotor_thrust_min_N": float(thrusts.min()), "rotor_thrust_max_N": float(thrusts.max()),
                      "requested_rotor_thrust_min_N": float(requested_thrusts.min()),
                      "requested_rotor_thrust_max_N": float(requested_thrusts.max()),
                      "rotor_limits_satisfied": bounded, "hover_stable": stable,
                      "nmpc_solve_p50_s": float(np.median(controller.solve_times)),
                      "nmpc_solve_p95_s": float(np.quantile(controller.solve_times, .95)),
                      "controller_config": controller.metadata(), "passed": stable and bounded}
            np.savez_compressed(args.output/"hover.npz", states=states, rotor_thrusts=thrusts,
                                requested_wrench=requested, allocated_wrench=allocated,
                                nmpc_solve_time_s=controller.solve_times)
        (args.output/"result.json").write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(report), flush=True)
        assert report["passed"], report
    finally:
        env.close()


if __name__ == "__main__":
    main()
