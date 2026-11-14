"""Run the complete point-cloud-to-Newton PathCover pipeline."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from isaac_pathcover.control.execution import TrajectoryExecution, execution_check
from isaac_pathcover.control.tracking import GeometricTracker
from isaac_pathcover.planning.pipeline import Planner
from isaac_pathcover.simulation.scene import Scene


def finite_json(value):
    """Represent invalid diagnostic scalars as null; raw arrays remain in NPZ."""
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def save_plan(path, result):
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path / "trajectory.npz", coefficients=result.trajectory.coefficients,
                        durations=result.trajectory.durations, search_path=result.search.path)
    corridor = {"polytopes": [{"A": a.tolist(), "b": b.tolist()} for a, b in result.corridor.polytopes],
                "seeds": result.corridor.seeds.tolist(), "goal": result.corridor.goal.tolist()}
    (path / "corridor.json").write_text(json.dumps(corridor, indent=2) + "\n")


def run_episode(env, scene, seed, duration, output, update_map=True):
    import torch
    from isaac_pathcover.simulation.metrics import summarize
    from isaac_pathcover.simulation.visualization import save_preview
    from isaac_pathcover.trajectory.diagnostics import trajectory_diagnostics

    output.mkdir(parents=True, exist_ok=True)
    env.reset()
    tracker = GeometricTracker()
    planner = Planner(scene, seed=seed)
    execution = TrajectoryExecution(scene.start)
    times, states, references, motor_states, contacts, substeps = [], [], [], [], [], []
    commands, allocated = [], []
    plans, map_updates = [], []
    last_velocity = np.zeros(3)
    acceleration = np.zeros(3)
    initial_result = None
    next_replan = 0.
    started = time.perf_counter()
    dt = env.step_dt
    arrived_ticks = 0
    for tick in range(int(round(duration / dt))):
        now = tick * dt
        state = env.physical_state()[0].detach().cpu().numpy().astype(float)
        if tick:
            acceleration = (state[7:10] - last_velocity) / dt
        last_velocity = state[7:10].copy()
        if now + 1e-9 >= next_replan and not arrived_ticks:
            cycle_started = time.perf_counter()
            if update_map and tick:
                scan_started = time.perf_counter()
                sensor_origin, endpoints = scene.lidar(state[:3], state[3:7])
                scan_elapsed = time.perf_counter() - scan_started
                update = planner.update_map(sensor_origin, endpoints)
                map_updates.append({"t": now, "points": len(endpoints), "sensor_s": scan_elapsed,
                                    "sensor_origin": sensor_origin.tolist(), **update})
            measured = np.column_stack((state[:3], state[7:10], acceleration))
            result = planner.plan(measured)
            diagnostics = validation = None
            if result.success:
                diagnostics = trajectory_diagnostics(result.trajectory, result.corridor.polytopes, planner.config)
                validation = execution_check(diagnostics, planner.config)
            handoff = execution.replace(result, now, measured, validation=validation)
            record = {"t": now, "success": result.success, "status": result.status,
                      "timings": result.timings, "handoff": handoff,
                      "cycle_wall_s": time.perf_counter() - cycle_started,
                      "execution_check": validation}
            if result.optimization is not None:
                record.update({"iterations": result.optimization.iterations,
                               "evaluations": result.optimization.evaluations,
                               "cost": result.optimization.cost if np.isfinite(result.optimization.cost) else None})
            if result.success:
                record["trajectory_diagnostics"] = finite_json(diagnostics)
                if initial_result is None and handoff["accepted"]:
                    initial_result = result
                save_plan(output / "plans" / f"{len(plans):04d}", result)
            plans.append(record)
            print(json.dumps({"scene": scene.name, "seed": seed, **record}), flush=True)
            next_replan = now + .5
        reference = execution.reference(now)
        wrench, _ = tracker.command(state, reference[:, 0], reference[:, 1], reference[:, 2], dt)
        env.step(torch.as_tensor(wrench[None], dtype=torch.float32, device=env.device))
        after = env.physical_state()[0].detach().cpu().numpy().astype(float)
        times.append(now + dt)
        states.append(after)
        references.append(execution.reference(now + dt))
        contacts.append(float(env.contact_force()))
        motor_states.append(env.actuator.motor_state[0].detach().cpu().numpy().copy())
        commands.append(wrench)
        allocated.append(env.actuator.allocated_wrench[0].detach().cpu().numpy().copy())
        substeps.extend(torch.stack(env.substep_states).detach().cpu().numpy())
        if not np.isfinite(after).all():
            break
        distance = np.linalg.norm(after[:3] - scene.goal)
        arrived_ticks = arrived_ticks + 1 if distance < .2 and np.linalg.norm(after[7:10]) < .3 else 0
        if arrived_ticks >= round(.5 / dt):
            break
        if np.any(after[:3] < scene.lower - 1.) or np.any(after[:3] > scene.upper + 1.):
            break
    wall = time.perf_counter() - started
    arrays = dict(time=np.asarray(times), states=np.asarray(states), references=np.asarray(references),
                  physics_states=np.asarray(substeps), contact_force=np.asarray(contacts), motor_states=np.asarray(motor_states),
                  requested_wrench=np.asarray(commands), allocated_wrench=np.asarray(allocated))
    np.savez_compressed(output / "flight.npz", **arrays)
    (output / "planning.json").write_text(json.dumps(plans, indent=2, allow_nan=False) + "\n")
    (output / "mapping.json").write_text(json.dumps(map_updates, indent=2, allow_nan=False) + "\n")
    report = summarize(scene, times, states, references, substeps, contacts, motor_states, plans, wall,
                       arrived=arrived_ticks >= round(.5 / dt))
    cycles = np.asarray([p["cycle_wall_s"] for p in plans])
    def finite_float(value):
        return float(value) if np.isfinite(value) else None

    report.update({"complete_cycle_p50_s": float(np.median(cycles)),
                   "complete_cycle_p95_s": float(np.quantile(cycles, .95)),
                   "complete_cycle_deadline_misses_0_5s": int((cycles > .5).sum()),
                   "max_requested_thrust_N": finite_float(np.max(arrays["requested_wrench"][:, 0])),
                   "max_allocated_thrust_N": finite_float(np.max(arrays["allocated_wrench"][:, 0])),
                   "peak_wrench_allocation_difference": [finite_float(x) for x in
                       np.max(np.abs(arrays["requested_wrench"] - arrays["allocated_wrench"]), axis=0)],
                   "max_observed_acceleration_m_s2": finite_float(np.linalg.norm(
                       np.diff(arrays["states"][:, 7:10], axis=0) / dt, axis=1).max(initial=0.)),
                   "lidar": {"horizontal_rays": 440, "vertical_rays": 16, "min_range_m": .1,
                             "max_range_m": 100., "body_translation_m": [0., 0., .25], "update_hz": 2}})
    report.update({"seed": seed, "map_updates": len(map_updates), "mapping_initialization": planner.initialization_timings,
                   "runtime": {"python": sys.version, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(env.device),
                               "numpy": np.__version__, "newton": importlib.metadata.version("newton"),
                               "isaaclab_release": "3.0.0", "isaaclab_core_distribution": importlib.metadata.version("isaaclab")}})
    (output / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if initial_result is not None:
        save_plan(output, initial_result)
        save_preview(output / "flight.html", scene, arrays["states"], initial_result.corridor, initial_result.trajectory)
    print("RESULT " + json.dumps(report, allow_nan=False), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=("medium", "dense", "both"), default="both")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--duration", type=float, default=60.)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=Path("results/demo"))
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.scene == "both":
        args.output.mkdir(parents=True, exist_ok=True)
        for name in ("medium", "dense"):
            command = [sys.executable, "-m", "isaac_pathcover.demo", "--scene", name, "--seeds", *map(str, args.seeds),
                       "--duration", str(args.duration), "--device", args.device, "--output", str(args.output / name)]
            if args.plan_only:
                command.append("--plan-only")
            subprocess.run(command, check=True)
        return
    scene = Scene(args.scene)
    args.output.mkdir(parents=True, exist_ok=True)
    source_root = Path(__file__).parent
    hashes = {str(path.relative_to(source_root)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(source_root.rglob("*.py"))}
    (args.output / "source_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    if args.plan_only:
        for seed in args.seeds:
            planner = Planner(scene, seed=seed)
            result = planner.plan(np.column_stack((scene.start, np.zeros((3, 2)))))
            print(result.status, result.timings, flush=True)
            if not result.success:
                raise RuntimeError(result.status)
            save_plan(args.output / f"seed{seed:03d}", result)
        return
    import torch
    from isaac_pathcover.simulation.environment import FlightEnv, make_env_cfg
    torch.set_num_threads(1)
    boot = time.perf_counter()
    env = FlightEnv(make_env_cfg(scene, device=args.device, seed=args.seeds[0]))
    startup = time.perf_counter() - boot
    try:
        reports = [run_episode(env, scene, seed, args.duration, args.output / f"seed{seed:03d}") for seed in args.seeds]
    finally:
        env.close()
    aggregate = {"scene": args.scene, "seeds": args.seeds, "episodes": len(reports), "startup_s": startup,
                 "arrival_rate": float(np.mean([r["arrived"] for r in reports])),
                 "collision_rate": float(np.mean([r["collision"] for r in reports])),
                 "collision_free_arrival_rate": float(np.mean([r["collision_free_arrival"] for r in reports])),
                 "reports": reports}
    (args.output / "summary.json").write_text(json.dumps(aggregate, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
