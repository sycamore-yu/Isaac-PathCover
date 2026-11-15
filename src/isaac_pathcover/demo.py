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


class FlightTrace:
    """The common Newton control step, flight samples and arrival condition."""

    def __init__(self, env, scene, execution):
        self.env, self.scene, self.execution = env, scene, execution
        self.dt = env.step_dt
        self.tracker = GeometricTracker()
        self.previous_velocity = np.zeros(3)
        self.acceleration = np.zeros(3)
        self.arrived_ticks = 0
        self.times, self.states, self.references = [], [], []
        self.motor_states, self.contacts, self.substeps = [], [], []
        self.commands, self.allocated = [], []

    @property
    def arrived(self):
        return self.arrived_ticks >= round(.5 / self.dt)

    def state(self, tick):
        state = self.env.physical_state()[0].detach().cpu().numpy().astype(float)
        if tick:
            self.acceleration = (state[7:10] - self.previous_velocity) / self.dt
        self.previous_velocity = state[7:10].copy()
        return state

    def step(self, now, state):
        import torch

        reference = self.execution.reference(now)
        wrench, _ = self.tracker.command(state, reference[:, 0], reference[:, 1], reference[:, 2], self.dt)
        self.env.step(torch.as_tensor(wrench[None], dtype=torch.float32, device=self.env.device))
        after = self.env.physical_state()[0].detach().cpu().numpy().astype(float)
        self.times.append(now + self.dt)
        self.states.append(after)
        self.references.append(self.execution.reference(now + self.dt))
        self.contacts.append(float(self.env.contact_force()))
        self.motor_states.append(self.env.actuator.motor_state[0].detach().cpu().numpy().copy())
        self.commands.append(wrench)
        self.allocated.append(self.env.actuator.allocated_wrench[0].detach().cpu().numpy().copy())
        self.substeps.extend(torch.stack(self.env.substep_states).detach().cpu().numpy())
        if not np.isfinite(after).all():
            return True
        distance = np.linalg.norm(after[:3] - self.scene.goal)
        self.arrived_ticks = self.arrived_ticks + 1 if distance < .2 and np.linalg.norm(after[7:10]) < .3 else 0
        return (self.arrived or np.any(after[:3] < self.scene.lower - 1.)
                or np.any(after[:3] > self.scene.upper + 1.))


def run_episode(env, scene, seed, duration, output, update_map=True):
    from isaac_pathcover.trajectory.diagnostics import trajectory_diagnostics

    output.mkdir(parents=True, exist_ok=True)
    env.reset()
    planner = Planner(scene, seed=seed)
    execution = TrajectoryExecution(scene.start)
    flight = FlightTrace(env, scene, execution)
    plans, map_updates = [], []
    initial_result = None
    next_replan = 0.
    started = time.perf_counter()
    for tick in range(int(round(duration / flight.dt))):
        now = tick * flight.dt
        state = flight.state(tick)
        if now + 1e-9 >= next_replan and not flight.arrived_ticks:
            cycle_started = time.perf_counter()
            if update_map and tick:
                scan_started = time.perf_counter()
                sensor_origin, endpoints = scene.lidar(state[:3], state[3:7])
                scan_elapsed = time.perf_counter() - scan_started
                update = planner.update_map(sensor_origin, endpoints)
                map_updates.append({"t": now, "points": len(endpoints), "sensor_s": scan_elapsed,
                                    "sensor_origin": sensor_origin.tolist(), **update})
            measured = np.column_stack((state[:3], state[7:10], flight.acceleration))
            result = planner.plan(measured, guide=execution.trajectory,
                                  guide_elapsed=now - execution.start_time)
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
                               "initialization": result.optimization.initialization,
                               "cost": result.optimization.cost if np.isfinite(result.optimization.cost) else None})
            if result.success:
                record["trajectory_diagnostics"] = finite_json(diagnostics)
                if initial_result is None and handoff["accepted"]:
                    initial_result = result
                save_plan(output / "plans" / f"{len(plans):04d}", result)
            plans.append(record)
            print(json.dumps({"scene": scene.name, "seed": seed, **record}), flush=True)
            next_replan = now + .5
        if flight.step(now, state):
            break
    wall = time.perf_counter() - started
    return finish_episode(scene, seed, output, flight, plans, map_updates, wall,
                          initial_result, planner.initialization_timings, mode="synchronous")


def finish_episode(scene, seed, output, flight, plans, map_updates, wall,
                   initial_result, planner_initialization, *, mode, extra=None):
    import torch
    from isaac_pathcover.simulation.metrics import summarize
    from isaac_pathcover.simulation.visualization import save_preview

    arrays = dict(time=np.asarray(flight.times), states=np.asarray(flight.states),
                  references=np.asarray(flight.references), physics_states=np.asarray(flight.substeps),
                  contact_force=np.asarray(flight.contacts), motor_states=np.asarray(flight.motor_states),
                  requested_wrench=np.asarray(flight.commands), allocated_wrench=np.asarray(flight.allocated))
    np.savez_compressed(output / "flight.npz", **arrays)
    (output / "planning.json").write_text(json.dumps(plans, indent=2, allow_nan=False) + "\n")
    (output / "mapping.json").write_text(json.dumps(map_updates, indent=2, allow_nan=False) + "\n")
    report = summarize(scene, flight.times, flight.states, flight.references, flight.substeps,
                       flight.contacts, flight.motor_states, plans, wall, arrived=flight.arrived)
    cycles = np.asarray([p["cycle_wall_s"] for p in plans])
    def finite_float(value):
        return float(value) if np.isfinite(value) else None

    report.update({"complete_cycle_p50_s": float(np.median(cycles)) if len(cycles) else None,
                   "complete_cycle_p95_s": float(np.quantile(cycles, .95)) if len(cycles) else None,
                   "complete_cycles_over_0_5s": int((cycles > .5).sum()),
                   "max_requested_thrust_N": finite_float(np.max(arrays["requested_wrench"][:, 0])),
                   "max_allocated_thrust_N": finite_float(np.max(arrays["allocated_wrench"][:, 0])),
                   "peak_wrench_allocation_difference": [finite_float(x) for x in
                       np.max(np.abs(arrays["requested_wrench"] - arrays["allocated_wrench"]), axis=0)],
                   "max_observed_acceleration_m_s2": finite_float(np.linalg.norm(
                       np.diff(arrays["states"][:, 7:10], axis=0) / flight.dt, axis=1).max(initial=0.)),
                   "lidar": {"horizontal_rays": 440, "vertical_rays": 16, "min_range_m": .1,
                             "max_range_m": 100., "body_translation_m": [0., 0., .25], "update_hz": 2}})
    report.update({"seed": seed, "map_updates": len(map_updates), "mapping_initialization": planner_initialization,
                   "runtime": {"python": sys.version, "torch": torch.__version__,
                               "gpu": torch.cuda.get_device_name(flight.env.device),
                               "numpy": np.__version__, "newton": importlib.metadata.version("newton"),
                               "isaaclab_release": "3.0.0", "isaaclab_core_distribution": importlib.metadata.version("isaaclab")}})
    if mode == "asynchronous":
        report["time_contract"] = "separate CPU planner; Newton advances during planning; handoff gated by measured state and physical constraints"
        report["lidar"]["nominal_request_hz"] = 2.
        report["lidar"]["update_hz"] = (len(map_updates) / flight.times[-1]) if flight.times else 0.
        report.update(extra or {})
    (output / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if initial_result is not None:
        save_plan(output, initial_result)
        save_preview(output / "flight.html", scene, arrays["states"], initial_result.corridor, initial_result.trajectory)
    print("RESULT " + json.dumps(report, allow_nan=False), flush=True)
    return report


def run_episode_async(env, scene, seed, duration, output, *, forward_s=0.8, update_map=True):
    """Keep stepping Newton while a separate process plans from a future state.

    Following SUPER's forward-state handoff, the new candidate is initialized
    on the active reference at a predicted time at least ``forward_s`` ahead.
    The active trajectory continues until handoff. Candidates are accepted
    whenever the measured state still matches their start and their sampled
    physical constraints pass, even if the planned handoff time has passed.
    """
    from isaac_pathcover.control.async_planner import AsyncPlanningWorker, handoff_horizon

    if forward_s <= 0:
        raise ValueError("forward horizon must be positive")
    output.mkdir(parents=True, exist_ok=True)
    env.reset()
    execution = TrajectoryExecution(scene.start)
    flight = FlightTrace(env, scene, execution)
    plans, map_updates = [], []
    finished_candidates = []
    initial_result = None
    next_replan = 0.
    inflight = pending = None
    ready_at_sim_s = None
    launched = skipped = steps_while_busy = ready_scheduled = mismatches = 0
    max_progress = 0.
    previous_solve_progress_s = forward_s
    started_wall = time.perf_counter()
    dt = flight.dt
    worker = AsyncPlanningWorker(scene.name, seed)

    def finish_candidate(record, result, validation, state, now, predicted):
        nonlocal initial_result, mismatches, previous_solve_progress_s
        measured = np.column_stack((state[:3], state[7:10], flight.acceleration))
        start = np.column_stack([result.trajectory.evaluate(0., derivative=k) for k in range(3)])
        record["prediction_error_pva"] = np.linalg.norm(predicted - measured, axis=0).tolist()
        error = np.linalg.norm(start[:, :2] - measured[:, :2], axis=0)
        if error[0] > 0.2 or error[1] > 0.5:
            record["handoff"] = {"accepted": False,
                                  "retained_previous": execution.trajectory is not None,
                                  "rejection_reason": ["predicted_state_mismatch"],
                                  "start_error_p_m": float(error[0]),
                                  "start_error_v_m_s": float(error[1])}
            mismatches += 1
        else:
            record["handoff"] = execution.replace(result, now, measured, validation=validation)
            if record["handoff"]["accepted"]:
                previous_solve_progress_s = record["physics_progress_during_solve_s"]
            if record["handoff"]["accepted"] and initial_result is None:
                initial_result = result
        record["activation_t"] = now
        record["activation_delay_sim_s"] = now - record["target_t"] if record["target_t"] is not None else 0.0
        record["activation_state_error_pva"] = np.linalg.norm(start - measured, axis=0).tolist()
        plans.append(record)

    try:
        for tick in range(int(round(duration / dt))):
            now = tick * dt
            state = flight.state(tick)

            # Nonblocking check: the Newton step is never held for a solve.
            message = worker.poll()
            if worker.ready and ready_at_sim_s is None:
                ready_at_sim_s = now
            if message is not None:
                request = inflight
                assert request is not None and message["id"] == request["id"]
                inflight = None
                elapsed = time.perf_counter() - request["submitted_wall"]
                max_progress = max(max_progress, now - request["t"])
                if message["mapping"] is not None:
                    map_updates.append(message["mapping"])
                result = message["result"]
                validation = message["validation"]
                record = {"t": request["t"], "target_t": request["target_t"],
                          "result_t": now, "physics_progress_during_solve_s": now - request["t"],
                          "success": result.success,
                          "status": result.status, "timings": result.timings,
                          "cycle_wall_s": elapsed, "worker_s": message["elapsed_wall_s"],
                          "execution_check": validation}
                if result.optimization is not None:
                    record.update({"iterations": result.optimization.iterations,
                                   "evaluations": result.optimization.evaluations,
                                   "initialization": result.optimization.initialization,
                                   "cost": result.optimization.cost if np.isfinite(result.optimization.cost) else None})
                if result.success:
                    finished_candidates.append((len(plans), result))
                    if message["diagnostics"] is not None:
                        record["trajectory_diagnostics"] = finite_json(message["diagnostics"])
                rejection = None
                if not result.success:
                    rejection = "solver_failure"
                elif validation is None or not validation["accepted"]:
                    rejection = "execution_check_failed"
                if rejection is not None:
                    record["handoff"] = {"accepted": False,
                                          "retained_previous": execution.trajectory is not None,
                                          "rejection_reason": [rejection]}
                    plans.append(record)
                elif request["target_t"] is None or now + dt * .5 >= request["target_t"]:
                    finish_candidate(record, result, validation, state, now, request["start_state"])
                else:
                    pending = (record, result, validation, request["target_t"], request["start_state"])

            if pending is not None and now + dt * .5 >= pending[3]:
                record, result, validation, target, predicted = pending
                pending = None
                ready_scheduled += 1
                finish_candidate(record, result, validation, state, now, predicted)

            if now + 1e-9 >= next_replan and not flight.arrived_ticks:
                if worker.ready and not worker.busy and pending is None:
                    measured = np.column_stack((state[:3], state[7:10], flight.acceleration))
                    horizon = handoff_horizon(forward_s, previous_solve_progress_s)
                    target = round(now + horizon, 6) if execution.trajectory is not None else None
                    predicted = execution.reference(target)[:, :3] if target is not None else measured
                    launched += 1
                    request = {"kind": "plan", "id": launched, "t": now,
                               "position": state[:3].copy(), "quaternion": state[3:7].copy(),
                               "start_state": predicted.copy(),
                               "guide": execution.trajectory, "guide_elapsed":
                                   (target - execution.start_time) if target is not None else 0.,
                               "target_t": target, "scan": update_map and target is not None,
                               "submitted_wall": time.perf_counter()}
                    worker.submit(request)
                    inflight = request
                else:
                    skipped += 1
                next_replan += .5

            if worker.busy:
                steps_while_busy += 1
            if flight.step(now, state):
                break
    finally:
        wall = time.perf_counter() - started_wall
        worker_init = worker.initialization_timings
        busy_at_end = worker.busy
        worker.close()

    if pending is not None:
        record = pending[0]
        record["handoff"] = {"accepted": False, "retained_previous": execution.trajectory is not None,
                              "rejection_reason": ["episode_ended_before_activation"]}
        plans.append(record)
    # Compress and serialize large debug artifacts only after the flight loop.
    for index, result in finished_candidates:
        save_plan(output / "plans" / f"{index:04d}", result)
    extra = {"forward_horizon_sim_s": forward_s, "planner_ready_sim_s": ready_at_sim_s,
             "planning_jobs_submitted": launched, "planning_jobs_returned": len(plans),
             "planning_slots_skipped": skipped, "physics_steps_while_worker_busy": steps_while_busy,
             "physics_progress_while_planning_sim_s": steps_while_busy * dt,
             "max_progress_during_one_solve_sim_s": max_progress,
             "scheduled_handoffs_examined": ready_scheduled,
             "handoff_state_mismatch_rejections": mismatches,
             "job_running_at_episode_end": busy_at_end}
    return finish_episode(scene, seed, output, flight, plans, map_updates, wall,
                          initial_result, worker_init, mode="asynchronous", extra=extra)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=("medium", "dense", "both"), default="both")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--duration", type=float, default=60.)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=Path("results/demo"))
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--synchronous", action="store_true",
                        help="Use the original paused-simulation baseline")
    parser.add_argument("--forward-horizon", type=float, default=0.8,
                        help="Simulation-time seconds to predict the next handoff")
    args = parser.parse_args()
    if args.scene == "both":
        args.output.mkdir(parents=True, exist_ok=True)
        for name in ("medium", "dense"):
            command = [sys.executable, "-m", "isaac_pathcover.demo", "--scene", name, "--seeds", *map(str, args.seeds),
                       "--duration", str(args.duration), "--device", args.device, "--output", str(args.output / name)]
            if args.plan_only:
                command.append("--plan-only")
            if args.synchronous:
                command.append("--synchronous")
            command += ["--forward-horizon", str(args.forward_horizon)]
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
        runner = run_episode if args.synchronous else run_episode_async
        runner_kwargs = {} if args.synchronous else {"forward_s": args.forward_horizon}
        reports = [runner(env, scene, seed, args.duration,
                          args.output / f"seed{seed:03d}", **runner_kwargs) for seed in args.seeds]
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
