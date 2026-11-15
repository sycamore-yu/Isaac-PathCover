"""Run the native Python SUPER + RISP planner with the existing Newton quadrotor."""
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

from isaac_pathcover.control.execution import ExecutionFSM, MachineState, TrajectoryExecution
from isaac_pathcover.control.tracking import GeometricTracker
from isaac_pathcover.planning.pipeline import Planner, PlannerConfig
from isaac_pathcover.simulation.scene import Scene

# fsm_ros2.hpp uses int(1000 / replan_rate) milliseconds; static_dense.yaml is 15 Hz.
FSM_COMMAND_WALL_PERIOD_NS = 10_000_000
REPLAN_WALL_PERIOD_NS = int(1.0 / PlannerConfig().replan_rate * 1000) * 1_000_000
REPLAN_WALL_PERIOD = REPLAN_WALL_PERIOD_NS / 1e9
LIDAR_SIM_PERIOD = .5


def finite_json(value):
    """Keep raw arrays in NPZ and represent nonfinite JSON diagnostics as null."""
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return finite_json(value.tolist())
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def save_polynomial(path, trajectory, **extra):
    np.savez_compressed(path, coefficients=trajectory.coefficients,
                        durations=trajectory.durations, **extra)


def next_wall_deadline(deadline, now, period):
    """Advance a due ROS-style periodic timer without resetting its phase."""
    return deadline + ((now - deadline) // period + 1) * period


def save_plan(path, result):
    path.mkdir(parents=True, exist_ok=True)
    if result.trajectory is not None:
        search_path = result.search.path if result.search is not None else np.empty((0, 3))
        save_polynomial(path / "trajectory.npz", result.trajectory, search_path=search_path)
    if result.guide_path is not None:
        np.savez_compressed(path / "guide.npz", positions=result.guide_path, times=result.guide_times)
    if result.optimization is not None and result.optimization.trajectory is not None:
        save_polynomial(path / "candidate.npz", result.optimization.trajectory,
                        init_times=result.optimization.init_times, init_points=result.optimization.init_points)
    if result.exp is not None and not result.exp.empty:
        save_polynomial(path / "exp.npz", result.exp.position, start_time=result.exp.start_time)
        save_polynomial(path / "exp_yaw.npz", result.exp.yaw, start_time=result.exp.start_time)
    if result.backup is not None and not result.backup.empty:
        save_polynomial(path / "backup.npz", result.backup.position,
                        start_time=result.backup.start_time, start_tt=result.backup.start_tt)
    if result.backup is not None and result.backup.corridor is not None:
        backup_corridor = {"planes": result.backup.corridor.planes,
                           "seed_line": result.backup.corridor.seed_line,
                           "robot_position": result.backup.robot_position,
                           "start_tt": result.backup.start_tt}
        (path / "backup_corridor.json").write_text(
            json.dumps(finite_json(backup_corridor), indent=2, allow_nan=False) + "\n")
    if result.command is not None:
        save_polynomial(path / "command.npz", result.command.position,
                        start_time=result.command.start_time,
                        backup_start=result.command.backup_start)
    if result.corridor is not None:
        corridor = {"polytopes": [{"A": a.tolist(), "b": b.tolist()}
                                  for a, b in result.corridor.polytopes],
                    "seeds": result.corridor.seeds.tolist(),
                    "risp_remaining_counts": result.corridor.remaining_counts,
                    "goal": result.corridor.goal.tolist()}
        (path / "corridor.json").write_text(json.dumps(corridor, indent=2) + "\n")


class FlightTrace:
    """Record actual Newton state and the source position/yaw command on one clock."""

    def __init__(self, env, scene, execution):
        self.env, self.scene, self.execution = env, scene, execution
        self.dt = env.step_dt
        self.tracker = GeometricTracker()
        self.previous_velocity = np.zeros(3)
        self.acceleration = np.zeros(3)
        self.jerk = np.zeros(3)
        self.arrived_ticks = 0
        self.times, self.states, self.references = [], [], []
        self.motor_states, self.contacts, self.substeps = [], [], []
        self.commands, self.allocated = [], []
        self.yaw_references, self.backup_flags, self.fsm_states = [], [], []
        self.command = execution.sample(0.)
        self.command_time = 0.
        self.command_times = []

    @property
    def arrived(self):
        return self.arrived_ticks >= round(.5 / self.dt)

    def state(self, tick):
        state = self.env.physical_state()[0].detach().cpu().numpy().astype(float)
        if tick:
            acceleration = (state[7:10] - self.previous_velocity) / self.dt
            self.jerk = (acceleration - self.acceleration) / self.dt
            self.acceleration = acceleration
        self.previous_velocity = state[7:10].copy()
        return state

    def measured_pvaj(self, state):
        return np.column_stack((state[:3], state[7:10], self.acceleration, self.jerk))

    def publish_command(self, now, state, fsm):
        """Source PubCmdCallback; the controller holds the last published command."""
        if fsm.stop:
            return None
        if fsm.state in (MachineState.FOLLOW_TRAJ, MachineState.EMER_STOP):
            self.command = self.execution.sample(now)
            self.command_time = now
            fsm.observe_command(self.command, state[:3], now)
            return self.command.on_backup
        return None

    def step(self, now, state, fsm):
        import torch

        sample = self.command
        reference = sample.state
        wrench, _ = self.tracker.command(state, reference[:, 0], reference[:, 1], reference[:, 2],
                                         yaw=sample.yaw, yaw_rate=sample.yaw_rate)
        self.env.step(torch.as_tensor(wrench[None], dtype=torch.float32, device=self.env.device))
        after = self.env.physical_state()[0].detach().cpu().numpy().astype(float)
        self.times.append(now + self.dt)
        self.states.append(after)
        self.references.append(reference.copy())
        self.command_times.append(self.command_time)
        self.yaw_references.append([sample.yaw, sample.yaw_rate])
        self.backup_flags.append(sample.on_backup)
        self.fsm_states.append(int(fsm.state))
        self.contacts.append(float(self.env.contact_force()))
        self.motor_states.append(self.env.actuator.motor_state[0].detach().cpu().numpy().copy())
        self.commands.append(wrench)
        self.allocated.append(self.env.actuator.allocated_wrench[0].detach().cpu().numpy().copy())
        self.substeps.extend(torch.stack(self.env.substep_states).detach().cpu().numpy())
        if not np.isfinite(after).all():
            return True
        distance = np.linalg.norm(after[:3] - fsm.goal)
        self.arrived_ticks = self.arrived_ticks + 1 if distance < .1 and np.linalg.norm(after[7:10]) < .3 else 0
        return (self.arrived or np.any(after[:3] < self.scene.lower - 1.)
                or np.any(after[:3] > self.scene.upper + 1.))


def finish_episode(scene, seed, output, flight, plans, map_updates, wall,
                   artifacts, planner_initialization, fsm, extra):
    import torch
    from isaac_pathcover.simulation.metrics import summarize, summarize_timeline
    from isaac_pathcover.simulation.visualization import save_preview

    arrays = dict(time=np.asarray(flight.times), states=np.asarray(flight.states),
                  references=np.asarray(flight.references), physics_states=np.asarray(flight.substeps),
                  contact_force=np.asarray(flight.contacts), motor_states=np.asarray(flight.motor_states),
                  requested_wrench=np.asarray(flight.commands), allocated_wrench=np.asarray(flight.allocated),
                  yaw_references=np.asarray(flight.yaw_references),
                  command_time=np.asarray(flight.command_times),
                  on_backup=np.asarray(flight.backup_flags), fsm_state=np.asarray(flight.fsm_states))
    np.savez_compressed(output / "flight.npz", **arrays)
    np.savetxt(output / "speed.csv", np.column_stack((arrays["time"],
               np.linalg.norm(arrays["states"][:, 7:10], axis=1),
               np.linalg.norm(arrays["references"][:, :, 1], axis=1), arrays["on_backup"],
               arrays["fsm_state"])), delimiter=",",
               header="sim_time_s,actual_speed_m_s,reference_speed_m_s,on_backup,fsm_state", comments="")
    (output / "planning.json").write_text(json.dumps(finite_json(plans), indent=2, allow_nan=False) + "\n")
    (output / "mapping.json").write_text(json.dumps(finite_json(map_updates), indent=2, allow_nan=False) + "\n")
    (output / "fsm.json").write_text(json.dumps(fsm.transitions, indent=2) + "\n")
    report = summarize(scene, flight.times, flight.states, flight.references, flight.substeps,
                       flight.contacts, flight.motor_states, plans, wall, arrived=flight.arrived)
    timeline = summarize_timeline(arrays, plans, fsm.goal)
    (output / "timeline.json").write_text(json.dumps(finite_json(timeline), indent=2, allow_nan=False) + "\n")
    cycles = np.asarray([p["cycle_wall_s"] for p in plans])
    report.update({
        "seed": seed, "map_updates": len(map_updates), "mapping_initialization": planner_initialization,
        "resolved_goal_error_m": float(np.linalg.norm(arrays["states"][-1, :3] - fsm.goal)),
        "goal_resolution_shift_m": float(np.linalg.norm(fsm.goal - scene.goal)),
        "arrival_rule": "within 0.1 m of the source-resolved goal and below 0.3 m/s for 0.5 s",
        "complete_cycle_p50_s": float(np.median(cycles)) if len(cycles) else None,
        "complete_cycle_p95_s": float(np.quantile(cycles, .95)) if len(cycles) else None,
        "max_requested_thrust_N": np.max(arrays["requested_wrench"][:, 0]),
        "max_allocated_thrust_N": np.max(arrays["allocated_wrench"][:, 0]),
        "peak_wrench_allocation_difference": np.max(
            np.abs(arrays["requested_wrench"] - arrays["allocated_wrench"]), axis=0),
        "max_observed_acceleration_m_s2": np.linalg.norm(
            np.diff(arrays["states"][:, 7:10], axis=0) / flight.dt, axis=1).max(initial=0.),
        "lidar": {"horizontal_rays": 440, "vertical_rays": 16, "min_range_m": .1,
                  "max_range_m": 100., "body_translation_m": [0., 0., .25],
                  "requested_update_hz": 1. / LIDAR_SIM_PERIOD,
                  "observed_update_hz": len(map_updates) / flight.times[-1]},
        "runtime": {"python": sys.version, "torch": torch.__version__,
                    "gpu": torch.cuda.get_device_name(flight.env.device), "numpy": np.__version__,
                    "scipy": importlib.metadata.version("scipy"),
                    "numba": importlib.metadata.version("numba"),
                    "newton": importlib.metadata.version("newton"),
                    "isaaclab_release": "3.0.0",
                    "isaaclab_core_distribution": importlib.metadata.version("isaaclab"),
                    "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
                    "omp_num_threads": os.environ.get("OMP_NUM_THREADS")},
        "time_contract": "independent CPU planner and continuously advancing Newton simulation; "
                         "source trajectory timestamps and source commit rules",
        "replan_wall_period_s": REPLAN_WALL_PERIOD,
        "fsm_command_wall_period_s": FSM_COMMAND_WALL_PERIOD_NS / 1e9,
        "reference_contract": "last published command applied over the recorded Newton step; "
                              "command_time stores its simulation timestamp",
        "timeline": timeline, **extra,
    })
    report = finite_json(report)
    (output / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    initial = None
    for index, result in artifacts:
        save_plan(output / "plans" / f"{index:04d}", result)
        if initial is None and result.command is not None:
            initial = result
    if initial is not None:
        save_plan(output, initial)
        save_preview(output / "flight.html", scene, arrays["states"], initial.corridor, initial.trajectory)
    print("RESULT " + json.dumps(report, allow_nan=False), flush=True)
    return report


def run_episode(env, scene, seed, duration, output):
    """Drive the source FSM while its planning callback runs in a CPU process."""
    from isaac_pathcover.control.async_planner import AsyncPlanningWorker

    output.mkdir(parents=True, exist_ok=True)
    env.reset()
    execution = TrajectoryExecution(scene.start)
    fsm = ExecutionFSM(scene.start, scene.goal)
    flight = FlightTrace(env, scene, execution)
    plans, map_updates, artifacts = [], [], []
    inflight = None
    ready_at_sim_s = None
    launched = busy_steps = 0
    max_progress = 0.
    next_scan_sim = LIDAR_SIM_PERIOD
    started_wall = time.perf_counter()
    next_main_wall_ns = next_replan_wall_ns = next_command_wall_ns = 0
    worker = AsyncPlanningWorker(scene.name, seed)
    try:
        for tick in range(int(round(duration / flight.dt))):
            now = tick * flight.dt
            state = flight.state(tick)
            qw, qx, qy, qz = state[3:7]
            yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
            worker.observe(now, flight.measured_pvaj(state), state[3:7], yaw)
            message = worker.poll()
            if worker.ready and ready_at_sim_s is None:
                ready_at_sim_s = now
                timer_epoch = time.perf_counter_ns()
                next_main_wall_ns = next_command_wall_ns = timer_epoch + FSM_COMMAND_WALL_PERIOD_NS
                next_replan_wall_ns = timer_epoch + REPLAN_WALL_PERIOD_NS
                if worker.resolved_goal is None:
                    fsm.started = fsm.new_goal = False
                else:
                    fsm.goal = np.asarray(worker.resolved_goal)
            if message is not None:
                request = inflight
                assert request is not None and request["id"] == message["id"]
                inflight = None
                result = message["result"]
                fsm.goal = np.asarray(result.events["resolved_goal"])
                progress = now - request["t"]
                max_progress = max(max_progress, progress)
                handoff = (execution.install(result.command, now) if result.command is not None
                           else {"accepted": False, "retained_previous": execution.command is not None})
                if result.success:
                    worker.set_backup_flag(False)
                fsm.accept_result(result, now)
                record = {
                    "id": request["id"], "operation": request["operation"], "t": request["t"],
                    "plan_start_t": message["plan_start_t"], "plan_end_t": message["plan_end_t"],
                    "result_t": now, "activation_t": now if result.command is not None else None,
                    "physics_progress_during_solve_s": progress, "success": result.success,
                    "code": int(result.code), "status": result.status, "timings": result.timings,
                    "exp_status": result.exp_status, "backup_status": result.backup_status,
                    "events": result.events,
                    "cycle_wall_s": time.perf_counter() - request["submitted_wall"],
                    "worker_s": message["elapsed_wall_s"], "handoff": handoff,
                    "fsm_state": fsm.state.name,
                }
                if result.optimization is not None:
                    opt = result.optimization
                    record.update({"iterations": opt.iterations, "cost": opt.cost,
                                   "evaluations": opt.evaluations,
                                   "optimizer_status": opt.status, "penalty_log": opt.penalty_log})
                if result.search is not None:
                    record["search"] = {"status": result.search.status,
                                        "expanded": result.search.expanded,
                                        "cost_m": result.search.cost,
                                        "elapsed_sim_s": result.search.elapsed}
                if result.command is not None:
                    record["command_duration"] = result.command.duration
                if result.backup_optimization is not None:
                    opt = result.backup_optimization
                    record["backup_optimizer"] = {"status": opt.status, "cost": opt.cost,
                                                   "iterations": opt.iterations,
                                                   "evaluations": opt.evaluations,
                                                   "penalty_log": opt.penalty_log,
                                                   "switch_time": opt.switch_time}
                if message["mapping"] is not None:
                    map_updates.append(message["mapping"])
                artifacts.append((len(plans), result))
                plans.append(record)
                print(json.dumps(finite_json({"scene": scene.name, "seed": seed, **record})), flush=True)

            wall_now_ns = time.perf_counter_ns()
            if worker.ready and not worker.busy:
                operation = None
                if wall_now_ns >= next_main_wall_ns:
                    operation = fsm.main_tick(state[:3], now)
                    next_main_wall_ns = next_wall_deadline(
                        next_main_wall_ns, wall_now_ns, FSM_COMMAND_WALL_PERIOD_NS)
                if operation is None and wall_now_ns >= next_replan_wall_ns:
                    operation = fsm.replan_tick()
                    next_replan_wall_ns = next_wall_deadline(
                        next_replan_wall_ns, wall_now_ns, REPLAN_WALL_PERIOD_NS)
                if operation is not None:
                    launched += 1
                    scan = now >= next_scan_sim
                    if scan:
                        next_scan_sim = now + LIDAR_SIM_PERIOD
                    inflight = {"kind": "plan", "id": launched, "t": now,
                                "operation": operation, "new_goal": fsm.new_goal,
                                "scan": scan, "submitted_wall": time.perf_counter()}
                    worker.submit(inflight)
            if worker.busy:
                busy_steps += 1
            if worker.ready and wall_now_ns >= next_command_wall_ns:
                on_backup = flight.publish_command(now, state, fsm)
                if on_backup is not None:
                    worker.set_backup_flag(on_backup)
                next_command_wall_ns = next_wall_deadline(
                    next_command_wall_ns, wall_now_ns, FSM_COMMAND_WALL_PERIOD_NS)
            stop = flight.step(now, state, fsm)
            worker.advance_clock(now + flight.dt)
            if stop:
                break
    finally:
        wall = time.perf_counter() - started_wall
        initialization = worker.initialization_timings
        busy_at_end = worker.busy
        worker.close()

    extra = {"planner_ready_sim_s": ready_at_sim_s, "planning_jobs_submitted": launched,
             "planning_jobs_returned": len(plans), "control_steps_while_worker_busy": busy_steps,
             "physics_steps_while_worker_busy": busy_steps * env.cfg.decimation,
             "physics_progress_while_planning_sim_s": busy_steps * flight.dt,
             "max_progress_during_one_solve_sim_s": max_progress,
             "job_running_at_episode_end": busy_at_end, "final_fsm_state": fsm.state.name,
             "resolved_goal_m": fsm.goal}
    return finish_episode(scene, seed, output, flight, plans, map_updates, wall,
                          artifacts, initialization, fsm, extra)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=("medium", "dense", "both"), default="both")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--duration", type=float, default=90.,
                        help="Episode observation length in simulation seconds")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=Path("results/super_risp"))
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.scene == "both":
        args.output.mkdir(parents=True, exist_ok=True)
        for name in ("medium", "dense"):
            command = [sys.executable, "-m", "isaac_pathcover.demo", "--scene", name,
                       "--seeds", *map(str, args.seeds), "--duration", str(args.duration),
                       "--device", args.device, "--output", str(args.output / name)]
            if args.plan_only:
                command.append("--plan-only")
            subprocess.run(command, check=True)
        return
    scene = Scene(args.scene)
    args.output.mkdir(parents=True, exist_ok=True)
    source_root = Path(__file__).parent
    hashes = {str(path.relative_to(source_root)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(source_root.rglob("*.py"))}
    if args.plan_only:
        from isaac_pathcover.trajectory.super_opt import warmup
        warmup()
        for seed in args.seeds:
            planner = Planner(scene, seed=seed, clock=lambda: 0.)
            planner.set_robot_state(np.column_stack((scene.start, np.zeros((3, 3)))), yaw=0.)
            resolved_goal = planner.map.nearest_inf_not_occupied(scene.goal, 3.)
            result = planner.plan_from_rest(goal=resolved_goal)
            print(result.status, result.timings, flush=True)
            case = args.output / f"seed{seed:03d}"
            save_plan(case, result)
            (case / "source_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
            (case / "result.json").write_text(json.dumps(finite_json({
                "success": result.success, "status": result.status,
                "timings": result.timings, "backup_status": result.backup_status,
            }), indent=2) + "\n")
            if not result.success:
                raise RuntimeError(result.status)
        return
    import torch
    from isaac_pathcover.simulation.environment import FlightEnv, make_env_cfg

    torch.set_num_threads(1)
    boot = time.perf_counter()
    env = FlightEnv(make_env_cfg(scene, device=args.device, seed=args.seeds[0]))
    startup = time.perf_counter() - boot
    try:
        reports = []
        for seed in args.seeds:
            case = args.output / f"seed{seed:03d}"
            case.mkdir(parents=True, exist_ok=True)
            (case / "source_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
            reports.append(run_episode(env, scene, seed, args.duration, case))
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
