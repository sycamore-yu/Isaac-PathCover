"""Convert a recorded Newton SUPER + RISP flight to an interactive Rerun .rrd.

Requires ``rerun-sdk`` and NumPy. Runs offline from saved flight arrays;
does not start Isaac Lab, Newton or another physical simulation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb


ROOT = Path(__file__).resolve().parents[1]


def sample_polynomial(path: Path, rate: float = 12.0) -> np.ndarray:
    """Sample original MINCO polynomial, coefficients in ascending order."""
    with np.load(path) as data:
        coefficients = data["coefficients"]
        durations = data["durations"]
    knots = np.r_[0.0, np.cumsum(durations)]
    times = np.linspace(0.0, knots[-1], max(2, int(np.ceil(knots[-1] * rate)) + 1))
    indices = np.minimum(np.searchsorted(knots[1:], times, side="left"), len(durations) - 1)
    local = times - knots[indices]
    powers = local[:, None] ** np.arange(coefficients.shape[1])[None, :]
    return np.einsum("nk,nkj->nj", powers, coefficients[indices])


def record(scene_name: str, seed: int, source: Path, output: Path, fps: float) -> dict:
    case = source / scene_name / f"seed{seed:03d}"
    with np.load(case / "flight.npz") as flight:
        times = flight["time"]
        states = flight["states"]
        refs = flight["references"]
        contacts = flight["contact_force"]
        backup = flight["on_backup"]
        fsm_states = flight["fsm_state"]

    scene_file = ROOT / "src/isaac_pathcover/simulation/scenes" / f"{scene_name}.npz"
    with np.load(scene_file) as scene:
        lower = scene["origin"]
        resolution = float(scene["resolution"])
        boxes = scene["boxes"]
        body_size = scene["body_size"]
        start, goal = scene["start"], scene["goal"]
    boxes_min = lower + boxes[:, :3] * resolution
    boxes_max = lower + boxes[:, 3:] * resolution
    centers = (boxes_min + boxes_max) * 0.5
    half_sizes = (boxes_max - boxes_min) * 0.5
    report = json.loads((case / "result.json").read_text())
    plan_records = json.loads((case / "planning.json").read_text())
    accepted = [(float(p["activation_t"]), i) for i, p in enumerate(plan_records)
                if p["handoff"]["accepted"]]
    accepted.sort()

    blueprint = rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial3DView(origin="world", name="Newton flight: 3D"),
                rrb.Vertical(
                    rrb.TimeSeriesView(origin="plots/speed", name="Speed (m/s)"),
                    rrb.TimeSeriesView(origin="plots/error", name="Position error (m)"),
                    rrb.TimeSeriesView(origin="plots/planning", name="Planning time (s)"),
                    rrb.TimeSeriesView(origin="plots/execution", name="Backup and FSM"),
                    rrb.TimeSeriesView(origin="plots/contact", name="Contact force (N)"),
                ),
                column_shares=[0.72, 0.28],
            ),
            rrb.TextLogView(origin="events", name="Planning and FSM events"),
            row_shares=[0.82, 0.18],
        ),
        collapse_panels=False,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with rr.RecordingStream("SUPER-RISP-Newton", recording_id=f"super-risp-{scene_name}-seed{seed:03d}") as rec:
        rec.save(output, default_blueprint=blueprint)
        rec.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        rec.log("world/obstacles", rr.Boxes3D(
            centers=centers, half_sizes=half_sizes,
            colors=np.repeat([[110, 141, 150, 105]], len(centers), axis=0)), static=True)
        rec.log("world/start", rr.Points3D([start], radii=0.24,
                                            colors=[[61, 207, 127]], labels=["START"]), static=True)
        rec.log("world/goal", rr.Points3D([goal], radii=0.24,
                                           colors=[[251, 202, 69]], labels=["GOAL"]), static=True)
        rec.log("world/resolved_goal", rr.Points3D([report["resolved_goal_m"]], radii=0.12,
                    colors=[[170, 238, 90]], labels=["SUPER GOAL"]), static=True)
        rec.log("world/actual_trajectory", rr.LineStrips3D([states[:, :3]],
                                                             colors=[[255, 135, 71]], radii=0.035), static=True)
        rec.log("world/reference_trajectory", rr.LineStrips3D([refs[:, :, 0]],
                                                                colors=[[64, 192, 253]], radii=0.025), static=True)
        # Parent transform moves the drone body, rotor markers and body axes together.
        rec.log("world/drone/body", rr.Boxes3D(half_sizes=[body_size * 0.5],
                                                colors=[[236, 119, 67]], radii=0.008), static=True)
        arm = float(max(body_size[0], body_size[1])) * 0.55
        rec.log("world/drone/arms", rr.LineStrips3D([
            [[-arm, -arm, 0.0], [arm, arm, 0.0]],
            [[-arm, arm, 0.0], [arm, -arm, 0.0]],
        ], colors=[[240, 190, 87]], radii=0.022), static=True)
        rec.log("world/drone/rotors", rr.Points3D([
            [-arm, -arm, 0.0], [arm, arm, 0.0], [-arm, arm, 0.0], [arm, -arm, 0.0],
        ], colors=[[31, 89, 179]], radii=0.055), static=True)
        rec.log("events/run", rr.TextLog(
            f"{scene_name} seed={seed}; arrived={report['arrived']}; "
            f"collision={report['collision']}; out_of_bounds={report['out_of_bounds']}"), static=True)

        for plan in plan_records:
            rec.set_time("sim_time", duration=plan["result_t"])
            rec.log("plots/planning/worker", rr.Scalars(plan["worker_s"]))
            rec.log("plots/planning/total", rr.Scalars(plan["timings"]["total_s"]))
            rec.log("events/planning", rr.TextLog(
                f"{plan['id']}: {plan['operation']} {plan['status']}; "
                f"backup={plan['backup_status']}; {plan['worker_s']:.4f} s"))
        for transition in json.loads((case / "fsm.json").read_text()):
            rec.set_time("sim_time", duration=transition["time"])
            rec.log("events/fsm", rr.TextLog(
                f"{transition['from']} -> {transition['to']} ({transition['callback']})"))

        # Preserve the full control-rate curves and short backup transitions.
        # Only the 3D animation below is sampled at the requested display rate.
        actual_speed = np.linalg.norm(states[:, 7:10], axis=1)
        reference_speed = np.linalg.norm(refs[:, :, 1], axis=1)
        position_error = np.linalg.norm(states[:, :3] - refs[:, :, 0], axis=1)
        for index, now in enumerate(times):
            rec.set_time("sim_time", duration=float(now))
            rec.log("plots/speed/actual", rr.Scalars(float(actual_speed[index])))
            rec.log("plots/speed/reference", rr.Scalars(float(reference_speed[index])))
            rec.log("plots/error/position", rr.Scalars(float(position_error[index])))
            rec.log("plots/contact/force", rr.Scalars(float(contacts[index])))
            rec.log("plots/execution/on_backup", rr.Scalars(int(backup[index])))
            rec.log("plots/execution/fsm", rr.Scalars(int(fsm_states[index])))

        stride = max(1, int(round(1.0 / (fps * float(np.median(np.diff(times)))))))
        frames = sorted(set(range(0, len(times), stride)) | {len(times) - 1})
        event_index = 0
        for index in frames:
            now = float(times[index])
            while event_index < len(accepted) and accepted[event_index][0] <= now:
                activation, plan_index = accepted[event_index]
                plan_path = case / "plans" / f"{plan_index:04d}"
                points = sample_polynomial(plan_path / "command.npz")
                rec.set_time("sim_time", duration=activation)
                rec.log("world/active_plan", rr.LineStrips3D([points],
                                                               colors=[[195, 95, 244]], radii=0.05))
                if (plan_path / "backup.npz").exists():
                    points = sample_polynomial(plan_path / "backup.npz")
                    rec.log("world/backup_plan", rr.LineStrips3D([points],
                            colors=[[255, 208, 62]], radii=0.055))
                else:
                    rec.log("world/backup_plan", rr.Clear(recursive=True))
                event_index += 1
            rec.set_time("sim_time", duration=now)
            state = states[index]
            # Flight state uses [qw, qx, qy, qz]; Rerun uses [qx, qy, qz, qw].
            rec.log("world/drone", rr.Transform3D(
                translation=state[:3], rotation=rr.Quaternion(xyzw=state[[4, 5, 6, 3]])))
            rec.log("world/target", rr.Points3D([refs[index, :, 0]],
                                                 colors=[[182, 95, 248]], radii=0.085))
        rec.flush()

    return {
        "scene": scene_name, "seed": seed, "output": str(output),
        "bytes": output.stat().st_size, "frames": len(frames),
        "duration_sim_s": float(times[-1]), "obstacle_boxes": len(centers),
        "active_plans": len(accepted), "arrived": report["arrived"],
        "collision": report["collision"], "out_of_bounds": report["out_of_bounds"],
        "collision_free_arrival": report["collision_free_arrival"],
        "tracking_rmse_m": report["tracking_rmse_m"], "control_samples": len(times),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=("medium", "dense", "both"), default="both")
    parser.add_argument("--seeds", type=int, nargs="+",
                        help="Recorded seeds to convert; default: all saved flights per scene")
    parser.add_argument("--source", type=Path,
                        default=ROOT / "results/super_risp_validation_20261115")
    parser.add_argument("--output", type=Path, default=ROOT / "results/rerun_super_risp_20261115")
    parser.add_argument("--fps", type=float, default=12.)
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be positive")
    names = ("medium", "dense") if args.scene == "both" else (args.scene,)
    reports = []
    for name in names:
        seeds = args.seeds
        if seeds is None:
            seeds = [int(path.parent.name[4:])
                     for path in sorted((args.source / name).glob("seed*/flight.npz"))]
        if not seeds:
            raise FileNotFoundError(f"No recorded flights in {args.source / name}")
        reports.extend(record(name, seed, args.source,
                              args.output / f"{name}_seed{seed:03d}.rrd", args.fps)
                       for seed in seeds)
    (args.output / "summary.json").write_text(json.dumps(reports, indent=2) + "\n")
    for item in reports:
        print(json.dumps(item, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
