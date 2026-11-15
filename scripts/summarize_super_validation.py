"""Summarize every completed SUPER flight and export aligned speed/timing figures.

Run from the repository root with the existing system plotting environment:
    /usr/bin/python3 scripts/summarize_super_validation.py

This reads completed flight artifacts only. It does not import the planner,
load its Numba cache, alter acceptance rules, or exclude unsuccessful runs.
"""
from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import numpy as np


def read_json(path):
    return json.loads(path.read_text())


def file_record(path, repository):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path.absolute()),
        "repository_relative_path": str(path.relative_to(repository)),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def statistics(values, unit):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return {
        "count": int(values.size),
        "nonfinite_count": int(values.size - finite.size),
        f"p50_{unit}": float(np.quantile(finite, .50)) if finite.size else None,
        f"p95_{unit}": float(np.quantile(finite, .95)) if finite.size else None,
        f"max_{unit}": float(finite.max()) if finite.size else None,
    }


def request_timings(plans):
    return {
        "count": len(plans),
        "planner_total": statistics([p["timings"]["total_s"] for p in plans], "s"),
        "cycle_including_sensor_and_delivery": statistics([p["cycle_wall_s"] for p in plans], "s"),
        "worker_including_sensor": statistics([p["worker_s"] for p in plans], "s"),
    }


def compact_pause(interval, index):
    result = {key: value for key, value in interval.items() if key != "concurrent_plans"}
    plans = interval["concurrent_plans"]
    result["source_timeline_interval_index"] = index
    result["concurrent_plan_count"] = len(plans)
    result["concurrent_plan_status_counts"] = dict(Counter(p["status"] for p in plans))
    result["concurrent_plan_ids"] = [p["id"] for p in plans]
    return result


def read_run(repository, directory):
    result = read_json(directory / "result.json")
    scene, seed = result["scene"], result["seed"]
    plans = read_json(directory / "planning.json")
    mappings = read_json(directory / "mapping.json")
    timeline = read_json(directory / "timeline.json")
    assert scene == directory.parent.name
    with np.load(directory / "flight.npz", allow_pickle=False) as trace:
        times = trace["time"]
        states = trace["states"]
        references = trace["references"]
        on_backup = trace["on_backup"]
    assert len(times) == len(states) == len(references) == len(on_backup)
    actual_speed = np.linalg.norm(states[:, 7:10], axis=1)
    reference_speed = np.linalg.norm(references[:, :, 1], axis=1)
    first_motion = timeline["first_motion_s"]
    motion_mask = times >= first_motion if first_motion is not None else np.zeros(len(times), bool)
    steps = np.diff(np.r_[0., times])
    threshold = timeline["diagnostic_definition"]["motion_onset_speed_m_s"]
    committed = [p for p in plans if p["handoff"]["accepted"]]
    exp_attempts = [p for p in plans if "exp_opt_s" in p["timings"]]
    cold_commits = sum(p["operation"] == "plan_from_rest" for p in committed)
    replan_commits = sum(p["operation"] == "replan_once" for p in committed)
    assert cold_commits == result["cold_start_commits"]
    assert replan_commits == result["replan_commits"]
    assert len(plans) == result["planning_attempts"]
    all_timing = request_timings(plans)
    if plans:
        np.testing.assert_allclose(all_timing["planner_total"]["p50_s"], result["planning_p50_s"],
                                   rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(all_timing["cycle_including_sensor_and_delivery"]["p95_s"],
                                   result["complete_cycle_p95_s"], rtol=1e-12, atol=1e-15)
    else:
        assert result["planning_p50_s"] is None and result["complete_cycle_p95_s"] is None
    modules = {
        key: statistics([p["timings"][key] for p in plans if key in p["timings"]], "s")
        for key in sorted({key for p in plans for key in p["timings"]})
    }
    raw_files = ["result.json", "flight.npz", "planning.json", "mapping.json",
                 "timeline.json", "fsm.json", "source_hashes.json"]
    summary = {
        "scene": scene,
        "seed": seed,
        "run_directory": str(directory.absolute()),
        "outcome": {key: result[key] for key in (
            "arrived", "collision", "out_of_bounds", "collision_free_arrival",
            "goal_error_m", "resolved_goal_error_m", "goal_resolution_shift_m",
            "arrival_rule", "finite", "collision_check_complete",
            "collision_physics_samples", "collision_unchecked_physics_samples",
            "out_of_bounds_physics_samples", "contact_force_max_N",
        )},
        "motion": {
            "duration_sim_s": result["duration_sim_s"],
            "wall_time_s": result["wall_time_s"],
            "planner_ready_sim_s": result["planner_ready_sim_s"],
            "first_motion_s": first_motion,
            "elapsed_from_first_motion_s": float(times[-1] - first_motion) if first_motion is not None else 0.,
            "time_above_motion_threshold_s": float(steps[actual_speed > threshold].sum()),
            "diagnostic_definition": timeline["diagnostic_definition"],
            "control_samples": len(times),
            "valid_state_samples": result["valid_state_samples"],
            "invalid_state_samples": result["invalid_state_samples"],
            "physics_max_speed_m_s": result["max_speed_m_s"],
        },
        "speed": {
            "all_control_samples": {
                "actual": statistics(actual_speed, "m_s"),
                "reference": statistics(reference_speed, "m_s"),
            },
            "from_first_motion_through_end": {
                "actual": statistics(actual_speed[motion_mask], "m_s"),
                "reference": statistics(reference_speed[motion_mask], "m_s"),
            },
        },
        "planning": {
            "attempts": len(plans),
            "reported_successes": result["planning_successes"],
            "reported_failures": result["planning_failures"],
            "commands_committed": len(committed),
            "cold_start_commits": cold_commits,
            "replan_commits": replan_commits,
            "requests_without_commit": result["plans_without_commit"],
            "status_counts": timeline["planning_status_counts"],
            "all_requests": all_timing,
            "requests_with_exp_optimization": request_timings(exp_attempts),
            "modules": modules,
            "initialization_s": result["mapping_initialization"],
            "job_running_at_episode_end": result["job_running_at_episode_end"],
            "submitted_jobs": result["planning_jobs_submitted"],
            "returned_jobs": result["planning_jobs_returned"],
        },
        "sensor": {
            "updates": len(mappings),
            "lidar": result["lidar"],
            "sensor": statistics([m["sensor_s"] for m in mappings], "s"),
            "map_fusion": statistics([m["fusion_s"] for m in mappings], "s"),
            "sensor_plus_map_fusion": statistics([m["sensor_s"] + m["fusion_s"] for m in mappings], "s"),
        },
        "backup": {
            "generation_status_counts": timeline["backup_status_counts"],
            "generated_success_count": sum(p["backup_status"] == "SUCCESS" for p in plans),
            "committed_with_backup": timeline["committed_with_backup"],
            "published_flag_entries": timeline["backup_entries"],
            "published_flag_intervals": timeline["backup_intervals"],
        },
        "pauses": {
            "count": len(timeline["intermediate_slow_intervals"]),
            "total_duration_s": timeline["intermediate_slow_total_s"],
            "intervals": [compact_pause(value, i) for i, value in enumerate(timeline["intermediate_slow_intervals"])],
        },
        "handoff": {
            "max_reference_jump_pvaj": timeline["reference_jump_max_pvaj"],
            "max_replan_reference_jump_pvaj": timeline["replan_reference_jump_max_pvaj"],
        },
        "execution": {key: result[key] for key in (
            "final_fsm_state", "control_steps_while_worker_busy", "physics_steps_while_worker_busy",
            "physics_progress_while_planning_sim_s", "max_progress_during_one_solve_sim_s",
            "time_contract", "reference_contract", "fsm_command_wall_period_s", "replan_wall_period_s",
        )},
        "scene_provenance": result["scene_provenance"],
        "resolved_goal_m": result["resolved_goal_m"],
        "runtime": result["runtime"],
        "source_sha256": read_json(directory / "source_hashes.json"),
        "raw_files": {name: file_record(directory / name, repository) for name in raw_files},
    }
    curves = {"times": times, "actual_speed": actual_speed, "reference_speed": reference_speed,
              "plans": plans, "timeline": timeline}
    return summary, curves


def plot_scene(scene, runs, curves, output_dir, page_index):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelcolor": "#30343b", "text.color": "#20252d",
        "axes.edgecolor": "#aab1ba", "grid.color": "#dfe3e8",
        "svg.fonttype": "none", "svg.hashsalt": "super-validation-20261115",
    })
    colors = {"actual": "#1263a0", "reference": "#da5c22",
              "total": "#16734b", "cycle": "#727986", "backup": "#ecb95a"}
    height = 1.7 + 3.6 * len(runs)
    figure = plt.figure(figsize=(14, height))
    grid = figure.add_gridspec(2 * len(runs), 1, height_ratios=[2.05, 1.] * len(runs), hspace=.57,
                              left=.075, right=.975, bottom=.55 / height, top=1. - 1.85 / height)
    seed_text = ", ".join(str(run["seed"]) for run in runs)
    figure.suptitle(f"SUPER + RISP / Newton — {scene} | seeds {seed_text}", x=.075, y=1. - .25 / height,
                   ha="left", fontsize=20, weight="bold")
    figure.text(.075, 1. - .68 / height,
                "All outcomes are retained. Speeds and planning wall times share each seed's simulation-time axis.",
                ha="left", fontsize=10.5, color="#505965")
    handles = [
        Line2D([0], [0], color=colors["actual"], lw=1.8, label="Actual speed"),
        Line2D([0], [0], color=colors["reference"], lw=1.5, ls="--", label="Reference speed"),
        Patch(facecolor=colors["backup"], alpha=.25, label="Published backup flag"),
        Patch(facecolor="none", edgecolor="#626872", hatch="////", label="Recorded pause"),
        Line2D([0], [0], color=colors["total"], lw=1.1, marker=".", ms=3, label="Planner total"),
        Line2D([0], [0], color=colors["cycle"], lw=1.1, marker=".", ms=3, label="Cycle incl. sensor / delivery"),
        Line2D([0], [0], color="#322745", lw=0, marker="o", ms=4, mfc="none",
               label="Request with Exp optimization"),
    ]
    figure.legend(handles=handles, loc="upper left", bbox_to_anchor=(.069, 1. - .85 / height),
                  ncol=4, frameon=False, columnspacing=1.6, handlelength=2.4, fontsize=9)
    for row, (run, data) in enumerate(zip(runs, curves)):
        speed_axis = figure.add_subplot(grid[2 * row])
        timing_axis = figure.add_subplot(grid[2 * row + 1], sharex=speed_axis)
        t = data["times"]
        outcome = run["outcome"]
        flag = lambda value: "yes" if value else "no"
        speed_axis.set_title(
            f"seed {run['seed']}  |  arrival: {flag(outcome['arrived'])}  |  collision: {flag(outcome['collision'])}"
            f"  |  outside bounds: {flag(outcome['out_of_bounds'])}  |  observed {t[-1]:.2f} s",
            loc="left", fontsize=10.5, pad=8, weight="bold")
        for interval in run["backup"]["published_flag_intervals"]:
            for axis in (speed_axis, timing_axis):
                axis.axvspan(interval["start_s"], interval["end_s"],
                             color=colors["backup"], alpha=.22, lw=0, zorder=0)
        for interval in run["pauses"]["intervals"]:
            speed_axis.axvspan(interval["start_s"], interval["end_s"],
                              facecolor="none", edgecolor="#737b85", hatch="////",
                              linewidth=.4, alpha=.45, zorder=1)
        speed_axis.plot(t, data["actual_speed"], color=colors["actual"], lw=1.5)
        speed_axis.plot(t, data["reference_speed"], color=colors["reference"], lw=1.2, ls="--")
        speed_axis.set_ylabel("Speed (m/s)")
        speed_axis.set_ylim(bottom=0.)
        speed_axis.tick_params(labelbottom=False)
        plans = data["plans"]
        result_times = np.asarray([p["result_t"] for p in plans])
        totals = np.asarray([p["timings"]["total_s"] for p in plans]) * 1000.
        cycles = np.asarray([p["cycle_wall_s"] for p in plans]) * 1000.
        has_exp = np.asarray(["exp_opt_s" in p["timings"] for p in plans], dtype=bool)
        timing_axis.plot(result_times, cycles, color=colors["cycle"], lw=.7,
                         marker=".", ms=2.3, alpha=.75)
        timing_axis.plot(result_times, totals, color=colors["total"], lw=.8,
                         marker=".", ms=2.6, alpha=.95)
        timing_axis.scatter(result_times[has_exp], totals[has_exp], s=11,
                            facecolors="none", edgecolors="#322745", linewidths=.55, zorder=4)
        timing_axis.set_ylabel("Wall time (ms)")
        timing_axis.set_xlabel("Simulation time (s)")
        timing_axis.set_ylim(bottom=0.)
        for axis in (speed_axis, timing_axis):
            axis.set_xlim(0., float(t[-1]))
            axis.grid(axis="both", linewidth=.55, alpha=.7)
            axis.set_axisbelow(True)
    suffix = "" if page_index == 0 else f"_seeds{runs[0]['seed']:03d}_{runs[-1]['seed']:03d}"
    svg_path = output_dir / f"super_validation_{scene}_speed{suffix}.svg"
    png_path = output_dir / f"super_validation_{scene}_speed{suffix}.png"
    figure.savefig(svg_path, metadata={"Date": None}, facecolor="white")
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
    figure.savefig(png_path, dpi=160, facecolor="white")
    plt.close(figure)
    return [svg_path, png_path], matplotlib.__version__


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/super_risp_validation_20261115"))
    parser.add_argument("--output", type=Path, default=Path("docs/data"))
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    input_root = (args.input if args.input.is_absolute() else repository / args.input).resolve()
    output_dir = (args.output if args.output.is_absolute() else repository / args.output).resolve()
    run_directories = sorted(path.parent for path in input_root.glob("*/seed*/result.json"))
    if not run_directories:
        raise FileNotFoundError(f"No completed flight results under {input_root}")
    loaded = [read_run(repository, directory) for directory in run_directories]
    runs = [run for run, _ in loaded]
    scenes = sorted({run["scene"] for run in runs})
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = []
    for scene in scenes:
        selected = [(run, data) for run, data in loaded if run["scene"] == scene]
        for page_index, start in enumerate(range(0, len(selected), 3)):
            page = selected[start:start + 3]
            paths, matplotlib_version = plot_scene(scene, [item[0] for item in page],
                                                   [item[1] for item in page], output_dir, page_index)
            figures.extend(file_record(path, repository) for path in paths)
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input_root": str(input_root.absolute()),
        "selection": {"scenes": scenes,
                      "seeds_by_scene": {scene: [run["seed"] for run in runs if run["scene"] == scene]
                                         for scene in scenes},
                      "all_runs_retained": True, "success_filter": False},
        "definitions": {
            "arrival_collision_bounds": "Copied independently from each original result.json; no combined pass rule is added.",
            "speed": "Euclidean norm of states[:,7:10] and references[:,:,1] at the recorded control timestamps.",
            "motion_start": "First recorded actual speed above the existing 0.5 m/s diagnostic threshold.",
            "elapsed_from_first_motion": "Run end minus first-motion timestamp; includes subsequent pauses and settling.",
            "time_above_motion_threshold": "Recorded step durations whose actual speed exceeds 0.5 m/s.",
            "pause": "Existing timeline.json definition: after motion starts, actual speed below 0.2 m/s for at least 0.2 s, more than 1 m from the resolved goal.",
            "backup_intervals": "Intervals of the recorded published on_backup flag. The flag can remain true after command duration when no new command is published; it does not imply an active moving backup segment.",
            "planner_total": "timings.total_s for all returned requests, including failures and requests without a new trajectory.",
            "cycle_including_sensor_and_delivery": "cycle_wall_s from submission to receipt; includes sensor and map update when requested, worker execution, IPC and result polling.",
            "worker_including_sensor": "worker_s includes requested sensor and map work, but excludes parent submission and delivery overhead.",
            "requests_with_exp_optimization": "All requests containing timings.exp_opt_s, regardless of success; statistics use each selected request's complete total_s / cycle_wall_s.",
            "module_statistics": "Only requests containing the named module contribute; each statistic records its sample count.",
            "quantiles": "NumPy linear sample quantiles; nonfinite values are counted explicitly and excluded from finite quantiles.",
            "plots": "Planning values are plotted at their recorded result_t; each seed's speed and planning panels share simulation time. No solver deadline line is added.",
            "raw_file_manifest": "Hashes cover the seven original files used for outcomes, intervals, metrics, curves and per-run source provenance; per-plan geometry files are not duplicated in this summary.",
            "source_hashes": "Each run's own source_hashes.json is retained; scene-level historical snapshots are not used.",
        },
        "counts_by_scene": {
            scene: {"runs": sum(run["scene"] == scene for run in runs), **{
                key: sum(run["outcome"][key] for run in runs if run["scene"] == scene)
                for key in ("arrived", "collision", "out_of_bounds")
            }}
            for scene in scenes
        },
        "aggregation_environment": {"python": platform.python_version(), "numpy": np.__version__,
                                    "matplotlib": matplotlib_version},
        "aggregation_script": file_record(Path(__file__).resolve(), repository),
        "figures": figures,
        "runs": runs,
    }
    output = output_dir / "super_validation_flights.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"summary": str(output), "counts": report["counts_by_scene"],
                      "figures": [entry["path"] for entry in figures]}, indent=2))


if __name__ == "__main__":
    main()
