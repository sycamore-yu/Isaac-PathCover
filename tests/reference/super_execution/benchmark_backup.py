"""Reproduce the recorded medium backup frontend without solving trajectories."""
from argparse import ArgumentParser
import copy
import cProfile
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from isaac_pathcover.control.execution import ExpTraj
from isaac_pathcover.planning.pipeline import Planner
from isaac_pathcover.simulation.scene import Scene
from isaac_pathcover.trajectory.minco import PiecewisePolynomial


parser = ArgumentParser()
parser.add_argument("input", type=Path)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--scan", action="store_true")
args = parser.parse_args()
exp_input = json.loads((args.input / "exp_input.json").read_text())
backup_input = json.loads((args.input / "backup_input.json").read_text())
planner = Planner(Scene("medium"), seed=0, clock=lambda: 0.)
planner.set_robot_state(np.column_stack((planner.scene.start, np.zeros((3, 3)))))
if args.scan:
    origin, points = planner.scene.lidar(planner.scene.start, np.array([1., 0., 0., 0.]))
    planner.update_map(origin, points)
corridor = planner.corridor_generator.search_polytope_on_path(np.asarray(exp_input["guide_path"]))
planner.shifted_start = corridor.shifted_start
rng_state = copy.deepcopy(planner.corridor_generator.rng.bit_generator.state)
captured = {}


def capture(trajectory, t0, te, start, end, duration, polytope):
    captured.update(t0=t0, te=te, heu_ts=start, heu_end=end.tolist(),
                    heu_duration=duration, corridor=polytope.planes.tolist())
    return SimpleNamespace(success=False)


planner.backup_optimizer.optimize = capture
trajectory = PiecewisePolynomial(np.asarray(backup_input["durations"]),
                                 np.asarray(backup_input["coefficients"]))
exp = ExpTraj(trajectory, None, 0.)
times = []
for _ in range(5):
    planner.corridor_generator.rng.bit_generator.state = copy.deepcopy(rng_state)
    started = perf_counter()
    planner._generate_backup(exp)
    times.append(perf_counter() - started)
differences = {}
for key, value in captured.items():
    actual, original = np.asarray(value), np.asarray(backup_input[key])
    differences[key] = (float(np.max(np.abs(actual - original))) if actual.shape == original.shape
                        else {"actual_shape": actual.shape, "original_shape": original.shape})
args.output.parent.mkdir(parents=True, exist_ok=True)
profile = cProfile.Profile()
planner.corridor_generator.rng.bit_generator.state = copy.deepcopy(rng_state)
profile.runcall(planner._generate_backup, exp)
profile.dump_stats(str(args.output.with_suffix(".pstats")))
record = {"input_directory": str(args.input), "initial_scan": args.scan,
          "timings_s": times, "p50_s": float(np.median(times)), "p95_s": float(np.percentile(times, 95)),
          "captured": captured, "differences": differences}
args.output.write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps({k: v for k, v in record.items() if k != "captured"}, indent=2))
