"""Reconstruct a recorded Newton search input and retain its numeric snapshot."""
from pathlib import Path
import argparse
import json
from time import perf_counter

import numpy as np
from scipy.spatial.transform import Rotation

from isaac_pathcover.mapping import ROGMap
from isaac_pathcover.planning.search import AStar
from isaac_pathcover.simulation.scene import Scene


parser = argparse.ArgumentParser()
parser.add_argument("flight", type=Path)
parser.add_argument("--until", type=float, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
scene = Scene(args.flight.parent.name)
mapper = ROGMap(scene.lower, scene.upper)
mapper.set_occupied(scene.point_cloud())
flight = np.load(args.flight / "flight.npz")
mapping = json.loads((args.flight / "mapping.json").read_text())
states, times = flight["states"], flight["time"]
sensor_origins = states[:, :3] + Rotation.from_quat(states[:, [4, 5, 6, 3]]).apply(np.array([0., 0., .25]))
scans, scan_origins, counts, replay = [], [], [], []
for entry in mapping:
    if entry["t"] > args.until:
        break
    if "sensor_position" in entry:
        body, quaternion = entry["sensor_position"], entry["sensor_quaternion_wxyz"]
        frame = -1
    else:
        distance = np.linalg.norm(sensor_origins - entry["sensor_origin"], axis=1)
        distance[times > entry["t"]] = np.inf
        frame = int(np.argmin(distance))
        body, quaternion = states[frame, :3], states[frame, 3:7]
    origin, points = scene.lidar(body, quaternion)
    update = mapper.update_map(origin, points)
    replay.append({"t": entry["t"], "frame": frame,
                   "origin_error_m": float(np.linalg.norm(origin-entry["sensor_origin"])),
                   "points": len(points), "recorded_points": entry["points"],
                   "map_version": mapper.version, "recorded_map_version": entry["map_version"]})
    scans.append(points)
    scan_origins.append(origin)
    counts.append(len(points))
frame = int(np.argmin(abs(times-args.until)))
start = states[frame, :3]
goal = np.asarray(json.loads((args.flight / "result.json").read_text())["resolved_goal_m"])
args.output.mkdir(parents=True, exist_ok=True)
np.savez_compressed(args.output / "input.npz", lower=scene.lower, upper=scene.upper,
                    values=mapper.values, origin=mapper.origin, start=start, goal=goal,
                    inflated_count=mapper.inflated_count, scan_points=np.vstack(scans),
                    scan_origins=scan_origins, scan_counts=counts, horizon=7.)
astar = AStar(mapper, clock=lambda: 0., clock_buffer=np.zeros(1))
astar.set_fine_inf_neighbors(4)
begun = perf_counter()
escape = astar.escape_path_search(start)
escape_time = perf_counter()-begun
shifted = escape.path[-1] if escape.success else start
begun = perf_counter()
result = astar.search(shifted, goal, horizon=7.)
elapsed = perf_counter()-begun
np.savez_compressed(args.output / "expected_path.npz", escape_path=escape.path, path=result.path,
                    expanded=result.expanded, cost=result.cost)
summary = {"scene": scene.name, "input_flight": str(args.flight), "until": args.until,
           "scan_replay": replay, "start": start.tolist(), "goal": goal.tolist(),
           "escape": {"status": escape.status, "expanded": escape.expanded, "wall_s": escape_time},
           "search": {"status": result.status, "expanded": result.expanded,
                      "wall_s": elapsed, "cost": result.cost, "points": len(result.path)}}
(args.output / "replay.json").write_text(json.dumps(summary, indent=2)+"\n")
print(json.dumps(summary, indent=2))
