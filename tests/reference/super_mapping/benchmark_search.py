"""Time one frozen Newton search input without advancing its source clock."""
from pathlib import Path
import argparse
import hashlib
import json
from time import perf_counter

import numpy as np

from isaac_pathcover.mapping import ROGMap
from isaac_pathcover.planning.search import AStar
from isaac_pathcover.planning import search_kernel


parser = argparse.ArgumentParser()
parser.add_argument("input", type=Path)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--repetitions", type=int, default=3)
args = parser.parse_args()
data = np.load(args.input / "input.npz")
expected = np.load(args.input / "expected_path.npz")
mapper = ROGMap(data["lower"], data["upper"])
np.testing.assert_array_equal(mapper.origin, data["origin"])
mapper.values[:] = data["values"]
mapper.inflated_count[:] = data["inflated_count"]
astar = AStar(mapper, clock=lambda: 0., clock_buffer=np.zeros(1))
astar.set_fine_inf_neighbors(4)
escape = astar.escape_path_search(data["start"])
np.testing.assert_array_equal(escape.path, expected["escape_path"])
start = escape.path[-1] if escape.success else data["start"]
measurements = []
for _ in range(args.repetitions):
    started = perf_counter()
    result = astar.search(start, data["goal"], horizon=float(data["horizon"]))
    elapsed = perf_counter() - started
    np.testing.assert_array_equal(result.path, expected["path"])
    assert result.expanded == int(expected["expanded"])
    assert result.cost == float(expected["cost"])
    measurements.append(elapsed)
report = {
    "input": str(args.input),
    "input_sha256": hashlib.sha256((args.input / "input.npz").read_bytes()).hexdigest(),
    "kernel_sha256": hashlib.sha256(Path(search_kernel.__file__).read_bytes()).hexdigest(),
    "status": result.status, "expanded": result.expanded, "cost": result.cost,
    "path_points": len(result.path), "same_path_and_expansions": True,
    "wall_s": measurements,
    "clock": "fixed zero float64 buffer; source timeout check remains at every expansion",
}
args.output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
