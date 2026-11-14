"""Compile standalone upstream oracles and recompute mapping/search fixtures.

This is an explicit offline validation utility. Runtime modules and normal
pytest never import it, build extensions, or execute either C++ program.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from isaac_pathcover.mapping import NanoVoxMap, OccupancyInflation
from isaac_pathcover.planning.search import DistanceMapPlanner, JPSPlanner, VoxelGrid
from isaac_pathcover.simulation.scene import Scene


REVISION = "8723a05d1993bf0199a453b47dc6ee167d493c23"
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
FIXTURES = PROJECT / "tests" / "fixtures"
JPS_SOURCES = (
    "src/third_party/jps_lib/src/jps_planner/graph_search.cpp",
    "src/third_party/jps_lib/src/jps_planner/jps_planner.cpp",
    "src/third_party/jps_lib/src/distance_map_planner/graph_search.cpp",
    "src/third_party/jps_lib/src/distance_map_planner/distance_map_planner.cpp",
)
INFLATION_SOURCES = (
    "src/mapping/occupancy_inflation/src/point_cloud_inflater.cpp",
    "src/mapping/occupancy_inflation/src/voxel_bit_grid.cpp",
)
SOURCE_DIRS = (
    "src/mapping/nanovoxmap_ros/nanovoxmap",
    "src/mapping/occupancy_inflation",
    "src/third_party/jps_lib",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def compile_oracles(source, output, compiler, eigen):
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REVISION:
        raise ValueError(f"Expected PathCover {REVISION}, found {revision}")
    subprocess.run(
        ["git", "-C", str(source), "diff", "--quiet", "HEAD", "--", *SOURCE_DIRS],
        check=True,
    )
    output.mkdir(parents=True, exist_ok=True)
    common = [
        compiler, "-O2", "-std=c++17", f"-I{eigen}",
        f"-I{source / 'src/third_party/jps_lib/include'}",
    ]
    mapping = output / "mapping_search_oracle"
    scene = output / "scene_search_oracle"
    commands = [
        common + [
            f"-I{source / 'src/mapping/nanovoxmap_ros/nanovoxmap/include'}",
            f"-I{source / 'src/mapping/occupancy_inflation/include'}",
            str(HERE / "mapping_search.cpp"),
            *(str(source / path) for path in INFLATION_SOURCES + JPS_SOURCES),
            "-o", str(mapping),
        ],
        common + [
            str(HERE / "scene_search.cpp"),
            *(str(source / path) for path in JPS_SOURCES),
            "-o", str(scene),
        ],
    ]
    for command in commands:
        print("Compiling", Path(command[-1]).name, flush=True)
        subprocess.run(command, check=True)
    return mapping, scene, commands


def run_json(executable, *arguments):
    text = subprocess.check_output([str(executable), *map(str, arguments)], text=True)
    return json.loads(text[text.index("{"):])


def fingerprint(values):
    result = 14695981039346656037
    for value in values.ravel(order="F"):
        result = ((result ^ (int(value) & 255)) * 1099511628211) & ((1 << 64) - 1)
    return result


def scene_reference(name, executable, output):
    scene = Scene(name)
    mapper = NanoVoxMap()
    mapper.set_occupied(scene.point_cloud())
    cloud = OccupancyInflation().inflate(mapper.occupied_points())
    grid = VoxelGrid.from_points(cloud, scene.lower, scene.upper)
    input_path = output / f"{name}.txt"
    with input_path.open("w") as stream:
        stream.write(" ".join(map(str, grid.values.shape)) + "\n")
        stream.write(" ".join(map(str, [*grid.origin, grid.resolution, *scene.start, *scene.goal])) + "\n")
        stream.write(" ".join(map(str, grid.values.ravel(order="F"))) + "\n")
    result = run_json(executable, input_path)
    jps = JPSPlanner(grid, .25)
    begin = time.perf_counter()
    prior = jps.plan(scene.start, scene.goal)
    jps_seconds = time.perf_counter() - begin
    dmp = DistanceMapPlanner(grid)
    begin = time.perf_counter()
    potential = dmp.potential_map(scene.start)
    potential_seconds = time.perf_counter() - begin
    begin = time.perf_counter()
    path = dmp.plan(scene.start, scene.goal, np.asarray(result["jps_path"]))
    dmp_seconds = time.perf_counter() - begin
    result.update({
        "input_sha256": sha256(input_path),
        "grid_sha256": hashlib.sha256(grid.values.tobytes(order="F")).hexdigest(),
        "grid_shape": list(grid.values.shape),
        "origin": grid.origin.tolist(), "resolution": grid.resolution,
        "start": scene.start.tolist(), "goal": scene.goal.tolist(),
        "python": {
            "jps_cost": jps.cost, "dmp_cost": dmp.cost,
            "jps_s": jps_seconds, "dmp_s": dmp_seconds,
            "potential_s": potential_seconds,
            "potential_fnv64": fingerprint(potential.values),
            "jps_path_identical": bool(np.array_equal(prior, result["jps_path"])),
            "dmp_path_identical": bool(np.array_equal(path, result["dmp_path"])),
        },
    })
    return result


def compare_objectives(actual, expected):
    errors = {}
    for stage in ("jps", "dmp"):
        key = f"{stage}_cost"
        np.testing.assert_allclose(actual[key], expected[key], rtol=0, atol=1e-12)
        errors[f"{stage}_objective_abs_error"] = abs(actual[key] - expected[key])
        errors[f"{stage}_path_identical"] = bool(
            np.array_equal(actual[f"{stage}_path"], expected[f"{stage}_path"])
        )
    return errors


def compare_frozen(small, scenes):
    expected_small = json.loads((FIXTURES / "mapping_search.json").read_text())
    expected_scenes = json.loads((FIXTURES / "search_scene_reference.json").read_text())["scenes"]
    phases = 0
    for actual, expected in zip(small["mapping"], expected_small["mapping"], strict=True):
        for phase, reference in zip(actual["phases"], expected["phases"], strict=True):
            assert phase["voxel_fingerprint"] == reference["voxel_fingerprint"]
            assert phase["occupied_fingerprint"] == reference["occupied_fingerprint"]
            np.testing.assert_allclose(phase["occupied"], reference["occupied"], rtol=0, atol=2e-15)
            phases += 1
    for actual, expected in zip(small["inflation"], expected_small["inflation"], strict=True):
        assert actual["kernel_size"] == expected["kernel_size"]
        assert actual["solid_count"] == expected["solid_count"]
        np.testing.assert_array_equal(actual["points"], expected["points"])
    assert small["neighbours"] == expected_small["neighbours"]
    search_errors = []
    for actual, expected in zip(small["search"], expected_small["search"], strict=True):
        assert actual["jps_ok"] and actual["dmp_ok"]
        np.testing.assert_array_equal(actual["potential_map"], expected["potential_map"])
        search_errors.append(compare_objectives(actual, expected))
    scene_errors = {}
    for name, actual in scenes.items():
        expected = expected_scenes[name]
        assert actual["input_sha256"] == expected["input_sha256"]
        assert actual["grid_sha256"] == expected["grid_sha256"]
        assert actual["potential_fnv64"] == expected["potential_fnv64"]
        assert actual["python"]["potential_fnv64"] == actual["potential_fnv64"]
        assert actual["jps_ok"] and actual["dmp_ok"]
        scene_errors[name] = compare_objectives(actual, expected)
        for stage in ("jps", "dmp"):
            np.testing.assert_allclose(actual["python"][f"{stage}_cost"], actual[f"{stage}_cost"], rtol=0, atol=1e-12)
    return {
        "mapping_phases_matched": phases,
        "inflation_configurations_matched": len(small["inflation"]),
        "jps_direction_tables_matched": len(small["neighbours"]),
        "small_search": search_errors, "scene_search": scene_errors,
        "timing_note": "Fresh timings are diagnostic; historical performance measurements are not overwritten.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pathcover", type=Path, required=True, help="Official PathCover checkout at the fixed revision")
    parser.add_argument("--output", type=Path, default=PROJECT / ".reference" / "mapping-recomputed")
    parser.add_argument("--cxx", default="g++")
    parser.add_argument("--eigen", type=Path, default=Path("/usr/include/eigen3"))
    args = parser.parse_args()
    source, output = args.pathcover.resolve(), args.output.resolve()
    if output == FIXTURES.resolve():
        raise ValueError("Choose a separate output directory; frozen fixtures include historical measurements")
    output.mkdir(parents=True, exist_ok=True)
    mapping_binary, scene_binary, commands = compile_oracles(source, output / "build", args.cxx, args.eigen)
    small = run_json(mapping_binary)
    scenes = {name: scene_reference(name, scene_binary, output) for name in ("medium", "dense")}
    dump_json(output / "mapping_search.json", small)
    dump_json(output / "search_scene_reference.json", {
        "source_commit": REVISION,
        "harness_sha256": sha256(HERE / "scene_search.cpp"),
        "source_parameters": {
            "JpsHeuristicWeight": .25, "DMP_heuristic": 0., "DMP_cost_weight": .1,
            "DMP_power": 1, "PotentialRadius": [1, 1, 1], "SearchRadius": [2, 2, 2],
        },
        "scenes": scenes,
    })
    comparison = compare_frozen(small, scenes)
    comparison.update({
        "source_commit": REVISION,
        "compiler": subprocess.check_output([args.cxx, "--version"], text=True).splitlines()[0],
        "compile_commands": commands,
        "driver_sha256": {name: sha256(HERE / name) for name in ("mapping_search.cpp", "scene_search.cpp")},
    })
    dump_json(output / "comparison.json", comparison)
    print(json.dumps(comparison, indent=2), flush=True)


if __name__ == "__main__":
    main()
