"""Compare the compiled LiDAR with its unchanged NumPy source from git.

The baseline exists only in this offline benchmark. Runtime has one traversal.
Actual flight sensor poses are replayed without changing rays or endpoint order.
"""
from pathlib import Path
import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from time import perf_counter

import numpy as np
from scipy.spatial.transform import Rotation

from isaac_pathcover.simulation.scene import Scene

ROOT = Path(__file__).resolve().parents[3]
BASE_COMMIT = "ded657cf1856393def3fce237fe3c4f94c4975f8"
SCENE_FILE = "src/isaac_pathcover/simulation/scene.py"


def original_lidar():
    source = subprocess.check_output(
        ["git", "show", f"{BASE_COMMIT}:{SCENE_FILE}"], cwd=ROOT, text=True)
    scene = next(node for node in ast.parse(source).body if isinstance(node, ast.ClassDef))
    method = next(node for node in scene.body if isinstance(node, ast.FunctionDef) and node.name == "lidar")
    namespace = {"np": np, "Rotation": Rotation}
    exec(compile(ast.Module(body=[method], type_ignores=[]), SCENE_FILE, "exec"), namespace)
    return namespace["lidar"], hashlib.sha256(source.encode()).hexdigest()


def quantiles(values):
    return {"p50_s": float(np.median(values)), "p95_s": float(np.quantile(values, .95))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path,
                        default=ROOT / "results/super_risp_development_20261115/newton_timed")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/super_risp_development_20261115/lidar_jit")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--write-fixtures", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    baseline, source_sha = original_lidar()
    report = {"baseline_commit": BASE_COMMIT, "baseline_source_sha256": source_sha,
              "kernel_sha256": hashlib.sha256((ROOT / "src/isaac_pathcover/simulation/lidar.py").read_bytes()).hexdigest(),
              "input_root": str(args.input_root), "repeats_per_frame": args.repeats,
              "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                          "numba": importlib.metadata.version("numba"), "machine": platform.machine(),
                          "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
                          "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS")},
              "timing_protocol": "Serial alternating NumPy/compiled calls per recorded pose; preparation and endpoint construction included; compilation reported separately.",
              "scenes": {}}
    fixtures = {"baseline_commit": BASE_COMMIT, "baseline_source_sha256": source_sha, "scenes": {}}
    arrays = {}
    for name in ("medium", "dense"):
        scene = Scene(name)
        path = args.input_root / name / "seed000/mapping.json"
        frames = json.loads(path.read_text())
        selected = set(np.unique([0, 1, 2, len(frames) // 3, 2 * len(frames) // 3, len(frames) - 1]).tolist())
        started = perf_counter()
        scene.lidar(scene.start, [1., 0., 0., 0.])
        warmup = perf_counter() - started
        rows, fixtures["scenes"][name] = [], []
        for index, frame in enumerate(frames):
            position, quaternion = frame["sensor_position"], frame["sensor_quaternion_wxyz"]
            before, after = [], []
            for _ in range(args.repeats):
                started = perf_counter()
                expected_origin, expected = baseline(scene, position, quaternion)
                before.append(perf_counter() - started)
                started = perf_counter()
                origin, points = scene.lidar(position, quaternion)
                after.append(perf_counter() - started)
                np.testing.assert_array_equal(origin, expected_origin)
                np.testing.assert_array_equal(points, expected)
                assert origin.tobytes() == expected_origin.tobytes()
                assert points.tobytes() == expected.tobytes()
            rows.append({"mapping_index": index, "sim_time_s": frame["t"], "points": len(points),
                         "numpy_s": before, "compiled_s": after, "exact_endpoints": True})
            if index in selected:
                key = f"{name}_{index:03d}"
                arrays[key + "_origin"], arrays[key + "_points"] = expected_origin, expected
                fixtures["scenes"][name].append({"key": key, "mapping_index": index,
                    "sim_time_s": frame["t"], "position": position, "quaternion_wxyz": quaternion})
        before = [v for row in rows for v in row["numpy_s"]]
        after = [v for row in rows for v in row["compiled_s"]]
        summary = {"frames": len(rows), "mapping_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                   "first_call_including_compile_or_cache_load_s": warmup,
                   "numpy": quantiles(before), "compiled": quantiles(after),
                   "speedup_p50": float(np.median(before) / np.median(after)),
                   "all_origins_and_endpoints_bitwise_equal": True, "rows": rows}
        report["scenes"][name] = summary
        print(json.dumps({"scene": name, **{k: v for k, v in summary.items() if k != "rows"}}), flush=True)
    (args.output / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    if args.write_fixtures:
        target = ROOT / "tests/fixtures"
        np.savez_compressed(target / "lidar_reference.npz", **arrays)
        (target / "lidar_reference.json").write_text(json.dumps(fixtures, indent=2) + "\n")


if __name__ == "__main__":
    main()
