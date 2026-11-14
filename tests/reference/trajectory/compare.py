"""Rebuild offline C++ oracles and compare fixed inputs with native Python.

This script is deliberately outside normal pytest discovery and packaging.
It writes only to --output; the pinned upstream checkout and fixtures are read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT / "src"))

import numpy as np
import scipy

from isaac_pathcover.planning.geometry import box_bounds
from isaac_pathcover.trajectory import FlatnessMap, GCopter, MINCO, TrajectoryConfig

COMMIT = "8723a05d1993bf0199a453b47dc6ee167d493c23"
FIXTURES = PROJECT / "tests" / "fixtures"


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_error(actual, expected):
    actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
    if not finite:
        return {"finite": False, "max_abs": None, "max_scaled": None, "relative_l2": None}
    difference = np.abs(actual - expected)
    return {
        "finite": True,
        "max_abs": float(np.max(difference)),
        "max_scaled": float(np.max(difference / np.maximum(1.0, np.abs(expected)))),
        "relative_l2": float(np.linalg.norm(difference)
                             / max(1.0, np.linalg.norm(expected))),
    }


def objective_error(cost, gradient, reference):
    return {
        "cost": array_error(cost, reference["initial_cost"]),
        "gradient": array_error(gradient, reference["initial_gradient"]),
    }


def same_objective(errors, tolerance):
    return (errors["cost"]["finite"] and errors["gradient"]["finite"]
            and errors["cost"]["max_scaled"] < 1e-9
            and errors["gradient"]["max_scaled"] < tolerance)


def build_oracle(args, output):
    upstream = args.upstream.resolve()
    actual_commit = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if actual_commit != COMMIT:
        raise RuntimeError(f"expected upstream {COMMIT}, got {actual_commit}")
    subprocess.run(
        ["git", "-C", str(upstream), "diff", "--exit-code", "HEAD", "--",
         "src/trajectory_server/include"], check=True, capture_output=True, text=True)
    include = upstream / "src" / "trajectory_server" / "include"
    expected_hashes = read_json(FIXTURES / "trajectory_validation.json")["source_sha256"]
    hashes = {name: sha256(include / name) for name in expected_hashes}
    if hashes != expected_hashes:
        raise RuntimeError("upstream header SHA256 does not match the recorded oracle")
    source = Path(__file__).with_name("oracle.cpp")
    binary = output / "build" / "oracle"
    binary.parent.mkdir(parents=True, exist_ok=True)
    command = [args.cxx, "-std=c++17", "-O3", "-DPATHCOVER_DIM=3",
               f"-I{args.eigen_include.resolve()}", f"-I{include}",
               str(source), "-o", str(binary)]
    subprocess.run(command, check=True)
    compiler = subprocess.check_output([args.cxx, "--version"], text=True).splitlines()[0]
    metadata = {
        "upstream_commit": actual_commit, "source_header_sha256": hashes,
        "oracle_source_sha256": sha256(source), "runner_sha256": sha256(Path(__file__)),
        "compiler": compiler, "compile_command": command, "python": sys.version,
        "numpy": np.__version__, "scipy": scipy.__version__,
        "openblas_threads": 1, "omp_threads": 1,
    }
    return binary, metadata


def run_oracle(binary, output, name, mode, payload="", optimize=False):
    command = [str(binary), mode]
    if optimize:
        command.append("optimize")
    if payload:
        input_path = output / "inputs" / f"{name}.txt"
        input_path.parent.mkdir(parents=True, exist_ok=True)
        input_path.write_text(payload + "\n")
    raw = subprocess.check_output(command, input=payload, text=True)
    result = json.loads(raw)
    write_json(output / "cpp" / f"{name}.json", result)
    return result


def primitive_values():
    values = {}
    for order in (3, 4):
        start = np.zeros((3, order))
        end = start.copy()
        end[:, 0] = [3, 2, 1]
        model = MINCO(start, end, 3, order).set_parameters(
            [[1, .5, .3], [2, 1.5, .7]], [1.1, 1.4, 1.2])
        energy, grad_c, partial_t = model.energy_with_partials()
        grad_points, grad_t = model.propagate_gradient(grad_c, partial_t)
        values[f"minco{order}"] = {
            "coefficients": model.coefficients.reshape(-1, 3),
            "energy": energy, "grad_c": grad_c.reshape(-1, 3),
            "partial_grad_t": partial_t, "grad_points": grad_points.T, "grad_times": grad_t,
        }
    model = FlatnessMap((.752, 9.8066, .1, .15, .08, .001))
    forward = model.forward(np.array([.8, -.3, .2]), np.array([.5, .2, .8]),
                            np.array([.2, .7, -.4]), .3, -.2)
    reverse = model.backward(np.array([.2, -.1, .4]), np.array([.1, .3, -.2]), .7,
                             np.array([.3, -.6, .2, .1]), np.array([.4, -.2, .5]))
    keys = ["thrust", "quat", "omega", "grad_pos", "grad_vel", "grad_acc",
            "grad_jer", "grad_yaw", "grad_yaw_rate"]
    values["flatness"] = dict(zip(keys, [*forward, *reverse]))
    return values


def compare_primitives(binary, output):
    reference = run_oracle(binary, output, "primitives", "primitives")
    stored = read_json(FIXTURES / "trajectory_reference.json")["expected"]
    report, passed = {}, True
    for group, values in primitive_values().items():
        tolerance = 2e-14 if group == "flatness" else 2e-9
        report[group] = {}
        for key, value in values.items():
            python_error = array_error(value, reference[group][key])
            fixture_error = array_error(reference[group][key], stored[group][key])
            ok = all(error["finite"] and error["max_scaled"] < tolerance
                     for error in (python_error, fixture_error))
            report[group][key] = {
                "python_vs_cpp": python_error, "cpp_vs_stored_fixture": fixture_error,
                "passed": ok,
            }
            passed &= ok
    return {"passed": passed, "checks": report}


def fixture_cases(extra_paths):
    basic = read_json(FIXTURES / "trajectory_reference.json")
    start, end = np.zeros((3, 3)), np.zeros((3, 3))
    start[:, 0], end[:, 0] = [-2, 0, 1], [2, 0, 1]
    cases = {"basic": {
        "physical_parameters": TrajectoryConfig().physical_parameters,
        "start_state": start, "end_state": end,
        "corridor": [box_bounds([-3, -2, 0], [1, 2, 3]),
                     box_bounds([-1, -2, 0], [3, 2, 3])],
        "vertex_coordinates": basic["gcopter_vertex_coordinates"],
        "initial_x": basic["gcopter_initial_x"], "expected": basic["expected"]["gcopter"],
    }}
    for name in ("medium", "dense"):
        cases[name] = read_json(FIXTURES / f"trajectory_{name}_reference.json")
    replans = read_json(FIXTURES / "trajectory_replan_reference.json")["cases"]
    cases.update({f"{name}_replan": data for name, data in replans.items()})
    for path in extra_paths:
        cases[f"extra_{path.stem}"] = read_json(path)
    return cases


def solver_for(data):
    corridor = [(np.array(a), np.array(b)) for a, b in data["corridor"]]
    config = TrajectoryConfig(physical_parameters=tuple(data["physical_parameters"]))
    solver = GCopter(corridor, np.array(data["start_state"]), np.array(data["end_state"]), config)
    native_counts = [len(v) for v in solver.vpolys]
    # Fix the coordinate system, without changing the source objective.
    solver.vpolys = [np.array(v) for v in data["vertex_coordinates"]]
    solver.point_polys = [solver.vpolys[i] for i in solver.vpoly_indices]
    solver.initial_x = np.array(data["initial_x"])
    return solver, native_counts


def serialize(data, x):
    parts = [*map(str, data["physical_parameters"]),
             *map(str, np.asarray(data["start_state"]).ravel()),
             *map(str, np.asarray(data["end_state"]).ravel()), str(len(data["corridor"]))]
    for a, b in data["corridor"]:
        parts.extend([str(len(a)), *map(str, np.column_stack([a, b]).ravel())])
    parts.append(str(len(data["vertex_coordinates"])))
    for vertices in data["vertex_coordinates"]:
        parts.extend([str(len(vertices)), *map(str, np.asarray(vertices).ravel())])
    parts.extend([str(len(x)), *map(str, x)])
    return " ".join(parts)


def compare_case(binary, output, name, data, optimize):
    solver, native_counts = solver_for(data)
    points = {"initial": (data["initial_x"], data["expected"])}
    if "cpp_at_cpp_final_x" in data:
        points["cpp_final"] = (data["expected"]["x"], data["cpp_at_cpp_final_x"])
    else:
        points["cpp_final"] = (data["expected"]["x"], None)
    if "python_final_x" in data:
        points["python_final"] = (data["python_final_x"], data["cpp_at_python_final_x"])
    report = {"fixture_input_sha256": data.get("input_sha256"), "points": {}}
    passed = True
    for point, (x, stored) in points.items():
        x = np.array(x)
        payload = serialize(data, x)
        cpp = run_oracle(binary, output, f"{name}__{point}", "gcopter", payload)
        if not cpp["setup_success"]:
            raise RuntimeError(f"upstream corridor setup failed: {name}")
        cost, gradient = solver.cost_gradient(x)
        errors = objective_error(cost, gradient, cpp)
        tolerance = 1e-6 if name == "medium_replan" and point == "cpp_final" else 3e-8
        ok = same_objective(errors, tolerance)
        row = {
            "protocol_input_sha256": hashlib.sha256(payload.encode()).hexdigest(),
            "same_x_bitwise": bool(np.array_equal(
                np.asarray(payload.split()[-len(x):], dtype=float).view(np.uint64),
                x.view(np.uint64))),
            "python_vs_cpp": errors, "gradient_scaled_tolerance": tolerance, "passed": ok,
        }
        if stored is not None:
            fixture_errors = objective_error(
                cpp["initial_cost"], np.asarray(cpp["initial_gradient"]), stored)
            row["cpp_vs_stored_fixture"] = fixture_errors
            row["passed"] &= same_objective(fixture_errors, tolerance)
        if point == "initial":
            report["python_native_vertex_counts"] = native_counts
            report["cpp_native_vertex_counts"] = cpp["native_vertex_counts"]
            report["native_vertex_counts_match"] = native_counts == cpp["native_vertex_counts"]
        passed &= row["passed"] and row["same_x_bitwise"]
        report["points"][point] = row
    if optimize:
        cpp = run_oracle(binary, output, f"{name}__optimization", "gcopter",
                         serialize(data, solver.initial_x), optimize=True)
        python = solver.optimize()
        python_x = solver.last_result.x.copy()
        cross = run_oracle(binary, output, f"{name}__optimized_python_x", "gcopter",
                           serialize(data, python_x))
        cost, gradient = solver.cost_gradient(python_x)
        cpp_x_cost, cpp_x_gradient = solver.cost_gradient(np.asarray(cpp["x"]))
        cpp_cross = run_oracle(binary, output, f"{name}__optimized_cpp_x", "gcopter",
                               serialize(data, np.asarray(cpp["x"])))
        report["optimization"] = {
            "python_success": python.success, "python_status": python.status_code,
            "python_cost": python.cost if np.isfinite(python.cost) else None,
            "python_iterations": python.iterations, "python_time_s": python.elapsed_s,
            "cpp_status": cpp["status"], "cpp_cost": cpp["cost"],
            "cpp_iterations": cpp["iterations"], "cpp_time_s": cpp["elapsed_s"],
            "python_final_x_cross_errors": objective_error(cost, gradient, cross),
            "cpp_final_x_cross_errors": objective_error(cpp_x_cost, cpp_x_gradient, cpp_cross),
            "interpretation": "Final nonconvex costs need not match; success is not feasibility.",
        }
    report["passed"] = passed
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, type=Path,
                        help="PathCover checkout at the fixed upstream commit")
    parser.add_argument("--output", type=Path, default=PROJECT / ".reference" / "trajectory_reproduce")
    parser.add_argument("--cxx", default="g++")
    parser.add_argument("--eigen-include", type=Path, default=Path("/usr/include/eigen3"))
    parser.add_argument("--optimize", action="store_true",
                        help="also rerun source and Python L-BFGS from identical initial variables")
    parser.add_argument("--extra-fixture", type=Path, action="append", default=[],
                        help="additional scene-format JSON with corridor, states, coordinates and x")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    binary, metadata = build_oracle(args, output)
    report = {"metadata": metadata, "primitives": compare_primitives(binary, output), "cases": {}}
    for name, data in fixture_cases(args.extra_fixture).items():
        row = compare_case(binary, output, name, data, args.optimize)
        report["cases"][name] = row
        print(f"{name}: {'PASS' if row['passed'] else 'FAIL'}", flush=True)
    report["passed"] = report["primitives"]["passed"] and all(
        row["passed"] for row in report["cases"].values())
    write_json(output / "report.json", report)
    print(f"Report: {output / 'report.json'}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

