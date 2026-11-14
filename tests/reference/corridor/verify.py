"""Explicit offline C++/Python comparison; never imported by package or pytest."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter

import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment

from isaac_pathcover.planning.corridor import path_cover
from isaac_pathcover.planning.risp import risp


UPSTREAM_COMMIT = "8723a05d1993bf0199a453b47dc6ee167d493c23"
HEADER_PATH = Path("src/corridor_planning/include/planner/PathCover.hpp")
REPOSITORY = Path(__file__).resolve().parents[3]


class ReferenceRandom:
    def __init__(self, values):
        self.values = iter(values)

    def integers(self, high):
        return next(self.values) % high


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plane_error(a, b, expected):
    a, b = np.asarray(a), np.asarray(b)
    reference_a, reference_b = np.asarray(expected["A"]), np.asarray(expected["b"])
    actual = np.column_stack((a, b)) / np.linalg.norm(a, axis=1)[:, None]
    reference = np.column_stack((reference_a, reference_b)) / np.linalg.norm(reference_a, axis=1)[:, None]
    if actual.shape != reference.shape:
        return None
    distances = np.max(np.abs(actual[:, None] - reference[None]), axis=2)
    rows, columns = linear_sum_assignment(distances)
    return float(distances[rows, columns].max())


def compare(corridor, single, reference):
    same_regions = len(corridor.polytopes) == len(reference["polytopes"])
    errors = [plane_error(a, b, expected)
              for (a, b), expected in zip(corridor.polytopes, reference["polytopes"])]
    reference_seeds = np.asarray(reference["seeds"])
    return {
        "region_counts_equal": same_regions,
        "risp_normalized_plane_max_abs_error": plane_error(single.A, single.b, reference["risp"]),
        "risp_discard_counts_equal": single.remaining_counts == reference["risp_counts"],
        "normalized_plane_max_abs_error": max(errors) if same_regions and None not in errors else None,
        "seed_max_abs_error": (float(np.max(np.abs(corridor.seeds - reference_seeds)))
                               if corridor.seeds.shape == reference_seeds.shape else None),
        "discard_counts_equal": corridor.remaining_counts == reference["counts"],
    }


def passed(comparison):
    return all(comparison[key] for key in ("region_counts_equal", "discard_counts_equal", "risp_discard_counts_equal")) and all(
        comparison[key] is not None and comparison[key] < 3e-13 for key in (
            "normalized_plane_max_abs_error", "risp_normalized_plane_max_abs_error", "seed_max_abs_error"
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, type=Path, help="checkout of the fixed official commit")
    parser.add_argument("--oracle", required=True, type=Path, help="standalone executable built from oracle.cpp")
    parser.add_argument("--fixture", type=Path, default=REPOSITORY / "tests/fixtures/risp_corridor_reference.json")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repetitions", type=int, default=30)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    fixture = json.loads(args.fixture.read_text())
    commit = subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip()
    header_hash = digest(args.upstream / HEADER_PATH)
    if commit != UPSTREAM_COMMIT or header_hash != fixture["header_sha256"]:
        raise ValueError("upstream commit/header do not match the saved reference")
    points, path = np.asarray(fixture["points"]), np.asarray(fixture["path"])
    a, b = np.asarray(fixture["bounds"]["A"]), np.asarray(fixture["bounds"]["b"])
    if a.shape != (6, 3) or b.shape != (6,):
        raise ValueError("the standalone fixture driver expects six workspace planes")
    random_seed, horizon = 37, 6
    rows = [f"{len(points)} {len(path)} {horizon} {random_seed}"]
    rows.extend(" ".join(format(float(v), ".17g") for v in row)
                for row in np.concatenate((points, path)))
    rows.extend(" ".join(format(float(v), ".17g") for v in row)
                for row in np.column_stack((a, b)))
    payload = "\n".join(rows) + "\n"
    python_times, cpp_times, comparisons = [], [], []
    for _ in range(args.repetitions):
        process = subprocess.run([str(args.oracle.resolve())], input=payload, text=True,
                                 capture_output=True, check=True)
        reference = json.loads(process.stdout)
        rng = ReferenceRandom(reference["rand_values"])
        started = perf_counter()
        corridor = path_cover(points, path, (a, b), max_horizon=horizon, rng=rng)
        python_times.append(perf_counter() - started)
        cpp_times.append(reference["wall_s"])
        single = risp(points, path[0], (a, b), rng=ReferenceRandom(reference["rand_values"]))
        comparisons.append(compare(corridor, single, reference))
    saved_comparison = compare(corridor, single, fixture["expected"])
    same_saved_draws = reference["rand_values"] == fixture["expected"]["rand_values"]
    report = {
        "upstream_commit": commit, "header_sha256": header_hash,
        "fixture_sha256": digest(args.fixture), "oracle_binary_sha256": digest(args.oracle),
        "driver_sha256": digest(Path(__file__).with_name("oracle.cpp")),
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        "platform": platform.platform(), "libc": platform.libc_ver(),
        "random_seed": random_seed, "max_horizon": horizon, "points": len(points),
        "path_points": len(path), "regions": len(corridor.polytopes), "repetitions": args.repetitions,
        "python_times_s": python_times, "cpp_times_s": cpp_times,
        "python_p50_ms": float(np.median(python_times) * 1e3),
        "python_p95_ms": float(np.quantile(python_times, .95) * 1e3),
        "cpp_p50_ms": float(np.median(cpp_times) * 1e3),
        "cpp_p95_ms": float(np.quantile(cpp_times, .95) * 1e3),
        "same_saved_random_draws": same_saved_draws, "last_cpp_comparison": comparisons[-1],
        "saved_fixture_comparison": saved_comparison,
        "passed": all(map(passed, comparisons)) and passed(saved_comparison) and same_saved_draws,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
