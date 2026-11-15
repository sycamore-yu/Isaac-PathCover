"""Recompute the point RISP fixture against the original PathCover header."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from time import perf_counter

import numpy as np
from scipy.optimize import linear_sum_assignment

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
    actual = np.column_stack((a, b)) / np.linalg.norm(a, axis=1)[:, None]
    ref_a = np.asarray(expected["A"])
    reference = np.column_stack((ref_a, expected["b"])) / np.linalg.norm(ref_a, axis=1)[:, None]
    if actual.shape != reference.shape:
        return None
    distances = np.max(np.abs(actual[:, None] - reference[None]), axis=2)
    rows, columns = linear_sum_assignment(distances)
    return float(distances[rows, columns].max())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--oracle", required=True, type=Path)
    parser.add_argument("--fixture", type=Path, default=REPOSITORY / "tests/fixtures/risp_corridor_reference.json")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    fixture = json.loads(args.fixture.read_text())
    commit = subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip()
    if commit != UPSTREAM_COMMIT or digest(args.upstream / HEADER_PATH) != fixture["header_sha256"]:
        raise ValueError("upstream commit/header do not match the saved reference")
    points, seed = np.asarray(fixture["points"]), np.asarray(fixture["path"][0])
    a, b = np.asarray(fixture["bounds"]["A"]), np.asarray(fixture["bounds"]["b"])
    rows = [f"{len(points)} 37"]
    rows.extend(" ".join(format(float(v), ".17g") for v in row) for row in np.vstack((points, seed)))
    rows.extend(" ".join(format(float(v), ".17g") for v in row) for row in np.column_stack((a, b)))
    process = subprocess.run([str(args.oracle.resolve())], input="\n".join(rows) + "\n",
                             text=True, capture_output=True, check=True)
    reference = json.loads(process.stdout)
    started = perf_counter()
    result = risp(points, seed, (a, b), rng=ReferenceRandom(reference["rand_values"]))
    elapsed = perf_counter() - started
    error = plane_error(result.A, result.b, reference["risp"])
    saved_error = plane_error(result.A, result.b, fixture["expected"]["risp"])
    counts_equal = result.remaining_counts == reference["risp_counts"] == fixture["expected"]["risp_counts"]
    report = {"upstream_commit": commit, "header_sha256": digest(args.upstream / HEADER_PATH),
              "fixture_sha256": digest(args.fixture), "oracle_sha256": digest(args.oracle),
              "driver_sha256": digest(Path(__file__).with_name("oracle.cpp")),
              "normalized_plane_max_abs_error": error, "saved_fixture_max_abs_error": saved_error,
              "discard_counts_equal": counts_equal, "python_s": elapsed, "cpp_s": reference["wall_s"],
              "passed": counts_equal and error is not None and saved_error is not None
                        and max(error, saved_error) < 3e-13}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
