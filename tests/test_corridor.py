"""PathCover sequence, obstacle separation and continuous path coverage."""

import numpy as np

from isaac_pathcover.planning.corridor import find_start_polytope, path_cover
from isaac_pathcover.planning.geometry import box_bounds, chebyshev_center

from test_risp_reference import REFERENCE, ReferenceRandom, plane_error


def fixture_corridor(**kwargs):
    case = REFERENCE
    return path_cover(
        case["points"], case["path"], (case["bounds"]["A"], case["bounds"]["b"]),
        rng=ReferenceRandom(case["expected"]["rand_values"]), **kwargs,
    )


def test_corridor_matches_official_intersections_and_planes():
    actual, expected = fixture_corridor(), REFERENCE["expected"]
    np.testing.assert_allclose(actual.seeds, expected["seeds"], atol=3e-13, rtol=0.0)
    assert actual.remaining_counts == expected["counts"]
    assert len(actual.polytopes) == len(expected["polytopes"])
    for (A, b), reference in zip(actual.polytopes, expected["polytopes"]):
        assert plane_error(A, b, reference) < 3e-13
    np.testing.assert_allclose(actual.goal, REFERENCE["path"][-1], atol=1e-13)


def test_horizon_uses_upstream_size_greater_than_rule_and_local_goal():
    corridor = fixture_corridor(max_horizon=1)
    assert len(corridor.polytopes) == 2
    np.testing.assert_allclose(corridor.goal, corridor.seeds[-2], atol=1e-13)
    assert np.linalg.norm(corridor.goal - REFERENCE["path"][-1]) > 1.0


def test_every_region_excludes_obstacles_and_overlaps_its_neighbor():
    corridor = fixture_corridor()
    points = np.asarray(REFERENCE["points"])
    for index, (A, b) in enumerate(corridor.polytopes):
        assert np.all(A @ corridor.seeds[index] < b)
        assert np.all(np.max(points @ A.T - b, axis=1) > 0.0)
        if index:
            old_A, old_b = corridor.polytopes[index - 1]
            _, radius = chebyshev_center(np.concatenate((old_A, A)), np.r_[old_b, b])
            assert radius > 1e-3


def test_union_covers_every_continuous_path_segment():
    corridor = fixture_corridor()
    path = np.asarray(REFERENCE["path"])
    for start, end in zip(path[:-1], path[1:]):
        intervals = []
        for A, b in corridor.polytopes:
            slack, slope = b - A @ start, A @ (end - start)
            lo, hi = 0.0, 1.0
            for s, d in zip(slack, slope):
                if abs(d) < 1e-12:
                    if s < -1e-10:
                        lo, hi = 1.0, 0.0
                        break
                elif d > 0.0:
                    hi = min(hi, s / d)
                else:
                    lo = max(lo, s / d)
            if lo <= hi + 1e-10:
                intervals.append((lo, hi))
        reached = 0.0
        for lo, hi in sorted(intervals):
            assert lo <= reached + 1e-10
            reached = max(reached, hi)
        assert reached >= 1.0 - 1e-10


def test_random_seed_reproducibility():
    case = REFERENCE
    args = (case["points"], case["path"], (case["bounds"]["A"], case["bounds"]["b"]))
    first, second = path_cover(*args, seed=12), path_cover(*args, seed=12)
    np.testing.assert_array_equal(first.seeds, second.seeds)
    assert first.remaining_counts == second.remaining_counts


def test_start_polytope_follows_source_containment_and_excess_ranking():
    polytopes = [box_bounds([x, -1.0, -1.0], [x + 2.0, 1.0, 1.0]) for x in [0.0, 1.0, 2.0]]
    assert find_start_polytope([1.5, 0.0, 0.0], polytopes) == 1
    assert find_start_polytope([1.5, 2.0, 0.0], polytopes) == 1
    assert find_start_polytope([3.5, 2.0, 0.0], polytopes) == 2
