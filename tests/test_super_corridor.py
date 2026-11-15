import numpy as np
import pytest

from isaac_pathcover.mapping import ROGMap, ROGMapConfig
from isaac_pathcover.planning.corridor import CorridorGenerator, Polytope, find_interior, simplify_sfc
from isaac_pathcover.planning.risp import risp, risp_segment
from isaac_pathcover.planning.geometry import box_bounds
from isaac_pathcover.planning.search import AStar


def make_map():
    return ROGMap([-5, -5, -2], [5, 5, 4], ROGMapConfig(
        resolution=.1, inflation_resolution=.2, inflation_step=1,
        virtual_ground_height=-2, virtual_ceil_height=4))


def test_risp_point_limit_matches_official_geometry():
    points = np.array([[1.0, .2, 0], [-1, -.1, .2], [.1, 1, .2], [0, -1, 0]])
    bounds = box_bounds([-2, -2, -2], [2, 2, 2])
    a = risp(points, np.zeros(3), bounds, rng=np.random.default_rng(7))
    b = risp_segment(points, np.zeros(3), np.zeros(3), bounds, rng=np.random.default_rng(7))
    np.testing.assert_array_equal(a.A, b.A)
    np.testing.assert_array_equal(a.b, b.b)


def test_risp_line_seed_keeps_segment_and_excludes_robot_balls():
    points = np.array([[-.4, .7, 1], [.8, .8, 1.4], [.4, -.8, 1], [-.9, -.3, 1.6]])
    start, end = np.array([-.5, 0, 1]), np.array([.5, 0, 1])
    result = risp_segment(points, start, end, box_bounds([-2, -2, 0], [2, 2, 2]), robot_radius=.2)
    assert np.all(result.A @ start <= result.b)
    assert np.all(result.A @ end <= result.b)
    distances = points @ result.A.T - result.b
    assert np.all(np.max(distances, axis=1) >= .2)


def test_super_horizon_search_corridor_and_backup_point():
    mapper = make_map()
    mapper.set_occupied([[0, 0, 1]])
    search = AStar(mapper, clock=lambda: 0.).search([-3, 1, 1], [3, 1, 1], horizon=2)
    assert search.status == "REACH_HORIZON"
    assert search.cost > 2
    generator = CorridorGenerator(mapper)
    corridor = generator.search_polytope_on_path(search.path)
    np.testing.assert_allclose(corridor.goal, search.path[-1])
    for first, second in zip(corridor.regions[:-1], corridor.regions[1:]):
        _, radius = find_interior(np.vstack((first.planes, second.planes)))
        assert radius > .01
    point = generator.generate_from_point(search.path[0])
    assert point.contains(search.path[0])


@pytest.mark.parametrize("occupied_prefix", [False, True])
def test_super_shifted_start_only_replaces_caller_sentinel_for_inflation(occupied_prefix):
    # super_planner.cpp:709 sets the sentinel; corridor_generator.cpp:80-85
    # overwrites it only after skipping an inflated occupied path prefix.
    mapper = make_map()
    path = np.array([[.15, .15, 1.05], [.55, .15, 1.05], [.95, .15, 1.05]])
    if occupied_prefix:
        mapper.set_occupied([[0., 0., 1.]])
    assert bool(mapper.is_occupied_inflate(path[0])) == occupied_prefix
    assert not mapper.is_occupied_inflate(path[1])
    corridor = CorridorGenerator(mapper).search_polytope_on_path(path)
    expected = path[1] if occupied_prefix else np.full(3, 9999.0)
    np.testing.assert_array_equal(corridor.shifted_start, expected)
    if occupied_prefix:
        assert corridor.regions[0].contains(path[0])
        assert corridor.regions[0].contains(path[1])


def test_super_sfc_simplification_selects_last_head_and_first_tail():
    regions = []
    for x in [0., .8, 1.6, 2.4, 3.2]:
        A, b = box_bounds([x - .6, -1, 0], [x + .6, 1, 2])
        regions.append(Polytope(np.column_stack((A, -b))))
    result = simplify_sfc(regions, [.4, 0, 1], [2.8, 0, 1])
    assert result[0] is regions[1]
    assert result[-1] is regions[3]
    raw = simplify_sfc([p.planes for p in regions], [.4, 0, 1], [2.8, 0, 1])
    np.testing.assert_array_equal(raw[0], regions[1].planes)
