"""Fixed-input comparisons with unchanged SUPER C++ ROG-Map."""
from pathlib import Path
import json

import numpy as np
import pytest

from isaac_pathcover.mapping import GridType, ROGMap, ROGMapConfig, ray_indices


PRELOAD = np.array([[-1.15, .25, 1.05], [.55, .05, 1.05], [.55, .05, 1.05], [1.05, .95, 1.55]])
SCAN = np.array([[1.95, .05, 1.05], [1.95, .95, 1.55], [-.95, -.95, .55]])
ORIGIN = np.array([-1.95, .05, 1.05])
RAYS = [([-.05, .05, 1.05], [1.05, 1.05, 2.05]), ([-.3, .3, 1.0], [-.3, .3, 1.0]),
        ([-.51, -.21, .71], [1.72, .51, 1.31]), ([1.72, .51, 1.31], [-.51, -.21, .71])]


def make_map(raycasting=False):
    return ROGMap([-4, -4, -2], [4, 4, 4], ROGMapConfig(
        resolution=.1, inflation_resolution=.2, inflation_step=1,
        virtual_ground_height=-2.0, virtual_ceil_height=4.0, raycasting=raycasting))


@pytest.mark.parametrize("mode", ["ray", "hits"])
def test_rog_map_original_queries_updates_and_raycaster(mode):
    fixture = json.loads((Path(__file__).parent / "fixtures/super_mapping_reference.json").read_text())
    assert fixture["super_commit"].startswith("2ad3419")
    expected = fixture["cases"][mode]
    rows = np.asarray(expected["queries"])
    mapper = make_map(mode == "ray")
    for phase in range(3):
        if phase == 1:
            mapper.set_occupied(PRELOAD)
        if phase == 2:
            for _ in range(4):
                mapper.update_map(ORIGIN, SCAN)
        selected = rows[rows[:, 0] == phase]
        points = selected[:, 1:4]
        np.testing.assert_array_equal(mapper.log_odds(points), selected[:, 4])
        np.testing.assert_array_equal(mapper.get_grid_type(points), selected[:, 5])
        np.testing.assert_array_equal(mapper.get_inf_grid_type(points), selected[:, 6])
        np.testing.assert_array_equal(mapper.is_occupied(points), selected[:, 7])
        np.testing.assert_array_equal(mapper.is_occupied_inflate(points), selected[:, 8])
        np.testing.assert_allclose(mapper.box_search([-2, -1, 0], [2, 1, 2]),
                                   np.asarray(expected["boxes"][phase]).reshape(-1, 3), atol=2e-15)
    for i, (start, end) in enumerate(RAYS):
        np.testing.assert_allclose(mapper.grid_to_world(ray_indices(start, end, .1)),
                                   np.asarray(expected["rays"][i]).reshape(-1, 3), atol=2e-15)
        assert bool(mapper.is_line_free(start, end)) == bool(expected["lines"][i][1])
        assert bool(mapper.is_line_free(start, end, inflated=True)) == bool(expected["lines"][i][2])


def test_incremental_inflation_removes_only_coverage_of_cleared_cell():
    mapper = make_map()
    points = np.array([[.05, .05, 1.05], [.25, .05, 1.05]])
    mapper.set_occupied(points)
    before = mapper.inflated_count.copy()
    mapper.set_free(points[:1])
    remaining = make_map()
    remaining.set_occupied(points[1:])
    np.testing.assert_array_equal(mapper.inflated_count, remaining.inflated_count)
    assert np.max(before) == 2
    assert np.max(mapper.inflated_count) == 1


def test_rog_hit_priority_and_zero_length_ray():
    mapper = make_map(True)
    mapper.update_map(ORIGIN, np.vstack((SCAN, PRELOAD)))
    assert mapper.get_grid_type(PRELOAD[1]) == GridType.OCCUPIED
    assert mapper.is_line_free([0, 0, 1], [0, 0, 1], neighbors=np.array([[0, 0, 0]]))


def test_sliding_retains_overlap_and_clears_departed_cells():
    mapper = make_map()
    mapper.set_occupied([[.05, .05, 1.05], [-3.95, .05, 1.05]])
    mapper.slide([1.0, 0.0, 1.0])
    assert mapper.is_occupied([.05, .05, 1.05])
    assert not mapper.inside([-3.95, .05, 1.05])
    assert mapper.get_grid_type([4.55, .05, 1.05]) == GridType.UNKNOWN


def test_optional_esdf_and_frontier_against_original_cpp():
    fixture = json.loads((Path(__file__).parent / "fixtures/super_mapping_reference.json").read_text())
    expected = fixture["cases"]["features"]
    mapper = ROGMap([-4, -4, -2], [4, 4, 4], ROGMapConfig(
        resolution=.1, inflation_resolution=.2, inflation_step=1,
        virtual_ground_height=-2.0, virtual_ceil_height=4.0, raycasting=True,
        frontier_extraction=True, esdf_enabled=True, esdf_resolution=.2,
        esdf_local_update_box=(3.2, 3.2, 2.4)))
    block = np.array([[x*.2+.05, y*.2+.05, z*.2+.05]
                      for x in range(-2, 3) for y in range(-2, 3) for z in range(3, 8)])
    mapper.esdf.update([0, 0, 1])
    occupancy = np.asarray(expected["queries"])
    rows = np.asarray(expected["esdf"])
    frontier = np.asarray(expected["frontier"])
    for phase in range(4):
        if phase == 1:
            mapper.set_occupied(PRELOAD)
            mapper.set_occupied(block)
            mapper.esdf.update([0, 0, 1])
        if phase == 2:
            for _ in range(4):
                mapper.update_map(ORIGIN, SCAN)
        if phase == 3:
            mapper.set_free(block.astype(np.float32).astype(float))
            mapper.set_occupied([ORIGIN])
            mapper.esdf.update([0, 0, 1])
        selected_occupancy = occupancy[occupancy[:, 0] == phase]
        query = selected_occupancy[:, 1:4]
        np.testing.assert_array_equal(mapper.log_odds(query), selected_occupancy[:, 4])
        np.testing.assert_array_equal(mapper.get_grid_type(query), selected_occupancy[:, 5])
        np.testing.assert_array_equal(mapper.get_inf_grid_type(query), selected_occupancy[:, 6])
        selected = rows[rows[:, 0] == phase]
        points = selected[:, 1:4]
        actual = np.column_stack((mapper.esdf.get_distance(points), mapper.esdf.evaluate_edt(points),
                                  mapper.esdf.evaluate_first_grad(points), mapper.esdf.evaluate_second_grad(points)))
        np.testing.assert_allclose(actual, selected[:, 4:], rtol=3e-14, atol=3e-13)
        bounds = np.asarray(expected["esdf_bounds"])[phase, 1:].reshape(2, 3)
        np.testing.assert_array_equal(mapper.esdf.get_updated_bbox(), bounds)
        selected_frontier = frontier[frontier[:, 0] == phase]
        np.testing.assert_array_equal(mapper.is_frontier(selected_frontier[:, 1:4]), selected_frontier[:, 4])
        np.testing.assert_allclose(mapper.box_search([-3, -1, 0], [3, 1, 2], GridType.FRONTIER),
                                   np.asarray(expected["frontier_boxes"][phase]).reshape(-1, 3), atol=2e-15)


def test_frontier_and_esdf_update_after_sliding():
    mapper = ROGMap([-2, -2, -1], [2, 2, 3], ROGMapConfig(
        resolution=.1, inflation_resolution=.2, inflation_step=1,
        virtual_ground_height=-1.0, virtual_ceil_height=3.0,
        frontier_extraction=True, esdf_enabled=True, esdf_resolution=.2))
    mapper.set_occupied([[.05, .05, 1.05], [-1.95, .05, 1.05]])
    mapper.set_free([[.25, .05, 1.05], [-1.95, .25, 1.05]])
    assert mapper.is_frontier([.35, .05, 1.05])
    mapper.slide([1, 0, 1])
    mapper.esdf.update([1, 0, 1])
    assert mapper.esdf.get_grid_type([.05, .05, 1.05]) == GridType.OCCUPIED
    assert mapper.is_frontier([.35, .05, 1.05])
    assert not mapper.is_frontier([-1.85, .25, 1.05])
    assert mapper.free_neighbor_count.min() == 0


def test_search_classification_cache_expires_after_map_update():
    from isaac_pathcover.planning.search import AStar

    mapper = make_map()
    obstacle = [[.05, .05, 1.05]]
    mapper.set_occupied(obstacle)
    astar = AStar(mapper, map_voxel_num=(81, 81, 61), clock=lambda: 0.)
    first = astar.search([-1.75, .05, 1.05], [1.65, .05, 1.05])
    assert first.status == "REACH_GOAL"
    mapper.set_free(obstacle)
    second = astar.search([-1.75, .05, 1.05], [1.65, .05, 1.05])
    assert second.status == "REACH_GOAL"
    assert second.cost < first.cost
    np.testing.assert_allclose(second.path[:, 1:], np.tile([.1, 1.1], (len(second.path), 1)))
    np.testing.assert_allclose(second.cost, 3.4)
