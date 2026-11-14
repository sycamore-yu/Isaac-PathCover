"""JPS neighbour pruning, original DMP potential and independent path cost."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from isaac_pathcover.planning.search import (VoxelGrid, JPSPlanner, DistanceMapPlanner,
                                            SearchFailure, jps_neighbours, plan_path)


FIXTURE = Path(__file__).parent / "fixtures" / "mapping_search.json"


def source_grid():
    values = np.zeros((20,18,10), dtype=np.int8)
    for y in range(2,16):
        for z in range(1,9):
            if not (7 <= y <= 9 and 4 <= z <= 5):
                values[9,y,z] = 100
    values[12:15,10:14,0:4] = 100
    return VoxelGrid(values, [-1.,-1.,0.], .25)


def test_all_jps_neighbour_tables_match_official():
    for expected in json.loads(FIXTURE.read_text())["neighbours"]:
        actual = jps_neighbours(tuple(expected["direction"]))
        for values, name in zip(actual, ("natural","forced","additions")):
            assert [list(row) for row in values] == expected[name]


@pytest.mark.parametrize("index", [0,1])
def test_jps_dmp_cpp_objective_and_voxel_legality(index):
    expected = json.loads(FIXTURE.read_text())["search"][index]
    grid = source_grid()
    start, goal = np.array([-.375,-.375,1.125]), np.array([3.375,2.875,1.125])
    jps = JPSPlanner(grid, expected["heuristic"])
    prior = jps.plan(start, goal)
    np.testing.assert_allclose(jps.cost, expected["jps_cost"], rtol=0, atol=1e-12)
    dmp = DistanceMapPlanner(grid)
    # Same prior path isolates DMP's contract from tied JPS shortest paths.
    path = dmp.plan(start, goal, np.asarray(expected["jps_path"]))
    np.testing.assert_array_equal(dmp.distance_grid.values.ravel(order="F"), expected["potential_map"])
    np.testing.assert_allclose(dmp.cost, expected["dmp_cost"], rtol=0, atol=1e-12)
    ids = grid.float_to_int(path)
    assert np.all(grid.values[tuple(ids.T)] == 0)
    assert np.max(np.abs(np.diff(ids, axis=0))) <= 1
    assert np.all(dmp.region[tuple(ids[1:].T)])
    edge_cost = np.linalg.norm(np.diff(ids,axis=0),axis=1) + .1*dmp.distance_grid.values[tuple(ids[1:].T)]
    np.testing.assert_allclose(edge_cost.sum()+dmp.distance_grid.values[tuple(ids[0])],dmp.cost,atol=1e-12)
    np.testing.assert_array_equal(path[0],start)
    np.testing.assert_array_equal(path[-1],goal)


def test_complete_search_inserts_exact_endpoint_states():
    grid = source_grid()
    start,goal = np.array([-.37,-.37,1.12]),np.array([3.37,2.87,1.12])
    result = plan_path(grid,start,goal)
    np.testing.assert_array_equal(result.path[0],start)
    np.testing.assert_array_equal(result.path[-1],goal)
    np.testing.assert_array_equal(result.path[1:-1],result.dmp_path)


@pytest.mark.parametrize("wall_value", [100, -1])
def test_sealed_wall_exhausts_jps_without_solver_substitution(wall_value):
    grid = VoxelGrid(np.zeros((12,12,12),dtype=np.int8),np.zeros(3),.25)
    grid.values[6,:,:] = wall_value
    with pytest.raises(SearchFailure,match="exhausted"):
        JPSPlanner(grid).plan([.625,1.125,1.125],[2.625,1.125,1.125])


@pytest.mark.parametrize("name", ["medium", "dense"])
def test_scene_point_cloud_to_search_matches_official(name):
    from isaac_pathcover.mapping import NanoVoxMap, OccupancyInflation
    from isaac_pathcover.simulation.scene import Scene

    expected = json.loads(FIXTURE.with_name("search_scene_reference.json").read_text())["scenes"][name]
    scene = Scene(name)
    mapper = NanoVoxMap()
    mapper.set_occupied(scene.point_cloud())
    shell = OccupancyInflation().inflate(mapper.occupied_points())
    grid = VoxelGrid.from_points(shell, scene.lower, scene.upper)
    assert hashlib.sha256(grid.values.tobytes(order="F")).hexdigest() == expected["grid_sha256"]
    jps = JPSPlanner(grid, .25)
    prior = jps.plan(scene.start, scene.goal)
    np.testing.assert_allclose(jps.cost, expected["jps_cost"], rtol=0, atol=1e-12)
    # Expand every jump to verify the graph's actual voxel steps.
    for first, second in zip(grid.float_to_int(prior[:-1]), grid.float_to_int(prior[1:])):
        difference = second - first
        steps = np.max(np.abs(difference))
        direction = np.sign(difference)
        np.testing.assert_array_equal(difference, steps * direction)
        cells = first + np.arange(steps + 1)[:, None] * direction
        assert np.all(grid.values[tuple(cells.T)] == 0)
    dmp = DistanceMapPlanner(grid)
    path = dmp.plan(scene.start, scene.goal, np.asarray(expected["jps_path"]))
    fingerprint = 14695981039346656037
    for value in dmp.distance_grid.values.ravel(order="F"):
        fingerprint = ((fingerprint ^ int(value)) * 1099511628211) & ((1 << 64) - 1)
    assert fingerprint == expected["potential_fnv64"]
    np.testing.assert_allclose(dmp.cost, expected["dmp_cost"], rtol=0, atol=1e-12)
    cells = grid.float_to_int(path)
    assert np.all(grid.values[tuple(cells.T)] == 0)
    assert np.max(np.abs(np.diff(cells, axis=0))) <= 1
    assert np.all(dmp.region[tuple(cells[1:].T)])
    np.testing.assert_array_equal(cells[[0, -1]], grid.float_to_int(np.array([scene.start, scene.goal])))
