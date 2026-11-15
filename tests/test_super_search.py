"""Compare A* paths and statuses with unchanged SUPER astar.cpp and ROG-Map."""
import json
from pathlib import Path

import numpy as np
import pytest

from isaac_pathcover.mapping import ROGMap, ROGMapConfig
from isaac_pathcover.planning.search import AStar


REFERENCE = json.loads((Path(__file__).parent / "fixtures/super_search_reference.json").read_text())
SCENARIOS = {
    "clear": ([-1.75, -1.55, 1.05], [1.65, -1.55, 1.05], 9., True, False, False),
    "wall": ([-1.75, .05, 1.05], [1.65, .05, 1.05], 9., True, False, False),
    "horizon": ([-1.75, .05, 1.05], [3.15, 2.45, 1.05], 1.2, True, False, False),
    "prob_fine": ([-1.75, .05, 1.05], [1.65, .05, 1.05], 9., False, False, True),
    "unknown_blocked": ([-1.75, -1.55, 1.05], [1.65, -1.55, 1.05], 9., False, True, False),
    "timeout": ([-1.75, -1.55, 1.05], [1.65, -1.55, 1.05], 9., True, False, False),
}
ESCAPE = {
    "escape_clear": [-1.75, -1.55, 1.05],
    "escape_inflated": [-.15, .05, 1.05],
    "escape_outside": [99., 0., 1.],
}


@pytest.fixture(scope="module")
def mapper():
    result = ROGMap([-4, -4, -2], [4, 4, 4], ROGMapConfig(
        resolution=.1, inflation_resolution=.2, inflation_step=1,
        virtual_ground_height=-2., virtual_ceil_height=4., raycasting=False))
    # The C++ input is pcl::PointXYZI, whose fields have float32 precision.
    points = [[.05, (y + .5) * .1, (z + .5) * .1]
              for y in range(-10, 11) for z in range(20)]
    result.set_occupied(np.asarray(points, dtype=np.float32))
    return result


@pytest.mark.parametrize("name", list(REFERENCE["cases"]))
def test_original_astar_paths_and_return_codes(mapper, name):
    assert REFERENCE["source_commit"].startswith("2ad3419")
    expected = REFERENCE["cases"][name]
    scenario = expected["scenario"]
    time = 0.

    def clock():
        nonlocal time
        result = time
        if scenario == "timeout":
            time += .051
        return result

    astar = AStar(mapper, map_voxel_num=(81, 81, 61),
                  allow_diag=expected["allow_diag"], clock=clock)
    astar.set_fine_inf_neighbors(2)
    if scenario.startswith("escape_"):
        result = astar.escape_path_search(ESCAPE[scenario])
    else:
        start, goal, horizon, inflated, unknown, fine = SCENARIOS[scenario]
        result = astar.search(start, goal, horizon=horizon, inflated=inflated,
                              unknown_as_occupied=unknown, use_inf_neighbor=fine)
    assert result.status == expected["status"]
    np.testing.assert_allclose(result.path, np.asarray(expected["path"]).reshape(-1, 3),
                               rtol=0., atol=2e-15)


def test_original_nearest_inflation_query_uses_raw_grid_centers(mapper):
    for expected in REFERENCE["nearest"].values():
        actual = mapper.nearest_inf_not_occupied(expected["input"], 3.)
        assert (actual is not None) == expected["found"]
        np.testing.assert_allclose(actual, expected["output"], rtol=0., atol=2e-15)


def test_original_line_neighbor_queries_and_last_ray_cell(mapper):
    for expected in REFERENCE["lines"].values():
        step = expected["neighbor_step"]
        neighbors = np.array([[x, y, z] for x in range(-step, step + 1)
                              for y in range(-step, step + 1) for z in range(-step, step + 1)
                              if x*x + y*y + z*z <= step*step], dtype=np.int64) if step else None
        free, last = mapper.is_line_free(expected["start"], expected["end"],
                                          max_distance=expected["max_distance"],
                                          neighbors=neighbors, return_last=True)
        assert free == expected["free"]
        np.testing.assert_allclose(last, expected["last"], rtol=0., atol=2e-15)
        offsets = np.empty((0, 3), dtype=np.int64) if neighbors is None else neighbors
        index = mapper.first_blocked_line(expected["start"],
                                          [expected["start"], expected["end"], expected["end"]],
                                          expected["max_distance"], offsets)
        assert index == (3 if expected["free"] else 1)
