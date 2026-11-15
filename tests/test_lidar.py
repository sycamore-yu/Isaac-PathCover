"""Exact LiDAR endpoints from recorded Newton sensor poses."""
from pathlib import Path
import json

import numpy as np
import pytest

from isaac_pathcover.simulation.lidar import cast_voxels
from isaac_pathcover.simulation.scene import Scene


@pytest.mark.parametrize("name", ["medium", "dense"])
def test_recorded_lidar_frames_match_original_numpy_bitwise(name):
    fixtures = Path(__file__).parent / "fixtures"
    cases = json.loads((fixtures / "lidar_reference.json").read_text())["scenes"][name]
    scene = Scene(name)
    with np.load(fixtures / "lidar_reference.npz") as expected:
        for case in cases:
            origin, points = scene.lidar(case["position"], case["quaternion_wxyz"])
            np.testing.assert_array_equal(origin, expected[case["key"] + "_origin"])
            np.testing.assert_array_equal(points, expected[case["key"] + "_points"])
            assert origin.tobytes() == expected[case["key"] + "_origin"].tobytes()
            assert points.tobytes() == expected[case["key"] + "_points"].tobytes()


def test_voxel_traversal_keeps_first_axis_ties_and_exact_range_termination():
    occupied = np.zeros((3, 3, 3), dtype=bool)
    occupied[1, 0, 0] = True
    indices = np.array([[0, 0, 0]], dtype=np.int64)
    step = np.ones((1, 3), dtype=np.int64)
    delta = np.ones((1, 3))
    boundary = np.full((1, 3), .5)
    hits, distance = cast_voxels(occupied, indices, step, delta, boundary, 1., 10)
    np.testing.assert_array_equal(hits, [True])
    np.testing.assert_array_equal(distance, [.5])
    # A hit entered exactly at max_range is discarded before the next lookup.
    hits, distance = cast_voxels(occupied, indices, step, delta, boundary, .5, 10)
    np.testing.assert_array_equal(hits, [False])
    np.testing.assert_array_equal(distance, [.5])
