"""Frozen outputs of unmodified PathCover headers, compiled as a test oracle."""

import json
from pathlib import Path

import numpy as np
import pytest

from isaac_pathcover.mapping import NanoVoxMap, OccupancyInflation


FIXTURE = Path(__file__).parent / "fixtures" / "mapping_search.json"


def _fingerprint(map_):
    mask = (1 << 64) - 1
    def mix(x):
        x &= mask
        x ^= x >> 30
        x = (x * 0xbf58476d1ce4e5b9) & mask
        x ^= x >> 27
        x = (x * 0x94d049bb133111eb) & mask
        return x ^ (x >> 31)
    def index_hash(xyz):
        x,y,z = map(int, xyz)
        h = mix(x & 0xffffffff)
        h ^= mix(((y + 0x9e3779b9) & 0xffffffff) + (h << 6))
        h ^= mix(((z + 0x85ebca6b) & 0xffffffff) + (h << 6))
        return h
    voxels, occupied = 0, 0
    xyz, values = map_.snapshot()
    for index, value in zip(xyz, values):
        h = index_hash(index)
        voxels ^= mix(h ^ ((int(value) & 0xff) + 0x100))
        if value > 0:
            occupied ^= mix(h + 0x2000)
    return voxels, occupied


@pytest.mark.parametrize("case", [0, 1])
def test_nanovox_cpp_full_state_fingerprint(case):
    data = json.loads(FIXTURE.read_text())["mapping"][case]
    map_ = NanoVoxMap(.2, p_hit=data["p_hit"], p_miss=data["p_miss"])
    map_.set_occupied([[-.5,.3,.5]])
    map_.set_free([[-.7,.3,.5]])
    for phase, expected in enumerate(data["phases"]):
        map_.insert_point_cloud([.1,.1,.1], data["cloud"], apply_hits=phase != 2)
        assert _fingerprint(map_) == (expected["voxel_fingerprint"], expected["occupied_fingerprint"])
        np.testing.assert_allclose(map_.occupied_points(), expected["occupied"], rtol=0, atol=2e-15)


@pytest.mark.parametrize("mode", [0, 1, 2, 3])
def test_inflation_cpp_identical_lattice_and_float32_points(mode):
    expected = json.loads(FIXTURE.read_text())["inflation"][mode]
    inflater = OccupancyInflation(radius_z=0. if mode == 3 else .25,
                                  shape="box" if mode == 2 else "ellipsoid", surface_only=mode == 1)
    cloud = [[.05,.05,1.05],[.35,.05,1.05],[-.15,.15,.95],[.05,.05,-.05]]
    actual = inflater.inflate(cloud)
    assert len(inflater.kernel) == expected["kernel_size"]
    assert len(inflater.solid_indices) == expected["solid_count"]
    np.testing.assert_array_equal(actual, np.asarray(expected["points"], dtype=np.float32))


def test_mapping_changes_and_clearing_rays():
    map_ = NanoVoxMap(.1)
    point = np.array([[1.05,.05,.05]])
    map_.set_occupied(point)
    version = map_.version
    map_.set_occupied(point)
    assert map_.version == version
    for _ in range(3):
        map_.insert_point_cloud([.05,.05,.05], point, apply_hits=False)
    assert not map_.is_occupied(point).any()
    assert map_.version > version


def test_same_scan_final_occupancy_does_not_invalidate_geometry():
    map_ = NanoVoxMap(.1)
    points = np.array([[1.05,.05,.05],[2.05,.05,.05],[2.05,.05,.05]])
    map_.set_occupied(points)
    version = map_.version
    for _ in range(3):
        map_.insert_point_cloud([.05,.05,.05],points)
        assert map_.version == version
        assert map_.is_occupied(points).all()


def test_inflation_replaces_removed_obstacles():
    inflater = OccupancyInflation()
    inflater.inflate([[0.,0.,1.]])
    assert len(inflater.solid_indices)
    assert len(inflater.inflate(np.empty((0,3)))) == 0


def test_incremental_inflation_matches_union():
    first, second = [[0.,0.,1.]], [[.7,-.1,1.]]
    incremental = OccupancyInflation(incremental=True)
    incremental.inflate(first)
    actual = incremental.inflate(second)
    expected = OccupancyInflation().inflate(first + second)
    np.testing.assert_array_equal(actual, expected)
