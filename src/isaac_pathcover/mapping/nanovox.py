"""NanoVoxMap's CPU occupancy semantics, using sparse 8³ NumPy blocks.

Ported from PathCover 8723a05d, nanovoxmap.hpp. Ray updates are reduced
within the miss and hit phases separately; monotone saturation makes each
reduction exactly equivalent to the source's serial updates.
"""

from __future__ import annotations

import math
import numpy as np
from scipy.spatial import cKDTree


def _logodds(probability: float) -> int:
    if not 0 < probability < 1:
        raise ValueError("occupancy probabilities must lie in (0, 1)")
    value = 20 * math.log(probability / (1 - probability))
    return max(-127, min(127, math.floor(value + .5) if value >= 0 else math.ceil(value - .5)))


class NanoVoxMap:
    """Unbounded world-zero voxel lattice; occupied iff quantized odds > 0."""

    def __init__(self, resolution=.05, p_hit=.9, p_miss=.1, p_min=.1192, p_max=.971):
        if not math.isfinite(resolution) or resolution <= 0:
            raise ValueError("resolution must be positive and finite")
        if not (p_miss < .5 < p_hit and p_min < .5 < p_max):
            raise ValueError("hit/max must exceed .5; miss/min must be below .5")
        self.resolution = float(resolution)
        self._inverse = 1. / self.resolution
        self.l_hit, self.l_miss, self.l_min, self.l_max = map(_logodds, (p_hit, p_miss, p_min, p_max))
        self.blocks: dict[tuple[int, int, int], int] = {}
        self._block_keys = []
        self._voxels = np.zeros((0, 512), dtype=np.int8)
        self._tree = None
        self.version = 0

    def world_to_grid(self, points):
        return np.floor(np.asarray(points, dtype=np.float64) * self._inverse).astype(np.int64)

    def grid_to_world(self, indices):
        return (np.asarray(indices) + .5) * self.resolution

    def _valid_points(self, points):
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        scaled = points * self._inverse
        valid = np.isfinite(scaled).all(axis=1)
        valid &= ((scaled >= -2147483647) & (scaled <= 2147483646)).all(axis=1)
        return points[valid]

    def _update(self, indices, delta, *, overwrite=False):
        if len(indices) == 0:
            return np.empty((0, 3), dtype=np.int64)
        indices = np.asarray(indices, dtype=np.int64)
        order = np.lexsort((indices[:, 2], indices[:, 1], indices[:, 0]))
        ordered = indices[order]
        first = np.r_[True, np.any(ordered[1:] != ordered[:-1], axis=1)]
        starts = np.flatnonzero(first)
        counts = np.diff(np.r_[starts, len(indices)])
        indices = ordered[first]
        block_coordinates = indices >> 3
        order = np.lexsort((block_coordinates[:, 2], block_coordinates[:, 1], block_coordinates[:, 0]))
        ordered_blocks = block_coordinates[order]
        first = np.r_[True, np.any(ordered_blocks[1:] != ordered_blocks[:-1], axis=1)]
        boundaries = np.r_[np.flatnonzero(first), len(indices)]
        block_ids = ordered_blocks[first]
        handles = np.empty(len(block_ids), dtype=np.int64)
        for number, key in enumerate(block_ids):
            key = tuple(key)
            handle = self.blocks.get(key)
            if handle is None:
                handle = self.blocks[key] = len(self.blocks)
                self._block_keys.append(key)
            handles[number] = handle
        if len(self.blocks) > len(self._voxels):
            capacity = ((len(self.blocks) + 1023) // 1024) * 1024
            pool = np.zeros((capacity, 512), dtype=np.int8)
            pool[:len(self._voxels)] = self._voxels
            self._voxels = pool
        handles = np.repeat(handles, np.diff(boundaries))
        xyz = indices[order] & 7
        offsets = xyz[:, 0] + 8 * xyz[:, 1] + 64 * xyz[:, 2]
        before = self._voxels[handles, offsets]
        if overwrite:
            after = np.full(len(offsets), delta, dtype=np.int8)
        else:
            after = np.clip(before.astype(np.int64) + delta * counts[order], self.l_min, self.l_max).astype(np.int8)
        self._voxels[handles, offsets] = after
        changed = (before > 0) != (after > 0)
        if changed.any():
            self._tree = None
            self.version += 1
            return indices[order[changed]]
        return np.empty((0, 3), dtype=np.int64)

    @staticmethod
    def trace_misses(start, endpoints):
        """Amanatides-Woo on voxel centres; x, y, z ties match the C++ code."""
        endpoints = np.asarray(endpoints, dtype=np.int64).reshape(-1, 3)
        current = np.broadcast_to(np.asarray(start, dtype=np.int64), endpoints.shape).copy()
        delta = endpoints - current
        direction = np.sign(delta)
        extent = np.abs(delta)
        increment = np.full(extent.shape, np.inf)
        np.divide(1., extent, out=increment, where=extent != 0)
        crossing = .5 * increment
        remaining = extent.sum(axis=1)
        visited = []
        while len(remaining) and remaining.max() > 0:
            active = remaining > 0
            visited.append(current[active].copy())
            choose_x = (crossing[:, 0] <= crossing[:, 1]) & (crossing[:, 0] <= crossing[:, 2])
            choose_y = (~choose_x) & (crossing[:, 1] <= crossing[:, 2])
            axis = np.where(choose_x, 0, np.where(choose_y, 1, 2))
            rows = np.flatnonzero(active)
            current[rows, axis[rows]] += direction[rows, axis[rows]]
            crossing[rows, axis[rows]] += increment[rows, axis[rows]]
            remaining -= active
        return np.concatenate(visited, axis=0) if visited else np.empty((0, 3), dtype=np.int64)

    def insert_point_cloud(self, origin, endpoints, apply_hits=True):
        """Fuse a scan: all ray misses precede all endpoint hits or clearings."""
        origin = np.asarray(origin, dtype=np.float64)
        cloud = np.asarray(endpoints, dtype=np.float64).reshape(-1, 3)
        if len(self._valid_points(origin)) == 0:
            return
        ends = self.world_to_grid(self._valid_points(cloud))
        start = self.world_to_grid(origin)
        initial_version, initial_tree = self.version, self._tree
        flips = []
        for first in range(0, len(ends), 2048):
            flips.append(self._update(self.trace_misses(start, ends[first:first + 2048]), self.l_miss))
        flips.append(self._update(ends, self.l_hit if apply_hits else self.l_miss))
        changed = np.concatenate(flips)
        _, counts = np.unique(changed, axis=0, return_counts=True)
        occupancy_changed = bool(np.any(counts % 2))
        self.version = initial_version + occupancy_changed
        if not occupancy_changed:
            self._tree = initial_tree

    def insert_ray(self, origin, endpoint):
        self.insert_point_cloud(origin, np.asarray(endpoint).reshape(1, 3))

    def set_occupied(self, points):
        self._update(self.world_to_grid(self._valid_points(points)), self.l_max, overwrite=True)

    def set_free(self, points):
        self._update(self.world_to_grid(self._valid_points(points)), self.l_min, overwrite=True)

    def log_odds(self, points):
        values = np.zeros(np.asarray(points).shape[:-1], dtype=np.int8)
        indices = self.world_to_grid(points).reshape(-1, 3)
        for i, (x, y, z) in enumerate(indices):
            handle = self.blocks.get((x >> 3, y >> 3, z >> 3))
            if handle is not None:
                values.reshape(-1)[i] = self._voxels[handle, (x & 7) + 8 * (y & 7) + 64 * (z & 7)]
        return values

    def is_occupied(self, points):
        return self.log_odds(points) > 0

    def snapshot(self, occupied_only=False):
        if not self.blocks:
            return np.empty((0, 3), dtype=np.int64), np.empty(0, dtype=np.int8)
        data = self._voxels[:len(self.blocks)]
        handles, offsets = np.nonzero(data > 0 if occupied_only else data != 0)
        local = np.column_stack((offsets & 7, (offsets >> 3) & 7, offsets >> 6))
        xyz = local + np.asarray(self._block_keys)[handles] * 8
        values = data[handles, offsets]
        order = np.lexsort((xyz[:, 2], xyz[:, 1], xyz[:, 0]))
        return xyz[order], values[order]

    def occupied_points(self):
        return self.grid_to_world(self.snapshot(occupied_only=True)[0])

    def distance_with_gradient(self, points):
        """Exact distance to occupied voxel centres, as getDistanceWithGradient."""
        if self._tree is None:
            self._tree = cKDTree(self.occupied_points())
        points = np.asarray(points, dtype=np.float64)
        if self._tree.n == 0:
            distance = float(np.sqrt(np.float32(1e18))) * self.resolution
            return np.full(points.shape[:-1], distance), np.zeros_like(points)
        distances, indices = self._tree.query(points)
        gradients = points - self._tree.data[indices]
        np.divide(gradients, np.asarray(distances)[..., None], out=gradients,
                  where=np.asarray(distances)[..., None] != 0)
        return distances, gradients

    def clear(self):
        self.blocks.clear()
        self._block_keys.clear()
        self._voxels = np.zeros((0, 512), dtype=np.int8)
        self._tree = None
        self.version += 1
