"""Optional ROG-Map ESDF (SUPER 2ad3419, LGPL-3.0).

SciPy supplies the exact Euclidean distance transform. Ring indices and the
source's signed-distance combination are retained, including its local-index
combination rule when the map origin is nonzero.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt

from .rog_map import GridType


class ESDFMap:
    def __init__(self, half_prob_size, prob_resolution, resolution, origin,
                 local_update_box, unknown_threshold):
        if resolution < prob_resolution:
            raise ValueError("ESDF resolution must be at least the probability resolution")
        self.ratio = int(np.floor(resolution / prob_resolution + 0.5))
        self.resolution = prob_resolution * self.ratio
        self.inverse_resolution = 1.0 / self.resolution
        self.half_size = (np.asarray(half_prob_size) * prob_resolution / self.resolution).astype(np.int64) + 1
        self.shape = self.half_size * 2 + 1
        self.half_update_box = self.world_to_grid(local_update_box) // 2
        self.sub_grid_num = self.ratio ** 3
        self.unknown_threshold = max(1, min(self.sub_grid_num, int(np.ceil(unknown_threshold * self.sub_grid_num))))
        self.occupied_count = np.zeros(tuple(self.shape), dtype=np.int32)
        self.unknown_count = np.full(tuple(self.shape), self.sub_grid_num, dtype=np.int32)
        self.distance_buffer = np.zeros(tuple(self.shape))
        self._negative = np.zeros(tuple(self.shape))
        self.reset_counters(origin)
        self.updated_min = self.lower_index.copy()
        self.updated_max = self.lower_index - 1

    def world_to_grid(self, points):
        return np.floor(np.asarray(points) * self.inverse_resolution).astype(np.int64)

    def grid_to_world(self, indices):
        return (np.asarray(indices) + 0.5) * self.resolution

    def _ring(self, indices):
        return (np.asarray(indices, dtype=np.int64) + self.half_size) % self.shape

    def reset_counters(self, origin):
        self.origin_index = self.world_to_grid(origin)
        self.lower_index = self.origin_index - self.half_size
        self.upper_index = self.origin_index + self.half_size
        self.occupied_count.fill(0)
        self.unknown_count.fill(self.sub_grid_num)
        # ESDFMap::resetLocalMap/resetOneCell deliberately leave distances intact.

    def update_counters(self, points, before, after):
        local = tuple(self._ring(self.world_to_grid(points)).T)
        occupied_delta = (after == GridType.OCCUPIED).astype(np.int32) - (before == GridType.OCCUPIED)
        unknown_delta = (after == GridType.UNKNOWN).astype(np.int32) - (before == GridType.UNKNOWN)
        np.add.at(self.occupied_count, local, occupied_delta)
        np.add.at(self.unknown_count, local, unknown_delta)

    def get_grid_type(self, points):
        ids = self._ring(self.world_to_grid(points))
        occupied = self.occupied_count[tuple(np.moveaxis(ids, -1, 0))] > 0
        unknown = self.unknown_count[tuple(np.moveaxis(ids, -1, 0))] >= self.unknown_threshold
        return np.where(occupied, GridType.OCCUPIED, np.where(unknown, GridType.UNKNOWN, GridType.KNOWN_FREE))

    def update(self, origin):
        center = self.world_to_grid(origin)
        self.updated_min = np.maximum(center - self.half_update_box, self.lower_index)
        self.updated_max = np.minimum(center + self.half_update_box, self.upper_index) - 1
        if np.any(self.updated_min > self.updated_max):
            return
        axes = [np.arange(lo, hi + 1) for lo, hi in zip(self.updated_min, self.updated_max)]
        ring_axes = [(axis + half) % size for axis, half, size in zip(axes, self.half_size, self.shape)]
        addresses = np.ix_(*ring_axes)
        occupied = self.occupied_count[addresses] > 0
        maximum = np.sqrt(np.finfo(float).max) * self.resolution
        positive = (distance_transform_edt(~occupied) * self.resolution if np.any(occupied)
                    else np.full(occupied.shape, maximum))
        negative = (distance_transform_edt(occupied) * self.resolution if np.any(~occupied)
                    else np.full(occupied.shape, maximum))
        self.distance_buffer[addresses] = positive
        self._negative[addresses] = negative
        # The fixed upstream commit combines the local box directly as hash
        # coordinates here (esdf_map.cpp:225-236), without the ring offset.
        lo = self.updated_min - self.lower_index
        hi = self.updated_max - self.lower_index + 1
        slices = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        negative = self._negative[slices]
        self.distance_buffer[slices] += np.where(negative > 0.0, -negative + self.resolution, 0.0)

    def get_updated_bbox(self):
        return self.grid_to_world(self.updated_min), self.grid_to_world(self.updated_max)

    def get_distance(self, points):
        ids = self._ring(self.world_to_grid(points))
        return self.distance_buffer[tuple(np.moveaxis(ids, -1, 0))]

    def _surround(self, points):
        points = np.asarray(points, float)
        indices = self.world_to_grid(points - 0.5 * self.resolution)
        diff = (points - self.grid_to_world(indices)) / self.resolution
        corners = np.indices((2, 2, 2)).reshape(3, -1).T
        surrounding = self.grid_to_world(indices[..., None, :] + corners)
        return surrounding, diff

    def _values(self, points):
        surrounding, diff = self._surround(points)
        values = self.get_distance(surrounding).reshape(diff.shape[:-1] + (2, 2, 2))
        return values, diff

    @staticmethod
    def _intermediates(values, diff):
        x, y = diff[..., 0], diff[..., 1]
        v00 = (1 - x) * values[..., 0, 0, 0] + x * values[..., 1, 0, 0]
        v01 = (1 - x) * values[..., 0, 0, 1] + x * values[..., 1, 0, 1]
        v10 = (1 - x) * values[..., 0, 1, 0] + x * values[..., 1, 1, 0]
        v11 = (1 - x) * values[..., 0, 1, 1] + x * values[..., 1, 1, 1]
        v0 = (1 - y) * v00 + y * v10
        v1 = (1 - y) * v01 + y * v11
        return v00, v01, v10, v11, v0, v1

    def evaluate_edt(self, points):
        values, diff = self._values(points)
        *_, v0, v1 = self._intermediates(values, diff)
        return (1 - diff[..., 2]) * v0 + diff[..., 2] * v1

    def evaluate_first_grad(self, points):
        values, diff = self._values(points)
        v00, v01, v10, v11, v0, v1 = self._intermediates(values, diff)
        y, z = diff[..., 1], diff[..., 2]
        grad = np.empty(diff.shape)
        grad[..., 2] = (v1 - v0) * self.inverse_resolution
        grad[..., 1] = ((1 - z) * (v10 - v00) + z * (v11 - v01)) * self.inverse_resolution
        grad[..., 0] = (1 - z) * (1 - y) * (values[..., 1, 0, 0] - values[..., 0, 0, 0])
        grad[..., 0] += (1 - z) * y * (values[..., 1, 1, 0] - values[..., 0, 1, 0])
        grad[..., 0] += z * (1 - y) * (values[..., 1, 0, 1] - values[..., 0, 0, 1])
        grad[..., 0] += z * y * (values[..., 1, 1, 1] - values[..., 0, 1, 1])
        grad[..., 0] *= self.inverse_resolution
        return grad

    def evaluate_second_grad(self, points):
        surrounding, diff = self._surround(points)
        first = self.evaluate_first_grad(surrounding).reshape(diff.shape[:-1] + (2, 2, 2, 3))
        x, y, z = diff[..., 0], diff[..., 1], diff[..., 2]
        grad = np.empty(diff.shape)
        grad[..., 0] = (1-y) * ((1-z) * (first[..., 1, 0, 0, 0]-first[..., 0, 0, 0, 0])
                               + z * (first[..., 1, 0, 1, 0]-first[..., 0, 0, 1, 0]))
        grad[..., 0] += y * ((1-z) * (first[..., 1, 1, 0, 0]-first[..., 0, 1, 0, 0])
                            + z * (first[..., 1, 1, 1, 0]-first[..., 0, 1, 1, 0]))
        grad[..., 1] = (1-z) * ((1-x) * (first[..., 0, 1, 0, 1]-first[..., 0, 0, 0, 1])
                               + x * (first[..., 1, 1, 0, 1]-first[..., 1, 0, 0, 1]))
        grad[..., 1] += z * ((1-x) * (first[..., 0, 1, 1, 1]-first[..., 0, 0, 1, 1])
                            + x * (first[..., 1, 1, 1, 1]-first[..., 1, 0, 1, 1]))
        grad[..., 2] = (1-y) * ((1-x) * (first[..., 0, 0, 1, 2]-first[..., 0, 0, 0, 2])
                               + x * (first[..., 1, 0, 1, 2]-first[..., 1, 0, 0, 2]))
        grad[..., 2] += y * ((1-x) * (first[..., 0, 1, 1, 2]-first[..., 0, 1, 0, 2])
                            + x * (first[..., 1, 1, 1, 2]-first[..., 1, 1, 0, 2]))
        return grad * self.inverse_resolution
