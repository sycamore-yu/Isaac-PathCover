"""ROG-Map occupancy and counter inflation (SUPER 2ad3419, LGPL-3.0).

The world lattice, float32 probabilities, hit-priority update cache, spherical
counter inflation and virtual-height queries follow the upstream C++ code.
NumPy arrays replace the ring buffer; sliding copies the retained rectangle.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from functools import lru_cache
from time import perf_counter

import numpy as np


class GridType(IntEnum):
    UNDEFINED = 0
    UNKNOWN = 1
    OUT_OF_MAP = 2
    OCCUPIED = 3
    KNOWN_FREE = 4
    FRONTIER = 5


@dataclass
class ROGMapConfig:
    resolution: float = 0.05
    inflation_resolution: float = 0.1
    inflation_step: int = 3
    unknown_inflation: bool = False
    unknown_inflation_step: int = 1
    unknown_threshold: float = 1.0
    virtual_ground_height: float = -0.1
    virtual_ceil_height: float = 3.5
    raycasting: bool = False
    ray_range: tuple[float, float] = (0.5, 100.0)
    local_update_box: tuple[float, float, float] = (100.0, 100.0, 5.0)
    batch_update_size: int = 1
    point_filter_num: int = 1
    intensity_threshold: float = -10.0
    p_min: float = 0.12
    p_miss: float = 0.49
    p_free: float = 0.499
    p_occ: float = 0.85
    p_hit: float = 0.9
    p_max: float = 0.98
    map_sliding: bool = False
    sliding_threshold: float = 3.0
    frontier_extraction: bool = False
    esdf_enabled: bool = False
    esdf_resolution: float = 0.1
    esdf_local_update_box: tuple[float, float, float] = (5.0, 5.0, 3.0)


@lru_cache(maxsize=32)
def spherical_neighbors(step, *, full_step_one=False):
    values = np.indices((2 * step + 1,) * 3, dtype=np.int64).reshape(3, -1).T - step
    if not (full_step_one and step == 1):
        values = values[np.sum(values * values, axis=1) <= step * step]
    return values[np.argsort(np.sum(values * values, axis=1), kind="stable")]


def ray_indices(start, end, resolution):
    """ROG RayCaster: exclude the terminal cell; ties prefer z, then y, x."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    current = np.floor(start / resolution).astype(np.int64)
    target = np.floor(end / resolution).astype(np.int64)
    delta = target - current
    count = int(np.abs(delta).sum())
    if count == 0:
        return np.empty((0, 3), dtype=np.int64)
    direction = np.sign(delta)
    extent = np.abs(end - start)
    length = float(np.linalg.norm(extent))
    unit = extent / length
    step = np.full(3, np.finfo(float).max)
    np.divide(resolution, unit, out=step, where=direction != 0)
    boundary = (current + 0.5) * resolution + direction * resolution * 0.5
    crossing = np.full(3, np.finfo(float).max)
    np.divide(np.abs(boundary - start), unit, out=crossing, where=direction != 0)
    times, axes = [], []
    for axis in (2, 1, 0):
        size = abs(int(delta[axis]))
        if size:
            increments = np.full(size, step[axis])
            increments[0] = crossing[axis]
            times.append(np.cumsum(increments))
            axes.append(np.full(size, axis, dtype=np.int64))
    times, axes = np.concatenate(times), np.concatenate(axes)
    axes = axes[np.argsort(times, kind="stable")]
    moves = np.zeros((count, 3), dtype=np.int64)
    moves[np.arange(count), axes] = direction[axes]
    output = np.empty_like(moves)
    output[0] = current
    output[1:] = current + np.cumsum(moves[:-1], axis=0)
    return output


class ROGMap:
    def __init__(self, lower, upper, config=None):
        self.config = config or ROGMapConfig()
        c = self.config
        self.resolution = float(c.resolution)
        self.inflation_ratio = int(np.ceil(c.inflation_resolution / self.resolution))
        self.inflation_resolution = self.resolution * self.inflation_ratio
        self.half_size = (np.asarray(upper, float) - np.asarray(lower, float)) / 2
        self.half_size = (self.half_size / self.inflation_resolution).astype(np.int64) * self.inflation_ratio
        self.origin_index = self.world_to_grid((np.asarray(lower) + np.asarray(upper)) / 2)
        self.origin = self.origin_index * self.resolution
        self.shape = 2 * self.half_size + 1
        probabilities = np.array([c.p_min, c.p_miss, c.p_free, c.p_occ, c.p_hit, c.p_max], dtype=np.float32)
        odds = np.log((probabilities / (1.0 - probabilities)).astype(float)).astype(np.float32)
        self.l_min, self.l_miss, self.l_free, self.l_occ, self.l_hit, self.l_max = odds
        self.unknown_value = np.float32(float(self.l_free + self.l_occ) / 2)
        self.values = np.full(tuple(self.shape), self.unknown_value, dtype=np.float32)
        inf_r = self.inflation_resolution
        self.inf_ground_index = int(np.floor(c.virtual_ground_height / inf_r)) + c.inflation_step
        self.inf_ceil_index = int(np.floor(c.virtual_ceil_height / inf_r)) - c.inflation_step
        self.inf_ground = self.inf_ground_index * inf_r + 0.5 * inf_r
        self.inf_ceil = self.inf_ceil_index * inf_r - 0.5 * inf_r
        self.ground_index = int(np.floor(self.inf_ground * (1.0 / self.resolution)))
        self.ceil_index = int(np.floor(self.inf_ceil * (1.0 / self.resolution)))
        self.virtual_ground_height = self.ground_index * self.resolution
        self.virtual_ceil_height = self.ceil_index * self.resolution
        self._neighbors = spherical_neighbors(c.inflation_step, full_step_one=True)
        self._unknown_neighbors = spherical_neighbors(c.unknown_inflation_step, full_step_one=True)
        self._set_bounds()
        self._init_counters()
        self.esdf = None
        if c.esdf_enabled:
            from .esdf import ESDFMap
            self.esdf = ESDFMap(self.half_size, self.resolution, c.esdf_resolution, self.origin,
                                c.esdf_local_update_box, c.unknown_threshold)
        self.version = 0
        self.map_empty = True
        self._first_update = True
        self._batch_count = 0
        self._cached_hits = []
        self._cached_misses = []
        self.last_timings = {}

    def _set_bounds(self):
        self.lower_index = self.origin_index - self.half_size
        self.upper_index = self.origin_index + self.half_size
        self.lower = self.grid_to_world(self.lower_index)
        self.upper = self.grid_to_world(self.upper_index)
        pad = max(self.config.inflation_step, self.config.unknown_inflation_step if self.config.unknown_inflation else 0) + 1
        self.inf_half_size = self.half_size // self.inflation_ratio + pad
        self.inf_origin_index = np.floor(self.origin / self.inflation_resolution).astype(np.int64)
        self.inf_lower_index = self.inf_origin_index - self.inf_half_size
        self.inf_shape = self.inf_half_size * 2 + 1

    def _init_counters(self):
        self.occupied_count = np.zeros(tuple(self.inf_shape), dtype=np.int32)
        self.unknown_count = np.full(tuple(self.inf_shape), self.inflation_ratio ** 3, dtype=np.int32)
        self.inflated_count = np.zeros(tuple(self.inf_shape), dtype=np.int16)
        self.unknown_inflated_count = (np.full(tuple(self.inf_shape), len(self._unknown_neighbors), dtype=np.int16)
                                      if self.config.unknown_inflation else None)
        self.free_neighbor_count = (np.zeros(tuple(self.shape), dtype=np.int16)
                                    if self.config.frontier_extraction else None)

    def world_to_grid(self, points, *, inflated=False):
        resolution = self.inflation_resolution if inflated else self.resolution
        return np.floor(np.asarray(points, float) * (1.0 / resolution)).astype(np.int64)

    def grid_to_world(self, indices, *, inflated=False):
        return (np.asarray(indices) + 0.5) * (self.inflation_resolution if inflated else self.resolution)

    def inside(self, points):
        indices = self.world_to_grid(points)
        return np.all((indices >= self.lower_index) & (indices <= self.upper_index), axis=-1)

    def _types(self, values):
        return np.where(values >= self.l_occ, int(GridType.OCCUPIED),
                        np.where(values < self.l_free, int(GridType.KNOWN_FREE), int(GridType.UNKNOWN)))

    def log_odds(self, points):
        points = np.asarray(points, float)
        ids = self.world_to_grid(points).reshape(-1, 3)
        valid = np.all((ids >= self.lower_index) & (ids <= self.upper_index), axis=1)
        result = np.zeros(len(ids), dtype=np.float32)
        result[valid] = self.values[tuple((ids[valid] - self.lower_index).T)]
        return result.reshape(points.shape[:-1])

    def get_grid_type(self, points):
        points = np.asarray(points, float)
        ids = self.world_to_grid(points)
        result = self._types(self.log_odds(points))
        result = np.where(self.inside(points), result, int(GridType.OUT_OF_MAP))
        vertical = ((points[..., 2] <= self.virtual_ground_height)
                    | (points[..., 2] >= self.virtual_ceil_height)
                    | (ids[..., 2] <= self.ground_index) | (ids[..., 2] >= self.ceil_index))
        return np.where(vertical, int(GridType.OCCUPIED), result)

    def _counter_values(self, points, counter):
        points = np.asarray(points, float)
        ids = self.world_to_grid(points, inflated=True).reshape(-1, 3) - self.inf_lower_index
        valid = np.all((ids >= 0) & (ids < self.inf_shape), axis=1)
        result = np.zeros(len(ids), dtype=counter.dtype)
        result[valid] = counter[tuple(ids[valid].T)]
        return result.reshape(points.shape[:-1])

    def get_inf_grid_type(self, points):
        points = np.asarray(points, float)
        result = np.where(self._counter_values(points, self.inflated_count) > 0,
                          int(GridType.OCCUPIED), int(GridType.KNOWN_FREE))
        if self.config.unknown_inflation:
            result = np.where((result != int(GridType.OCCUPIED))
                              & (self._counter_values(points, self.unknown_inflated_count) > 0),
                              int(GridType.UNKNOWN), result)
        margin = self.inflation_resolution * (1 + self.config.inflation_step)
        vertical = ((points[..., 2] >= self.inf_ceil - margin)
                    | (points[..., 2] <= self.inf_ground + margin))
        result = np.where(vertical, int(GridType.OCCUPIED), result)
        return np.where(self.inside(points), result, int(GridType.OUT_OF_MAP))

    def is_occupied(self, points):
        points = np.asarray(points, float)
        vertical = (points[..., 2] > self.virtual_ceil_height) | (points[..., 2] < self.virtual_ground_height)
        return self.inside(points) & ((self.log_odds(points) >= self.l_occ) | vertical)

    def _occupied_indices(self, indices):
        """Integer overload of ProbMap::isOccupied, used for seed neighbors."""
        indices = np.asarray(indices, dtype=np.int64)
        valid = np.all((indices >= self.lower_index) & (indices <= self.upper_index), axis=-1)
        occupied = np.zeros(valid.shape, dtype=bool)
        local = indices[valid] - self.lower_index
        occupied[valid] = self.values[tuple(local.T)] >= self.l_occ
        occupied |= valid & ((indices[..., 2] > self.ceil_index) | (indices[..., 2] < self.ground_index))
        return occupied

    def _occupied_neighbors(self, indices, neighbors):
        neighbors = np.asarray(neighbors, dtype=np.int64)
        query = (np.asarray(indices)[:, None] + neighbors[None]).reshape(-1, 3)
        return self._occupied_indices(query).reshape(len(indices), len(neighbors)).any(axis=1)

    def is_occupied_inflate(self, points):
        points = np.asarray(points, float)
        ids = self.world_to_grid(points, inflated=True)
        inside = np.all(np.abs(ids - self.inf_origin_index) <= self.inf_half_size, axis=-1)
        vertical = (points[..., 2] > self.inf_ceil) | (points[..., 2] < self.inf_ground)
        return inside & ((self._counter_values(points, self.inflated_count) > 0) | vertical)

    def is_known_free(self, points):
        points = np.asarray(points, float)
        return (self.inside(points) & (self.log_odds(points) < self.l_free)
                & (points[..., 2] <= self.virtual_ceil_height) & (points[..., 2] >= self.virtual_ground_height))

    def is_unknown(self, points):
        points = np.asarray(points, float)
        values = self.log_odds(points)
        return (~self.inside(points) | ((values >= self.l_free) & (values < self.l_occ)
                & (points[..., 2] <= self.virtual_ceil_height) & (points[..., 2] >= self.virtual_ground_height)))

    def is_unknown_inflate(self, points):
        if not self.config.unknown_inflation:
            raise RuntimeError("ROG-Map unknown inflation is disabled")
        points = np.asarray(points, float)
        return ((self._counter_values(points, self.unknown_inflated_count) > 0)
                & (points[..., 2] < self.inf_ceil - self.inflation_resolution)
                & (points[..., 2] > self.inf_ground + self.inflation_resolution))

    def is_frontier(self, points):
        if self.free_neighbor_count is None:
            raise RuntimeError("ROG-Map frontier extraction is disabled")
        points = np.asarray(points, float)
        ids = self.world_to_grid(points).reshape(-1, 3)
        valid = np.all((ids >= self.lower_index) & (ids <= self.upper_index), axis=1)
        adjacent_free = np.zeros(len(ids), dtype=bool)
        adjacent_free[valid] = self.free_neighbor_count[tuple((ids[valid] - self.lower_index).T)] > 0
        return (adjacent_free.reshape(points.shape[:-1]) & self.is_unknown(points)
                & self.inside(points))

    def _update_frontier(self, ids, before, after):
        delta = (after == GridType.KNOWN_FREE).astype(np.int16) - (before == GridType.KNOWN_FREE)
        changed = delta != 0
        ids, delta = ids[changed], delta[changed]
        neighbors = spherical_neighbors(1, full_step_one=True)
        for begin in range(0, len(ids), 2048):
            local = (ids[begin:begin + 2048, None] + neighbors[None]).reshape(-1, 3) - self.lower_index
            valid = np.all((local >= 0) & (local < self.shape), axis=1)
            np.add.at(self.free_neighbor_count, tuple(local[valid].T),
                      np.repeat(delta[begin:begin + 2048], len(neighbors))[valid])

    def _add_inflation(self, indices, signs, counter, neighbors):
        for begin in range(0, len(indices), 2048):
            expanded = (indices[begin:begin + 2048, None] + neighbors[None]).reshape(-1, 3)
            local = expanded - self.inf_lower_index
            keep = np.all((local >= 0) & (local < self.inf_shape), axis=1)
            np.add.at(counter, tuple(local[keep].T), np.repeat(signs[begin:begin + 2048], len(neighbors))[keep])

    def _update_counters(self, ids, before, after):
        changed = before != after
        if not np.any(changed):
            return
        ids, before, after = ids[changed], before[changed], after[changed]
        coarse = self.world_to_grid(self.grid_to_world(ids), inflated=True)
        unique, inverse = np.unique(coarse, axis=0, return_inverse=True)
        local = tuple((unique - self.inf_lower_index).T)
        old_occ = self.occupied_count[local].copy()
        old_unknown = self.unknown_count[local].copy()
        occ_delta = (after == GridType.OCCUPIED).astype(int) - (before == GridType.OCCUPIED)
        unk_delta = (after == GridType.UNKNOWN).astype(int) - (before == GridType.UNKNOWN)
        self.occupied_count[local] += np.bincount(inverse, weights=occ_delta, minlength=len(unique)).astype(np.int32)
        self.unknown_count[local] += np.bincount(inverse, weights=unk_delta, minlength=len(unique)).astype(np.int32)
        new_occ = self.occupied_count[local]
        transition = (old_occ > 0) != (new_occ > 0)
        signs = np.where(new_occ[transition] > 0, 1, -1)
        self._add_inflation(unique[transition], signs, self.inflated_count, self._neighbors)
        if self.config.unknown_inflation:
            threshold = max(1, min(self.inflation_ratio ** 3, int(np.ceil(self.config.unknown_threshold * self.inflation_ratio ** 3))))
            old_type = (old_occ == 0) & (old_unknown >= threshold)
            new_type = (new_occ == 0) & (self.unknown_count[local] >= threshold)
            transition = old_type != new_type
            self._add_inflation(unique[transition], np.where(new_type[transition], 1, -1),
                                self.unknown_inflated_count, self._unknown_neighbors)
        if self.free_neighbor_count is not None:
            self._update_frontier(ids, before, after)
        if self.esdf is not None:
            self.esdf.update_counters(self.grid_to_world(ids), before, after)
        self.version += 1

    def _apply(self, ids, increments=None, value=None):
        ids = np.asarray(ids, dtype=np.int64).reshape(-1, 3)
        keep = np.all((ids >= self.lower_index) & (ids <= self.upper_index), axis=1)
        ids = ids[keep]
        if not len(ids):
            return
        local = tuple((ids - self.lower_index).T)
        before_values = self.values[local].copy()
        if value is None:
            after_values = np.clip(before_values + np.asarray(increments, dtype=np.float32)[keep], self.l_min, self.l_max)
        else:
            after_values = np.full(len(ids), value, dtype=np.float32)
        self.values[local] = after_values
        self._update_counters(ids, self._types(before_values), self._types(after_values))

    def set_occupied(self, points):
        """ROG updateOccPointCloud, including float32 PCL coordinate conversion."""
        points = np.asarray(points, dtype=np.float32).astype(float).reshape(-1, 3)
        keep = (np.isfinite(points).all(axis=1) & (points[:, 2] <= self.virtual_ceil_height)
                & (points[:, 2] >= self.virtual_ground_height) & self.inside(points))
        ids, counts = np.unique(self.world_to_grid(points[keep]), axis=0, return_counts=True)
        self._apply(ids, increments=(counts * int(np.ceil(self.l_occ / self.l_hit))).astype(np.float32) * self.l_hit)
        self.map_empty = False

    def set_free(self, points):
        ids = np.unique(self.world_to_grid(points), axis=0)
        self._apply(ids, value=self.l_min)

    def _flush_cache(self):
        hits = np.concatenate(self._cached_hits) if self._cached_hits else np.empty((0, 3), dtype=np.int64)
        misses = np.concatenate(self._cached_misses) if self._cached_misses else np.empty((0, 3), dtype=np.int64)
        combined = np.vstack((hits, misses))
        if len(combined):
            unique, inverse = np.unique(combined, axis=0, return_inverse=True)
            hit_count = np.bincount(inverse[:len(hits)], minlength=len(unique))
            miss_count = np.bincount(inverse[len(hits):], minlength=len(unique))
            # Upstream hit candidates suppress every miss for the same cell.
            increments = np.where(hit_count > 0, hit_count.astype(np.float32) * self.l_hit,
                                  miss_count.astype(np.float32) * self.l_miss)
            self._apply(unique, increments)
        self._cached_hits.clear()
        self._cached_misses.clear()
        self._batch_count = 0

    def update_map(self, origin, endpoints):
        start = perf_counter()
        origin = np.asarray(origin, float)
        points = np.asarray(endpoints, dtype=np.float32).astype(float)
        if points.size == 0:
            return {"fusion_s": 0.0, "occupancy_changed": False, "map_version": self.version}
        points = points.reshape(-1, points.shape[-1])
        if points.shape[1] > 3 and self.config.intensity_threshold > 0:
            points = points[points[:, 3] >= self.config.intensity_threshold]
        points = points[::self.config.point_filter_num, :3]
        points = points[np.isfinite(points).all(axis=1)]
        version = self.version
        if self.config.map_sliding and self._batch_count == 0:
            if not self.inside(origin):
                self.slide(origin)
                return {"fusion_s": perf_counter() - start, "occupancy_changed": True, "map_version": self.version}
            if self.map_empty or np.linalg.norm(origin - self.origin) > self.config.sliding_threshold:
                self.slide(origin)
        if origin[2] > self.virtual_ceil_height or origin[2] < self.virtual_ground_height:
            return {"fusion_s": perf_counter() - start, "occupancy_changed": False, "map_version": self.version}
        if not self.config.raycasting:
            keep = self.inside(points) & (np.sum((points - origin) ** 2, axis=1) >= self.config.ray_range[0] ** 2)
            self._cached_hits.append(self.world_to_grid(points[keep]))
        else:
            half = (np.asarray(self.config.local_update_box) / (2 * self.resolution)).astype(np.int64)
            origin_id = self.world_to_grid(origin)
            lo = np.maximum(self.grid_to_world(origin_id - half), self.lower)
            hi = np.minimum(self.grid_to_world(origin_id + half), self.upper)
            for endpoint in points:
                p = endpoint.copy()
                hit = True
                if p[2] > self.virtual_ceil_height or p[2] < self.virtual_ground_height:
                    z = self.virtual_ceil_height if p[2] > self.virtual_ceil_height else self.virtual_ground_height
                    delta = p - origin
                    p = origin + delta / np.linalg.norm(delta) * (z - origin[2]) / delta[2]
                    hit = False
                delta = p - origin
                squared = float(delta @ delta)
                if squared > self.config.ray_range[1] ** 2:
                    p = origin + delta * self.config.ray_range[1] / np.sqrt(squared)
                    hit = False
                if squared < self.config.ray_range[0] ** 2:
                    continue
                if np.any(p < lo) or np.any(p > hi):
                    delta = p - origin
                    ratio = np.divide(np.where(delta >= 0, hi - origin, lo - origin), delta,
                                      out=np.full(3, np.inf), where=delta != 0)
                    p = origin + delta * (np.min(ratio) - 1e-3)
                    hit = False
                if hit:
                    self._cached_hits.append(self.world_to_grid(p)[None])
                delta = p - origin
                ray_start = origin + delta / np.linalg.norm(delta) * self.config.ray_range[0]
                ids = ray_indices(ray_start, p, self.resolution)
                self._cached_misses.append(ids[np.all((ids >= self.lower_index) & (ids <= self.upper_index), axis=1)])
        self._batch_count += 1
        if self._batch_count >= self.config.batch_update_size:
            self._flush_cache()
            self.map_empty = False
        if self.esdf is not None:
            self.esdf.update(origin)
        if self._first_update:
            self._first_update = False
            radius = self.config.ray_range[0]
            axis_values, value = [], -radius
            while value <= radius:
                axis_values.append(value)
                value += self.resolution
            axes = np.asarray(axis_values)
            offsets = np.array(np.meshgrid(axes, axes, axes, indexing="ij")).reshape(3, -1).T
            offsets = offsets[np.linalg.norm(offsets, axis=1) <= radius]
            ids = np.unique(self.world_to_grid(origin + offsets), axis=0)
            self._apply(ids, increments=np.full(len(ids), 999, dtype=np.float32) * self.l_miss)
        self.last_timings = {"fusion_s": perf_counter() - start, "occupancy_changed": self.version != version,
                             "map_version": self.version}
        return self.last_timings

    def slide(self, origin):
        new_index = self.world_to_grid(origin)
        shift = new_index - self.origin_index
        if np.any(np.abs(shift) > self.shape):
            self.values.fill(self.unknown_value)
        else:
            self.values = np.roll(self.values, tuple(-shift), axis=(0, 1, 2))
            for axis, amount in enumerate(shift):
                if amount:
                    sl = [slice(None)] * 3
                    sl[axis] = slice(-amount, None) if amount > 0 else slice(None, -amount)
                    self.values[tuple(sl)] = 0.0
        self.origin_index = new_index
        self.origin = new_index * self.resolution
        self._set_bounds()
        self._init_counters()
        if self.esdf is not None:
            self.esdf.reset_counters(self.origin)
        active = np.argwhere((self.values >= self.l_occ) | (self.values < self.l_free))
        if len(active):
            types = self._types(self.values[tuple(active.T)])
            self._update_counters(active + self.lower_index, np.full(len(active), GridType.UNKNOWN), types)
        self.version += 1

    def bound_box(self, lower, upper):
        lo, hi = np.maximum(lower, self.lower).copy(), np.minimum(upper, self.upper).copy()
        lo[2] = max(lo[2], self.virtual_ground_height)
        hi[2] = min(hi[2], self.virtual_ceil_height)
        return lo, hi

    def box_search(self, lower, upper, grid_type=GridType.OCCUPIED):
        if self.map_empty:
            return np.empty((0, 3))
        lo, hi = self.bound_box(np.asarray(lower, float), np.asarray(upper, float))
        begin, end = self.world_to_grid(lo) + 1, self.world_to_grid(hi)
        if np.any(begin >= end):
            return np.empty((0, 3))
        slices = tuple(slice(a, b) for a, b in zip(begin - self.lower_index, end - self.lower_index))
        values = self.values[slices]
        if grid_type == GridType.OCCUPIED:
            mask = values >= self.l_occ
            z = np.arange(begin[2], end[2])
            mask |= (z[None, None] > self.ceil_index) | (z[None, None] < self.ground_index)
        elif grid_type == GridType.UNKNOWN:
            mask = (values >= self.l_free) & (values < self.l_occ)
        elif grid_type == GridType.FRONTIER:
            if self.free_neighbor_count is None:
                raise RuntimeError("ROG-Map frontier extraction is disabled")
            mask = (values >= self.l_free) & (values < self.l_occ) & (self.free_neighbor_count[slices] > 0)
        else:
            raise ValueError("ROG box_search supports occupied, unknown or frontier cells")
        return self.grid_to_world(np.argwhere(mask) + begin)

    def occupied_points(self):
        return self.grid_to_world(np.argwhere(self.values >= self.l_occ) + self.lower_index)

    def is_line_free(self, start, end, max_distance=999999.0, neighbors=None, *, inflated=False,
                     unknown_as_occupied=False, return_last=False):
        start, end = np.asarray(start, float), np.asarray(end, float)
        if not inflated and not unknown_as_occupied:
            from .raycast import raw_line_free
            offsets = np.empty((0, 3), dtype=np.int64) if neighbors is None else np.asarray(neighbors, dtype=np.int64)
            free, x, y, z = raw_line_free(start, end, self.resolution, float(max_distance), offsets,
                                          self.values, self.lower_index, self.upper_index, self.l_occ,
                                          self.ground_index, self.ceil_index,
                                          self.virtual_ground_height, self.virtual_ceil_height)
            return (free, np.array([x, y, z])) if return_last else free
        resolution = self.inflation_resolution if inflated else self.resolution
        ids = ray_indices(start, end, resolution)
        if not len(ids):
            return (True, end.copy()) if return_last else True
        points = self.grid_to_world(ids, inflated=inflated)
        blocked = np.zeros(len(points), dtype=bool)
        if max_distance > 0:
            blocked |= np.linalg.norm(points - start, axis=1) > max_distance
        if neighbors is not None and len(neighbors):
            if inflated:
                query = (ids[:, None] + np.asarray(neighbors)[None]).reshape(-1, 3)
                occupied = self.is_occupied_inflate(self.grid_to_world(query, inflated=True))
                blocked |= occupied.reshape(len(ids), -1).any(axis=1)
            else:
                blocked |= self._occupied_neighbors(ids, neighbors)
        else:
            blocked |= self.is_occupied_inflate(points) if inflated else self.is_occupied(points)
        if unknown_as_occupied:
            blocked |= self.is_unknown_inflate(points) if inflated else ~self.is_known_free(points)
        if np.any(blocked):
            last = points[np.flatnonzero(blocked)[0]]
            return (False, last) if return_last else False
        return (True, end.copy()) if return_last else True

    def first_blocked_line(self, start, ends, max_distance, neighbors):
        from .raycast import raw_lines_first_blocked
        return raw_lines_first_blocked(np.asarray(start, float), np.asarray(ends, float).reshape(-1, 3),
                                        self.resolution, float(max_distance), np.asarray(neighbors, dtype=np.int64),
                                        self.values, self.lower_index, self.upper_index, self.l_occ,
                                        self.ground_index, self.ceil_index,
                                        self.virtual_ground_height, self.virtual_ceil_height)

    def _nearest_not_occupied(self, position, max_distance, inflated):
        position = np.asarray(position, float)
        center = self.world_to_grid(position)
        center_point = self.grid_to_world(center)
        if np.linalg.norm(center_point - position) > max_distance:
            return None
        center_type = self.get_inf_grid_type(center_point) if inflated else self.get_grid_type(center_point)
        if center_type != GridType.OCCUPIED:
            return center_point
        offsets = spherical_neighbors(int(np.ceil(max_distance / self.resolution)))
        for begin in range(0, len(offsets), 1024):
            query = self.grid_to_world(center + offsets[begin:begin + 1024])
            # The source terminates at its first centre farther than max_dis.
            outside = np.flatnonzero(np.linalg.norm(query - position, axis=1) > max_distance)
            if len(outside):
                query = query[:outside[0]]
            kinds = self.get_inf_grid_type(query) if inflated else self.get_grid_type(query)
            valid = np.flatnonzero(kinds != GridType.OCCUPIED)
            if len(valid):
                return query[valid[0]].copy()
            if len(outside):
                return None
        return None

    def nearest_cell_not_occupied(self, position, max_distance):
        return self._nearest_not_occupied(position, max_distance, False)

    def nearest_inf_not_occupied(self, position, max_distance):
        return self._nearest_not_occupied(position, max_distance, True)
