"""SUPER's 26-connected A* and escape search (2ad3419, LGPL-3.0)."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
from time import perf_counter

import numpy as np

from ..mapping.rog_map import spherical_neighbors
from . import search_kernel


@dataclass
class SearchResult:
    path: np.ndarray
    status: str
    expanded: int
    cost: float
    elapsed: float = 0.0

    @property
    def success(self):
        return self.status in ("REACH_GOAL", "REACH_HORIZON", "NO_NEED")


class AStar:
    def __init__(self, map, *, map_voxel_num=(500, 500, 100), allow_diag=True,
                 heuristic_type=2, clock=perf_counter, clock_buffer=None):
        self.map = map
        self.half_size = tuple(int(v) // 2 for v in map_voxel_num)
        self.allow_diag = allow_diag
        self.heuristic_type = heuristic_type
        self.clock = clock
        self.clock_buffer = np.empty(0) if clock_buffer is None else clock_buffer
        self._node_lookup = np.full(np.prod(2*np.asarray(self.half_size)+1), -1, dtype=np.int32)
        neighbors = [(dx, dy, dz, math.sqrt(dx * dx + dy * dy + dz * dz))
                     for dx, dy, dz in product((-1, 0, 1), repeat=3)
                     if (dx or dy or dz) and (allow_diag or abs(dx) + abs(dy) + abs(dz) == 1)]
        self.offsets = np.asarray([n[:3] for n in neighbors], dtype=np.int64)
        self.distances = np.asarray([n[3] for n in neighbors])
        self.escape_offsets = np.asarray([v for v in product((-1, 0, 1), repeat=3) if any(v)], dtype=np.int64)
        self.escape_distances = np.sqrt(np.sum(self.escape_offsets ** 2, axis=1))
        self.fine_neighbors = np.empty((0, 3), dtype=np.int64)
        self._empty_unknown = np.empty((1, 1, 1), dtype=np.int16)
        # Compile/load before planning begins; compilation is initialization,
        # while every live expansion still checks the original clock.
        original_clock, self.clock = self.clock, lambda: 1.0
        self._run((0, 0, 0), (1, 0, 0), (0, 0, 0), self.map.inflation_resolution,
                  True, False, False, False, 1.0, 0.0, 0.0)
        self.clock = original_clock

    def set_fine_inf_neighbors(self, step):
        neighbors = spherical_neighbors(int(step))
        self.fine_neighbors = neighbors[np.any(neighbors != 0, axis=1)]

    def _run(self, start, goal, center, resolution, inflated, fine, unknown, escape,
             horizon, started, timeout):
        m = self.map
        parameters = np.array([m.resolution, m.inflation_resolution, m.l_free, m.l_occ,
                               m.virtual_ground_height, m.virtual_ceil_height,
                               m.inf_ground, m.inf_ceil,
                               m.inflation_resolution * (1 + m.config.inflation_step)])
        search_kernel.active_clock = self.clock
        path, status, expanded, cost = search_kernel.run(
            start, goal, center, self.half_size,
            self.escape_offsets if escape else self.offsets,
            self.escape_distances if escape else self.distances, self.fine_neighbors,
            resolution, inflated, fine, unknown, escape, self.heuristic_type,
            float(horizon), float(started), -1.0 if timeout is None else float(timeout), m.values,
            m.inflated_count, m.unknown_inflated_count if m.config.unknown_inflation else self._empty_unknown,
            m.lower_index, m.upper_index, m.inf_lower_index, parameters,
            (m.ground_index, m.ceil_index), m.config.unknown_inflation, self._node_lookup, self.clock_buffer)
        return SearchResult((path + 0.5) * resolution,
                            ("NO_PATH", "REACH_GOAL", "REACH_HORIZON", "TIME_OUT")[status],
                            expanded, cost, self.clock() - started)

    def search(self, start, goal, *, horizon=9999.0, inflated=True, unknown_as_occupied=False,
               use_inf_neighbor=False, time_out=0.1):
        started = self.clock()
        start, goal = np.asarray(start, float), np.asarray(goal, float)
        resolution = self.map.inflation_resolution if inflated else self.map.resolution
        center = start if horizon > 0 else (start + goal) / 2
        center_id = tuple(map(int, self.map.world_to_grid(center, inflated=inflated)))
        local_min = center - resolution * np.asarray(self.half_size)
        local_max = center + resolution * np.asarray(self.half_size)
        local_start, local_goal = start.copy(), goal.copy()
        start_outside = False

        def inside(index):
            return all(abs(index[k] - center_id[k]) <= self.half_size[k] for k in range(3))

        def clip_toward(point, seed, extension):
            d = seed - point
            lo = np.divide(local_min - point, d, out=np.full(3, -np.inf), where=d != 0)
            hi = np.divide(local_max - point, d, out=np.full(3, np.inf), where=d != 0)
            enter = float(np.max(np.minimum(lo, hi)))
            hit = point + enter * d
            distance = float(np.linalg.norm(hit - point))
            return point + (hit - point) / distance * (distance + extension)

        if not inside(np.floor(local_start / resolution).astype(int)):
            local_start = clip_toward(start, center, 2 * resolution)
            local_start = self.map.nearest_inf_not_occupied(local_start, 3.0)
            start_outside = True
        if not inside(np.floor(local_goal / resolution).astype(int)):
            local_goal = clip_toward(goal, center if start_outside else start, 2.5)
            local_goal = self.map.nearest_inf_not_occupied(local_goal, 2.0)
        if local_start is None or local_goal is None:
            return SearchResult(np.empty((0, 3)), "INIT_ERROR", 0, 0.0, self.clock() - started)
        start_id = tuple(map(int, self.map.world_to_grid(local_start, inflated=inflated)))
        goal_id = tuple(map(int, self.map.world_to_grid(local_goal, inflated=inflated)))
        if not inside(start_id) or not inside(goal_id):
            return SearchResult(np.empty((0, 3)), "INIT_ERROR", 0, 0.0, self.clock() - started)
        result = self._run(start_id, goal_id, center_id, resolution, inflated, use_inf_neighbor,
                           unknown_as_occupied, False, horizon, started, time_out)
        if result.success and start_outside:
            result.path = np.vstack(((np.floor(start / resolution) + 0.5) * resolution, result.path))
        return result

    def escape_path_search(self, start, *, unknown_as_occupied=False, inflated=False, time_out=0.2):
        started = self.clock()
        start = np.asarray(start, float)
        if not self.map.inside(start):
            return SearchResult(np.empty((0, 3)), "INIT_ERROR", 0, 0.0)
        local_start = self.map.nearest_cell_not_occupied(start, 3.0)
        if local_start is None:
            return SearchResult(np.empty((0, 3)), "INIT_ERROR", 0, 0.0)
        resolution = self.map.inflation_resolution if inflated else self.map.resolution
        start_id = tuple(map(int, self.map.world_to_grid(local_start, inflated=inflated)))
        center_id = tuple(map(int, self.map.world_to_grid(start, inflated=inflated)))
        return self._run(start_id, start_id, center_id, resolution, inflated, False,
                         unknown_as_occupied, True, -1.0, started, time_out)
