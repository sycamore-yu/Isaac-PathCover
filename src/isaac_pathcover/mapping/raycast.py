"""CPU compilation of SUPER's ROG raw-map line and seed-neighbor query.

Port of SUPER 2ad3419 rog_map.cpp and rog_map_core/raycaster.cpp, LGPL-3.0.
Copyright 2025 Yunfan REN, MaRS Lab, HKU.
The source remains Python. Numba compiles the sequential DDA and occupancy
loop; floating-point reassociation and parallel execution are disabled.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, types
from numba.typed import Dict

_CELL_KEY = types.UniTuple(types.int64, 3)


@njit(cache=True, fastmath=False, nogil=True)
def raw_line_free(start, end, resolution, max_distance, neighbors, values,
                  lower, upper, occupied_threshold, ground_index, ceil_index,
                  ground_height, ceil_height, cache=None):
    """ROGMap::isLineFree with raw probability cells and optional neighbors."""
    x = int(math.floor(start[0] / resolution))
    y = int(math.floor(start[1] / resolution))
    z = int(math.floor(start[2] / resolution))
    end_x = int(math.floor(end[0] / resolution))
    end_y = int(math.floor(end[1] / resolution))
    end_z = int(math.floor(end[2] / resolution))
    dx = (end_x > x) - (end_x < x)
    dy = (end_y > y) - (end_y < y)
    dz = (end_z > z) - (end_z < z)
    if dx == 0 and dy == 0 and dz == 0:
        return True, end[0], end[1], end[2]
    extent_x = abs(end[0] - start[0])
    extent_y = abs(end[1] - start[1])
    extent_z = abs(end[2] - start[2])
    length = math.sqrt(extent_x * extent_x + extent_y * extent_y + extent_z * extent_z)
    unit_x, unit_y, unit_z = extent_x / length, extent_y / length, extent_z / length
    tx = ty = tz = np.finfo(np.float64).max
    step_x = step_y = step_z = np.finfo(np.float64).max
    if dx != 0:
        step_x = resolution / unit_x
        tx = abs((x + .5) * resolution + dx * resolution * .5 - start[0]) / unit_x
    if dy != 0:
        step_y = resolution / unit_y
        ty = abs((y + .5) * resolution + dy * resolution * .5 - start[1]) / unit_y
    if dz != 0:
        step_z = resolution / unit_z
        tz = abs((z + .5) * resolution + dz * resolution * .5 - start[2]) / unit_z
    last_x, last_y, last_z = start[0], start[1], start[2]
    while x != end_x or y != end_y or z != end_z:
        last_x = (x + .5) * resolution
        last_y = (y + .5) * resolution
        last_z = (z + .5) * resolution
        rx, ry, rz = last_x - start[0], last_y - start[1], last_z - start[2]
        if max_distance > 0. and math.sqrt(rx * rx + ry * ry + rz * rz) > max_distance:
            return False, last_x, last_y, last_z
        if len(neighbors):
            key = (x, y, z)
            if cache is not None and key in cache:
                occupied = cache[key]
            else:
                occupied = False
                for j in range(len(neighbors)):
                    nx, ny, nz = x + neighbors[j, 0], y + neighbors[j, 1], z + neighbors[j, 2]
                    if (lower[0] <= nx <= upper[0] and lower[1] <= ny <= upper[1]
                            and lower[2] <= nz <= upper[2]):
                        if (nz < ground_index or nz > ceil_index
                                or values[nx - lower[0], ny - lower[1], nz - lower[2]] >= occupied_threshold):
                            occupied = True
                            break
                if cache is not None:
                    cache[key] = occupied
            if occupied:
                return False, last_x, last_y, last_z
        elif (lower[0] <= x <= upper[0] and lower[1] <= y <= upper[1]
                and lower[2] <= z <= upper[2]):
            if (last_z < ground_height or last_z > ceil_height
                    or values[x - lower[0], y - lower[1], z - lower[2]] >= occupied_threshold):
                return False, last_x, last_y, last_z
        if tx < ty:
            if tx < tz:
                x += dx
                tx += step_x
            else:
                z += dz
                tz += step_z
        elif ty < tz:
            y += dy
            ty += step_y
        else:
            z += dz
            tz += step_z
    return True, end[0], end[1], end[2]


@njit(cache=True, fastmath=False, nogil=True)
def raw_lines_first_blocked(start, ends, resolution, max_distance, neighbors, values,
                             lower, upper, occupied_threshold, ground_index, ceil_index,
                             ground_height, ceil_height):
    """Evaluate lines in source order, stopping at the first blocked ray."""
    cache = Dict.empty(key_type=_CELL_KEY, value_type=types.boolean)
    for i in range(len(ends)):
        free, _, _, _ = raw_line_free(start, ends[i], resolution, max_distance, neighbors,
                                      values, lower, upper, occupied_threshold, ground_index,
                                      ceil_index, ground_height, ceil_height, cache)
        if not free:
            return i
    return len(ends)
