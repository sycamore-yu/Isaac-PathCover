"""PathCover occupancy_inflation: lattice ellipsoid/box dilation and shell."""

from __future__ import annotations

import math
import numpy as np
from scipy.ndimage import binary_erosion, maximum_filter1d


class OccupancyInflation:
    """Source-equivalent one-shot/incremental inflation; no GPU/native backend."""

    def __init__(self, resolution=.1, radius_xy=.45, radius_z=.25,
                 shape="ellipsoid", origin=(0., 0., 0.), surface_only=True,
                 surface_thickness=1, incremental=False,
                 input_bounds=((-1000., -1000., 0.), (1000., 1000., 1000.)),
                 output_bounds=None):
        if resolution <= 0 or min(radius_xy, radius_z) < 0:
            raise ValueError("resolution must be positive and radii nonnegative")
        if shape not in ("ellipsoid", "box") or surface_thickness < 1:
            raise ValueError("invalid inflation shape or surface thickness")
        self.resolution = float(resolution)
        self._inverse = 1. / self.resolution
        self.origin = np.asarray(origin, dtype=np.float64)
        self.surface_only, self.surface_thickness = surface_only, surface_thickness
        self.incremental = incremental
        self.input_bounds = None if input_bounds is None else np.asarray(input_bounds, dtype=float)
        self.output_bounds = None if output_bounds is None else np.asarray(output_bounds, dtype=float)
        radii = np.array([radius_xy, radius_xy, radius_z])
        reach = np.ceil(radii / resolution).astype(int)
        runs = []
        for y in range(-reach[1], reach[1] + 1):
            for z in range(-reach[2], reach[2] + 1):
                span = -1
                for x in range(reach[0] + 1):
                    physical = np.array([x, y, z]) * resolution
                    if np.any((radii == 0) & (physical != 0)):
                        continue
                    ratios = np.divide(physical, radii, out=np.zeros(3), where=radii > 0)
                    if shape == "box" or float(ratios @ ratios) <= 1.:
                        span = x
                if span >= 0:
                    runs.append((y, z, span))
        self.kernel_runs = runs
        self.kernel = np.array([(x, y, z) for y, z, span in runs for x in range(-span, span + 1)], dtype=np.int64)
        self.reach = np.max(np.abs(self.kernel), axis=0)
        self.seed_indices = np.empty((0, 3), dtype=np.int64)
        self.solid_indices = self.seed_indices.copy()
        self.output_indices = self.seed_indices.copy()
        self.stats = {}

    def _indices(self, points):
        # ROS PointCloud2 is float32; the source adds a bias before truncation.
        points = np.asarray(points, dtype=np.float32).astype(np.float64)
        return (((points - self.origin) * self._inverse + 1048576.).astype(np.int64) - 1048576)

    def inflate(self, points):
        points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        valid = np.isfinite(points).all(axis=1)
        if self.input_bounds is not None:
            valid &= ((points >= self.input_bounds[0]) & (points <= self.input_bounds[1])).all(axis=1)
        seeds = np.unique(self._indices(points[valid]), axis=0)
        if self.incremental:
            seeds = np.unique(np.concatenate((self.seed_indices, seeds)), axis=0)
        changed = not np.array_equal(seeds, self.seed_indices)
        self.stats = {"changed": changed, "seed_voxels": len(seeds), "kernel_voxels": len(self.kernel)}
        if not changed:
            return self.points()
        self.seed_indices = seeds
        if len(seeds) == 0:
            self.solid_indices = seeds.copy()
            self.output_indices = seeds.copy()
            return self.points()
        lower = seeds.min(axis=0) - self.reach
        upper = seeds.max(axis=0) + self.reach + 1
        dimensions = upper - lower
        seed_grid = np.zeros(tuple(dimensions), dtype=bool)
        local = seeds - lower
        seed_grid[tuple(local.T)] = True
        solid = np.zeros_like(seed_grid)
        # The source ellipsoid is a union of x-axis runs. Reuse each run length
        # across its (y,z) shifts, retaining the exact lattice structuring set.
        for span in sorted({span for _, _, span in self.kernel_runs}):
            x_run = maximum_filter1d(seed_grid, size=2 * span + 1, axis=0, mode="constant", cval=0)
            for y, z, length in self.kernel_runs:
                if length != span:
                    continue
                source, target = [slice(None)], [slice(None)]
                for size, shift in zip(dimensions[1:], (y, z)):
                    if shift >= 0:
                        source.append(slice(0, size - shift)); target.append(slice(shift, size))
                    else:
                        source.append(slice(-shift, size)); target.append(slice(0, size + shift))
                solid[tuple(target)] |= x_run[tuple(source)]
        if self.output_bounds is not None:
            xyz = np.argwhere(solid)
            centers = (xyz + lower + .5) * self.resolution + self.origin
            keep = ((centers >= self.output_bounds[0]) & (centers <= self.output_bounds[1])).all(axis=1)
            remove = xyz[~keep]
            solid[remove[:, 0], remove[:, 1], remove[:, 2]] = False
        self.solid_indices = np.argwhere(solid) + lower
        if self.surface_only:
            width = 2 * self.surface_thickness + 1
            interior = binary_erosion(solid, structure=np.ones((width, width, width), bool), border_value=0)
            solid &= ~interior
        self.output_indices = np.argwhere(solid) + lower
        self.stats.update(inflated_voxels=len(self.solid_indices), output_voxels=len(self.output_indices))
        return self.points()

    def points(self, *, solid=False):
        indices = self.solid_indices if solid else self.output_indices
        return ((indices + .5) * self.resolution + self.origin).astype(np.float32)

    def reset(self):
        self.seed_indices = np.empty((0, 3), dtype=np.int64)
        self.solid_indices = self.seed_indices.copy()
        self.output_indices = self.seed_indices.copy()
