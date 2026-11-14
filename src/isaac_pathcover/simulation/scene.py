"""Immutable scenario solids and a finite-resolution geometric LiDAR."""
from __future__ import annotations

from importlib.resources import files
import json

import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial.transform import Rotation


class Scene:
    def __init__(self, name="medium"):
        if name not in ("medium", "dense"):
            raise ValueError("scene must be medium or dense")
        self.name = name
        assets = files(__package__).joinpath("scenes")
        self.provenance = json.loads(assets.joinpath("provenance.json").read_text())["scenarios"][name]
        with np.load(assets.joinpath(f"{name}.npz")) as data:
            self.occupied = data["occupied"]
            self.boxes = data["boxes"]
            self.start = data["start"]
            self.goal = data["goal"]
            self.lower = data["origin"]
            self.resolution = float(data["resolution"])
            self.body_size = data["body_size"]
        self.upper = self.lower + self.resolution * np.asarray(self.occupied.shape)
        self.box_lower = self.lower + self.boxes[:, :3] * self.resolution
        self.box_upper = self.lower + self.boxes[:, 3:] * self.resolution

    def point_cloud(self):
        shell = self.occupied & ~binary_erosion(self.occupied, structure=np.ones((3, 3, 3)), border_value=0)
        return self.lower + (np.argwhere(shell) + .5) * self.resolution

    def contains_obstacle(self, points):
        indices = np.floor((np.asarray(points) - self.lower) / self.resolution).astype(int)
        inside = ((indices >= 0) & (indices < self.occupied.shape)).all(axis=-1)
        result = np.zeros(inside.shape, dtype=bool)
        valid = indices[inside]
        result[inside] = self.occupied[tuple(valid.T)]
        return result

    def lidar(self, position, quaternion_wxyz, *, horizontal_rays=440, vertical_rays=16,
              min_range=.1, max_range=100.):
        """Cast VLP-16 style rays against the same raw solids used by Newton.

        The upstream mount is 0.25 m above the body origin. Rays rotate with the
        measured body attitude. Return the sensor origin and first-hit points;
        the planner never receives the simulation occupancy grid.
        """
        azimuth, elevation = np.meshgrid(np.linspace(-np.pi, np.pi, horizontal_rays, endpoint=False),
                                       np.linspace(-np.pi / 12, np.pi / 12, vertical_rays))
        direction = np.column_stack((np.cos(elevation.ravel()) * np.cos(azimuth.ravel()),
                                     np.cos(elevation.ravel()) * np.sin(azimuth.ravel()),
                                     np.sin(elevation.ravel())))
        rotation = Rotation.from_quat(np.asarray(quaternion_wxyz)[[1, 2, 3, 0]]).as_matrix()
        origin = np.asarray(position, dtype=float) + rotation @ np.array([0., 0., .25])
        direction = direction @ rotation.T
        indices = np.broadcast_to(np.floor((origin - self.lower) / self.resolution).astype(int), direction.shape).copy()
        step = np.sign(direction).astype(int)
        inv = np.divide(1., direction, out=np.full_like(direction, np.inf), where=np.abs(direction) > 1e-12)
        delta = np.abs(self.resolution * inv)
        boundary = self.lower + (indices + (step > 0)) * self.resolution
        tmax = np.multiply(boundary - origin, inv, out=np.full_like(inv, np.inf), where=np.isfinite(inv))
        distance = np.zeros(len(direction))
        alive = np.ones(len(direction), dtype=bool)
        hits = np.zeros(len(direction), dtype=bool)
        shape = np.asarray(self.occupied.shape)
        for _ in range(int(np.ceil(max_range / self.resolution) * 3) + 3):
            ids = np.flatnonzero(alive)
            if not len(ids):
                break
            cells = indices[ids]
            inside = ((cells >= 0) & (cells < shape)).all(axis=1)
            alive[ids[~inside]] = False
            ids = ids[inside]
            if not len(ids):
                continue
            occupied = self.occupied[tuple(indices[ids].T)]
            hits[ids[occupied]] = True
            alive[ids[occupied]] = False
            ids = ids[~occupied]
            axes = np.argmin(tmax[ids], axis=1)
            distance[ids] = tmax[ids, axes]
            tmax[ids, axes] += delta[ids, axes]
            indices[ids, axes] += step[ids, axes]
            alive[ids[distance[ids] >= max_range]] = False
        hits &= distance >= min_range
        return origin, origin + direction[hits] * (distance[hits, None] + 1e-7)

    def bounds_halfspaces(self, margin=0.):
        return np.r_[np.eye(3), -np.eye(3)], np.r_[self.upper - margin, -(self.lower + margin)]
