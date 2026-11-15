"""Point cloud → NanoVoxMap → inflation → JPS/DMP → PathCover → GCOPTER.

The default resolutions and geometry settings follow the upstream drone
configuration. The L2F physical parameters are the sole airframe adaptation.
Known-static benchmarks load a point cloud through NanoVoxMap.setOccupied;
subsequent sensor scans use the same map's ray fusion.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.spatial import QhullError

from ..mapping import NanoVoxMap, OccupancyInflation
from ..trajectory import GCopter, OptimizationResult, PiecewisePolynomial, TrajectoryConfig
from .corridor import Corridor, find_start_polytope, path_cover
from .geometry import GeometryError
from .search import DistanceMapPlanner, JPSPlanner, SearchFailure, SearchResult, VoxelGrid


@dataclass
class PlanResult:
    trajectory: PiecewisePolynomial | None
    corridor: Corridor | None
    search: SearchResult | None
    optimization: OptimizationResult | None
    timings: dict[str, float]
    success: bool
    status: str
    map_version: int


class Planner:
    """One planning worker with persistent map and random state.

    ``plan`` never replaces an executing trajectory. Its caller owns the
    upstream rule: replace on success and retain the active plan on failure.
    """

    def __init__(self, scene, seed=0, *, config=None, max_horizon=6,
                 inflation_radius_xy=0.45, inflation_radius_z=0.25):
        started = perf_counter()
        self.scene = scene
        self.map = NanoVoxMap()
        self.map.set_occupied(scene.point_cloud())
        self.initialization_timings = {"point_cloud_mapping_s": perf_counter() - started}
        # The upstream cumulative-inflation mode cannot remove cleared cells.
        # Track exact coverage counts so ray clearing remains correct.
        self.inflation = OccupancyInflation(radius_xy=inflation_radius_xy,
                                            radius_z=inflation_radius_z, exact_updates=True)
        self.config = config or TrajectoryConfig(physical_parameters=(0.752, 9.8066, 0.0, 0.0, 0.0, 0.001))
        self.max_horizon = max_horizon
        self.rng = np.random.default_rng(seed)
        self._map_version = -1
        self.inflated_points = np.empty((0, 3))
        self.grid = self.jps = self.dmp = None
        self.previous_path = None

    @staticmethod
    def _safe_suffix(previous_path, position, goal, grid):
        """Reuse a map-valid remaining guide path, as in SUPER's front end."""
        if previous_path is None or len(previous_path) < 2:
            return None
        if np.linalg.norm(previous_path[-1] - goal) > 1e-6:
            return None
        segments = np.diff(previous_path, axis=0)
        lengths_sq = np.sum(segments * segments, axis=1)
        fractions = np.clip(np.sum((position - previous_path[:-1]) * segments, axis=1)
                            / np.maximum(lengths_sq, 1e-12), 0.0, 1.0)
        closest = previous_path[:-1] + fractions[:, None] * segments
        distances = np.linalg.norm(closest - position, axis=1)
        segment = int(np.argmin(distances))
        if distances[segment] > 0.5:
            return None
        suffix = np.vstack((position, closest[segment], previous_path[segment + 1:]))
        delta = np.diff(suffix, axis=0)
        # Validate the entire suffix against the CURRENT inflated search grid,
        # including the new link from measured position to the old path.
        samples = np.vstack([a + np.linspace(0., 1., max(2, int(np.ceil(np.linalg.norm(d) /
                                      (grid.resolution / 4))) + 1))[:, None] * d
                             for a, d in zip(suffix[:-1], delta)])
        ids = grid.float_to_int(samples)
        if not ((ids >= 0) & (ids < grid.shape)).all():
            return None
        if np.any(grid.values[tuple(ids.T)] != 0):
            return None
        return suffix[np.r_[True, np.linalg.norm(np.diff(suffix, axis=0), axis=1) > 1e-9]]

    def update_map(self, origin, endpoints):
        """Integrate one actual scan; defer dependent rebuilds until planning."""
        started, before = perf_counter(), self.map.version
        self.map.insert_point_cloud(origin, endpoints)
        return {"fusion_s": perf_counter() - started,
                "occupancy_changed": self.map.version != before,
                "map_version": self.map.version}

    def _refresh_map(self):
        timings = {"map_snapshot_s": 0.0, "inflation_s": 0.0, "grid_s": 0.0}
        if self._map_version == self.map.version:
            return timings
        started = perf_counter()
        points = self.map.occupied_points()
        timings["map_snapshot_s"] = perf_counter() - started
        started = perf_counter()
        self.inflated_points = self.inflation.inflate(points).astype(np.float64)
        timings["inflation_s"] = perf_counter() - started
        started = perf_counter()
        self.grid = VoxelGrid.from_points(self.inflated_points, self.scene.lower, self.scene.upper)
        self.jps = JPSPlanner(self.grid, heuristic_weight=self.grid.resolution)
        self.dmp = DistanceMapPlanner(self.grid)
        timings["grid_s"] = perf_counter() - started
        self._map_version = self.map.version
        return timings

    def plan(self, start_state, goal=None, *, guide=None, guide_elapsed=0.0):
        """Plan from columns position/velocity/acceleration of a (3,3) array."""
        start_state = np.asarray(start_state, dtype=float)
        if start_state.shape != (3, 3) or not np.all(np.isfinite(start_state)):
            raise ValueError("start_state must have finite (3,3) columns p,v,a")
        goal = np.asarray(self.scene.goal if goal is None else goal, dtype=float)
        if goal.shape != (3,) or not np.all(np.isfinite(goal)):
            raise ValueError("goal must have three finite coordinates")
        started = perf_counter()
        timings = self._refresh_map()
        search = corridor = optimization = None
        status, success, trajectory = "", False, None
        stage = "search"
        try:
            position = start_state[:, 0]
            # convertAndFilter removes self-hits in XY even at FilterRadius=0.
            keep = np.any(self.inflated_points[:, :2] != position[:2], axis=1)
            points = self.inflated_points if keep.all() else self.inflated_points[keep]
            jps, dmp = self.jps, self.dmp
            if not keep.all():
                grid = VoxelGrid.from_points(points, self.scene.lower, self.scene.upper)
                jps, dmp = JPSPlanner(grid, heuristic_weight=grid.resolution), DistanceMapPlanner(grid)
            if np.linalg.norm(position - goal) <= 1e-3:
                search = SearchResult(position[None], position[None], position[None], 0.0,
                                      {"jps": 0, "dmp": 0}, {"jps_s": 0.0, "dmp_s": 0.0})
            else:
                reused = self._safe_suffix(self.previous_path, position, goal, jps.grid)
                if reused is not None:
                    search = SearchResult(reused, reused, reused[1:-1],
                                          float(np.linalg.norm(np.diff(reused, axis=0), axis=1).sum()),
                                          {"jps": 0, "dmp": 0}, {"jps_s": 0.0, "dmp_s": 0.0})
                    timings["search_reused"] = 1.0
                else:
                    t = perf_counter()
                    jps_path = jps.plan(position, goal)
                    timings["jps_s"] = perf_counter() - t
                    t = perf_counter()
                    dmp_path = dmp.plan(position, goal, jps_path)
                    timings["dmp_s"] = perf_counter() - t
                    search = SearchResult(np.vstack((position, dmp_path, goal)), jps_path, dmp_path,
                                          dmp.cost, {"jps": jps.expanded, "dmp": dmp.expanded},
                                          {"jps_s": timings["jps_s"], "dmp_s": timings["dmp_s"]})
                self.previous_path = search.path.copy()
            stage = "corridor"
            t = perf_counter()
            corridor = path_cover(points, search.path, self.scene.bounds_halfspaces(),
                                  max_horizon=self.max_horizon, rng=self.rng)
            start_index = find_start_polytope(position, corridor.polytopes)
            if start_index:
                corridor = Corridor(corridor.polytopes[start_index:], corridor.seeds[start_index:],
                                    corridor.remaining_counts[start_index:], start_index)
            timings["corridor_s"] = perf_counter() - t
            # The published terminal seed is the local goal after horizon truncation.
            end_state = np.zeros((3, 3))
            end_state[:, 0] = corridor.goal
            stage = "optimizer_setup"
            t = perf_counter()
            optimizer = GCopter(corridor.polytopes, start_state, end_state, self.config)
            timings["optimizer_setup_s"] = perf_counter() - t
            stage = "optimization"
            t = perf_counter()
            optimization = optimizer.optimize(guide=guide, elapsed=guide_elapsed)
            timings["optimization_s"] = perf_counter() - t
            trajectory, success = optimization.trajectory, optimization.success
            status = optimization.status
        except (SearchFailure, GeometryError, QhullError, np.linalg.LinAlgError) as error:
            status = f"{stage}: {type(error).__name__}: {error}"
        timings["total"] = perf_counter() - started
        return PlanResult(trajectory, corridor, search, optimization, timings,
                          success, status, self.map.version)
