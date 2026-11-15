"""SUPER CorridorGenerator with CIRI replaced by RISP (2ad3419)."""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
from .geometry import GeometryError, chebyshev_center
from .risp import risp_segment


@dataclass
class Polytope:
    planes: np.ndarray
    seed_line: tuple[np.ndarray, np.ndarray] | None = None
    overlap_depth: float = 0.0
    interior: np.ndarray = field(default_factory=lambda: np.zeros(3))
    known_free: bool = False
    remaining_counts: list[int] = field(default_factory=list)

    def __post_init__(self):
        self.planes = np.asarray(self.planes, float).reshape(-1, 4)

    @property
    def A(self):
        return self.planes[:, :3]

    @property
    def b(self):
        return -self.planes[:, 3]

    def contains(self, point, margin=0.01):
        return bool(np.max(self.A @ np.asarray(point) - self.b) <= margin)

    def cross(self, other):
        return Polytope(np.vstack((self.planes, other.planes)))


@dataclass
class Corridor:
    regions: list[Polytope]
    seeds: np.ndarray
    shifted_start: np.ndarray

    @property
    def polytopes(self):
        return [(poly.A, poly.b) for poly in self.regions]

    @property
    def remaining_counts(self):
        return [poly.remaining_counts for poly in self.regions]

    @property
    def goal(self):
        return self.seeds[-1]


def find_interior(planes):
    """SUPER findInteriorDist; signed radius also represents disjoint sets."""
    planes = np.asarray(planes, float)
    return chebyshev_center(planes[:, :3], -planes[:, 3], signed=True)


def _planes(poly):
    return np.asarray(getattr(poly, "planes", poly), float)


def _contains(poly, point):
    planes = _planes(poly)
    return bool(np.max(planes[:, :3] @ np.asarray(point) + planes[:, 3]) <= 0.01)


def simplify_sfc(corridors, head, tail):
    """Exact selection loop of SUPER data_structure/base/polytope.h."""
    sfcs = list(corridors.regions if isinstance(corridors, Corridor) else corridors)
    if len(sfcs) <= 2:
        return sfcs
    start_id, end_id = -1, -1
    for i in range(len(sfcs)):
        if _contains(sfcs[i], head):
            start_id = i
        reverse = len(sfcs) - 1 - i
        if _contains(sfcs[reverse], tail):
            end_id = reverse
    if start_id < 0 or end_id < 0:
        raise GeometryError("SUPER SimplifySFC: head or tail outside corridor")
    end_id = max(end_id, start_id)
    sfcs = sfcs[start_id:end_id + 1]
    if len(sfcs) <= 2:
        return sfcs
    candidate, last = sfcs[0], sfcs[1]
    output = [sfcs[0]]
    i = 2
    while i < len(sfcs):
        _, radius = find_interior(np.vstack((_planes(candidate), _planes(sfcs[i]))))
        if radius > 0:
            last = sfcs[i]
            if _contains(last, tail):
                output.append(last)
                break
            i += 1
        else:
            output.append(last)
            candidate = last
    return output


def _box_planes(lo, hi):
    return np.column_stack((np.vstack((np.eye(3), -np.eye(3))), np.r_[-hi, lo]))


class CorridorGenerator:
    def __init__(self, map, *, bound_distance=0.8, line_max_length=1.0, min_overlap=None,
                 robot_radius=0.2, line_neighbors=None, seed=0, rng=None, alpha=0.01):
        self.map = map
        self.bound_distance = float(bound_distance)
        self.line_max_length = float(line_max_length)
        self.min_overlap = map.resolution if min_overlap is None else float(min_overlap)
        self.robot_radius = float(robot_radius)
        self.line_neighbors = np.empty((0, 3), dtype=int) if line_neighbors is None else np.asarray(line_neighbors)
        self.rng = np.random.default_rng(seed) if rng is None else rng
        self.alpha = alpha
        self.latest_cloud = []
        self.geometry_seconds = 0.0
        self.geometry_calls = 0

    def generate_empty(self, point, distance):
        point = np.asarray(point, float)
        return Polytope(_box_planes(point - distance, point + distance), (point.copy(), point.copy()))

    def generate_from_point(self, point):
        return self.generate_from_line(point, point)

    def generate_from_line(self, start, end):
        start, end = np.asarray(start, float), np.asarray(end, float)
        lo = np.minimum(start, end) - self.bound_distance
        hi = np.maximum(start, end) + self.bound_distance
        lo, hi = self.map.bound_box(lo, hi)
        points = self.map.box_search(lo, hi)
        lo[2] += self.robot_radius
        hi[2] -= self.robot_radius
        planes = _box_planes(lo, hi)
        if not len(points):
            return Polytope(planes, (start.copy(), end.copy()))
        self.latest_cloud.append(points)
        started = perf_counter()
        result = risp_segment(points, start, end, (planes[:, :3], -planes[:, 3]),
                              rng=self.rng, alpha=self.alpha, robot_radius=self.robot_radius)
        self.geometry_seconds += perf_counter() - started
        self.geometry_calls += 1
        return Polytope(np.column_stack((result.A, -result.b)), (start.copy(), end.copy()),
                        remaining_counts=result.remaining_counts)

    def search_polytope_on_path(self, path, cut_first_poly=False):
        path = np.asarray(path, float).reshape(-1, 3)
        if not len(path):
            raise GeometryError("SUPER corridor input path is empty")
        regions, seeds = [], []
        # SUPER leaves this caller sentinel intact unless inflation shifts the start.
        shifted_start = np.full(3, 9999.0)
        first = 0
        while first < len(path) and self.map.is_occupied_inflate(path[first]):
            first += 1
        if first == len(path):
            raise GeometryError("SUPER corridor path has no start outside inflation")
        if first:
            shifted_start = path[first].copy()
            distance = np.linalg.norm(path[first] - path[0]) * 1.2
            regions.append(self.generate_empty(path[0], distance))
            seeds.append(path[0].copy())
        count = 0
        while count < 1000:
            count += 1
            second = first
            for j in range(first + 1, len(path)):
                if not self.map.is_line_free(path[first], path[j], self.line_max_length, self.line_neighbors):
                    second = j - 1
                    if second - 1 > first:
                        second -= 1
                    break
                second = j
            if second == first and second + 1 < len(path):
                second += 1
            if np.linalg.norm(path[first] - path[second]) > self.line_max_length * 1.5:
                raise GeometryError("SUPER seed line too long")
            poly = self.generate_from_line(path[first], path[second])
            if regions:
                interior, radius = find_interior(np.vstack((regions[-1].planes, poly.planes)))
                poly.overlap_depth, poly.interior = radius, interior
                if radius < self.min_overlap:
                    fix = self.generate_from_point(path[first])
                    interior, radius = find_interior(np.vstack((regions[-1].planes, fix.planes)))
                    if radius <= 0.01:
                        raise GeometryError(f"SUPER corridor overlap only {radius:g}")
                    fix.overlap_depth, fix.interior = radius, interior
                    regions.append(fix)
                    seeds.append(path[first].copy())
                    _, radius = find_interior(np.vstack((fix.planes, poly.planes)))
                    if radius <= 0.01:
                        raise GeometryError(f"SUPER corridor overlap only {radius:g}")
                else:
                    temp_id = len(regions) - 2
                    if temp_id > 0:
                        interior, radius = find_interior(np.vstack((regions[temp_id].planes, poly.planes)))
                        if radius > regions[temp_id + 1].overlap_depth * 0.25:
                            poly.overlap_depth, poly.interior = radius, interior
                            regions.pop()
                            seeds.pop()
            regions.append(poly)
            seeds.append(path[first].copy())
            if second == len(path) - 1:
                break
            first = second
        if count >= 1000:
            raise GeometryError("SUPER corridor reached its original 1000-iteration limit")
        return Corridor(regions, np.vstack((seeds, path[-1])), shifted_start)

    def get_latest_cloud(self):
        cloud = np.vstack(self.latest_cloud) if self.latest_cloud else np.empty((0, 3))
        self.latest_cloud.clear()
        return cloud
