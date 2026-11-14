"""Native 3-D JPS and distance-map planning from PathCover's jps_lib.

Source: PathCover 8723a05d, third_party/jps_lib and planner/VoxelGrid.hpp.
The 26-neighbour graph permits diagonal steps exactly as the source does.
JPS jump memoization and NumPy potential-field updates preserve that graph.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import itertools
import math
import time
import numpy as np
from scipy.ndimage import binary_dilation, grey_dilation, maximum_filter1d


class SearchFailure(RuntimeError):
    pass


@dataclass
class VoxelGrid:
    values: np.ndarray
    origin: np.ndarray
    resolution: float

    def __post_init__(self):
        self.values = np.asarray(self.values, dtype=np.int8)
        self.origin = np.asarray(self.origin, dtype=np.float64)
        if self.values.ndim != 3 or self.origin.shape != (3,) or self.resolution <= 0:
            raise ValueError("expected a 3-D occupancy grid, xyz origin and positive resolution")
        self.shape = np.array(self.values.shape)

    @classmethod
    def from_points(cls, points, lower, upper, resolution=.25):
        lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
        if np.any(upper <= lower):
            raise ValueError("upper bounds must exceed lower bounds")
        shape = np.ceil((upper - lower) / resolution).astype(int)
        values = np.zeros(tuple(shape), dtype=np.int8)
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1)]
        indices = np.floor((points - lower) / resolution).astype(np.int64)
        indices = indices[((indices >= 0) & (indices < shape)).all(axis=1)]
        if len(indices):
            values[tuple(indices.T)] = 100
        return cls(values, lower, resolution)

    def float_to_int(self, points):
        shifted = (np.asarray(points) - self.origin) / self.resolution - .5
        return np.where(shifted >= 0, np.floor(shifted + .5), np.ceil(shifted - .5)).astype(np.int64)

    def int_to_float(self, indices):
        return (np.asarray(indices) + .5) * self.resolution + self.origin

    def is_free(self, index):
        return bool(np.all((index >= 0) & (index < self.shape)) and self.values[tuple(index)] == 0)

    def ray_trace(self, first, second):
        first, second = np.asarray(first), np.asarray(second)
        diff = second - first
        count = int(np.max(np.abs(diff / self.resolution)) / .8)
        if count <= 1:
            return np.empty((0, 3), dtype=np.int64)
        points = first + np.arange(1, count)[:, None] * (diff / count)
        indices = self.float_to_int(points)
        inside = ((indices >= 0) & (indices < self.shape)).all(axis=1)
        if not inside.all():
            indices = indices[:np.flatnonzero(~inside)[0]]
        if len(indices):
            indices = indices[np.r_[True, np.any(indices[1:] != indices[:-1], axis=1)]]
        return indices

    def is_blocked(self, first, second, threshold=100):
        indices = self.ray_trace(first, second)
        return bool(len(indices) and np.any(self.values[tuple(indices.T)] >= threshold))


_DIRECTIONS = tuple(d for d in itertools.product((-1, 0, 1), repeat=3) if d != (0, 0, 0))
_SOURCE_INITIAL = ((1,0,0),(-1,0,0),(0,1,0),(1,1,0),(-1,1,0),(0,-1,0),(1,-1,0),(-1,-1,0),
                   (0,0,1),(1,0,1),(-1,0,1),(0,1,1),(1,1,1),(-1,1,1),(0,-1,1),(1,-1,1),(-1,-1,1),
                   (0,0,-1),(1,0,-1),(-1,0,-1),(0,1,-1),(1,1,-1),(-1,1,-1),(0,-1,-1),(1,-1,-1),(-1,-1,-1))


def jps_neighbours(direction):
    """The source JPS3DNeib natural and forced neighbour table, in source order."""
    dx, dy, dz = direction
    norm = abs(dx) + abs(dy) + abs(dz)
    if norm == 0:
        return _SOURCE_INITIAL, (), ()
    if norm == 1:
        first, second = [], []
        for fx, fy in ((0,1),(0,-1),(1,0),(1,1),(1,-1),(-1,0),(-1,1),(-1,-1)):
            fz, nx, ny, nz = 0, fx, fy, dz
            if dx:
                fz, fx = fx, 0
                nz, nx = fz, dx
            if dy:
                fz, fy = fy, 0
                nz, ny = fz, dy
            first.append((fx, fy, fz)); second.append((nx, ny, nz))
        return (direction,), tuple(first), tuple(second)
    if norm == 2:
        natural = ((0, dy, 0) if dz == 0 else (0, 0, dz),
                   (0, dy, 0) if dx == 0 else (dx, 0, 0), direction)
        active = [i for i, v in enumerate(direction) if v]
        a, b = active
        c = next(i for i, v in enumerate(direction) if not v)
        da, db = direction[a], direction[b]
        # Express all three plane cases in (active-a, active-b, unused) axes.
        f = ((0,-db,0),(-da,0,0),(0,0,1),(0,0,-1),
             (0,-db,1),(-da,0,1),(0,-db,-1),(-da,0,-1),
             (0,0,1),(0,0,1),(0,0,-1),(0,0,-1))
        n = ((da,-db,0),(-da,db,0),(da,db,1),(da,db,-1),
             (da,-db,1),(-da,db,1),(da,-db,-1),(-da,db,-1),
             (da,0,1),(0,db,1),(da,0,-1),(0,db,-1))
        def permute(row):
            out = [0, 0, 0]
            out[a], out[b], out[c] = row
            return tuple(out)
        return natural, tuple(map(permute, f)), tuple(map(permute, n))
    natural = ((dx,0,0),(0,dy,0),(0,0,dz),(dx,dy,0),(dx,0,dz),(0,dy,dz),direction)
    first = ((-dx,0,0),(0,-dy,0),(0,0,-dz),(0,-dy,-dz),(-dx,0,-dz),(-dx,-dy,0),
             (-dx,0,0),(-dx,0,0),(0,-dy,0),(0,-dy,0),(0,0,-dz),(0,0,-dz))
    second = ((-dx,dy,dz),(dx,-dy,dz),(dx,dy,-dz),(dx,-dy,-dz),(-dx,dy,-dz),(-dx,-dy,dz),
              (-dx,0,dz),(-dx,dy,0),(0,-dy,dz),(dx,-dy,0),(0,dy,-dz),(dx,0,-dz))
    return natural, first, second


_NEIGHBOURS = {d: jps_neighbours(d) for d in _DIRECTIONS + ((0, 0, 0),)}


class _QueueItem:
    __slots__ = ("f", "g", "node", "serial")

    def __init__(self, f, g, node, serial):
        self.f, self.g, self.node, self.serial = f, g, node, serial

    def __lt__(self, other):
        if abs(self.f - other.f) <= 1e-6:
            return self.g > other.g if self.g != other.g else self.serial < other.serial
        return self.f < other.f


def _shift_occupancy(occupied, offset):
    result = np.zeros_like(occupied)
    target, source = [], []
    for length, delta in zip(occupied.shape, offset):
        if delta >= 0:
            target.append(slice(0, length - delta)); source.append(slice(delta, length))
        else:
            target.append(slice(-delta, length)); source.append(slice(0, length + delta))
    result[tuple(target)] = occupied[tuple(source)]
    return result


class JPSPlanner:
    def __init__(self, grid: VoxelGrid, heuristic_weight=1.):
        self.grid, self.heuristic_weight = grid, heuristic_weight
        self.expanded = 0
        self.cost = math.inf
        self.raw_path = np.empty((0, 3))
        self.path = self.raw_path.copy()
        self._prepared = False

    def _prepare(self):
        # A blocked padding cell ends jumps; padding is excluded from forced
        # obstacle tests because source isOccupied(outside) returns false.
        occupied = self.grid.values > 0
        self._dims = tuple(self.grid.shape + 2)
        self._strides = (1, self._dims[0], self._dims[0]*self._dims[1])
        self._occupied = np.pad(self.grid.values != 0, 1, constant_values=True).ravel(order="F")
        self._blockers = np.pad(occupied, 1, constant_values=False).ravel(order="F")
        self._directions = _DIRECTIONS + ((0,0,0),)
        direction_ids = {d:i for i,d in enumerate(self._directions)}
        self._offsets = [int(np.dot(d,self._strides)) for d in self._directions]
        self._norms = [sum(v*v for v in d) for d in self._directions]
        self._natural, self._forced_additions = [], []
        masks = []
        shifted = {d:_shift_occupancy(occupied,d) for d in _DIRECTIONS}
        for d in self._directions:
            natural, blockers, additions = _NEIGHBOURS[d]
            self._natural.append(tuple(direction_ids[v] for v in natural))
            self._forced_additions.append(tuple((int(np.dot(f,self._strides)),direction_ids[v])
                                                 for f,v in zip(blockers,additions)))
            count = 6 if sum(map(abs,d)) == 3 else 8
            mask = np.zeros(occupied.shape,dtype=bool)
            for f in blockers[:count]:
                mask |= shifted[f]
            masks.append(np.pad(mask,1,constant_values=False).ravel(order="F"))
        self._forced = masks
        self._coordinates = np.indices(self._dims).reshape(3,-1,order="F")
        self._prepared = True

    def plan(self, start, goal):
        start_i, goal_i = self.grid.float_to_int(np.array([start,goal]))
        if not self.grid.is_free(start_i) or not self.grid.is_free(goal_i):
            raise SearchFailure("JPS start or goal is occupied, unknown, or outside the grid")
        if not self._prepared:
            self._prepare()
        start = int(np.dot(start_i+1,self._strides))
        goal = int(np.dot(goal_i+1,self._strides))
        n = len(self._occupied)
        # Goal-dependent memoization avoids repeated lower-dimensional scans.
        # -2 means unevaluated; -1 means that this ray has no jump successor.
        jump_cache = np.full((26,n),-2,dtype=np.int32)
        occupied, blockers = self._occupied, self._blockers
        offsets, forced, natural = self._offsets, self._forced, self._natural

        def jump(position, direction):
            cache = jump_cache[direction]
            cached = int(cache[position])
            if cached != -2:
                return cached
            visited = []
            offset = offsets[direction]
            subdirections = natural[direction][:-1]
            forced_here = forced[direction]
            while True:
                cached = int(cache[position])
                if cached != -2:
                    result = cached
                    break
                visited.append(position)
                next_point = position + offset
                if occupied[next_point]:
                    result = -1
                    break
                if next_point == goal or forced_here[next_point]:
                    result = next_point
                    break
                found = False
                for subdirection in subdirections:
                    if jump(next_point,subdirection) >= 0:
                        found = True
                        break
                if found:
                    result = next_point
                    break
                position = next_point
            if len(visited) == 1:
                cache[visited[0]] = result
            else:
                cache[visited] = result
            return result

        coordinate = self._coordinates - (goal_i+1)[:,None]
        heuristics = self.heuristic_weight*np.sqrt(np.sum(coordinate*coordinate,axis=0))
        costs = np.full(n,np.inf)
        costs[start] = 0.
        parents = np.full(n,-1,dtype=np.int32)
        directions = np.full(n,26,dtype=np.int8)
        closed = np.zeros(n,dtype=bool)
        queue = [_QueueItem(float(heuristics[start]),0.,start,0)]
        serial = 1
        self.expanded = 0
        while queue:
            item = heapq.heappop(queue)
            current = item.node
            if item.g != costs[current] or closed[current]:
                continue
            closed[current] = True
            self.expanded += 1
            if current == goal:
                path = [goal]
                while path[-1] != start:
                    path.append(int(parents[path[-1]]))
                coordinates = np.column_stack(np.unravel_index(path[::-1],self._dims,order="F"))-1
                self.cost = item.g
                self.raw_path = self.grid.int_to_float(coordinates)
                self.path = simplify_path(self.grid,self.raw_path)
                return self.raw_path
            d = int(directions[current])
            candidates = list(natural[d])
            for offset,direction in self._forced_additions[d]:
                if blockers[current+offset]:
                    candidates.append(direction)
            for direction in candidates:
                child = jump(current,direction)
                if child < 0:
                    continue
                steps = abs((child-current)//offsets[direction])
                # Match source sqrt(dx*dx + dy*dy + dz*dz), not steps*sqrt(n).
                norm = self._norms[direction]
                candidate = item.g + math.sqrt(steps*steps*norm)
                if candidate < costs[child]:
                    parents[child],costs[child],directions[child] = current,candidate,direction
                    if not closed[child]:
                        heapq.heappush(queue,_QueueItem(candidate+float(heuristics[child]),candidate,child,serial))
                        serial += 1
        raise SearchFailure("JPS exhausted the open set")


def _remove_line_points(path):
    if len(path) < 3:
        return path.copy()
    keep = np.r_[True, np.sum(np.abs(path[2:] - 2*path[1:-1] + path[:-2]), axis=1) > .01, True]
    return path[keep]


def _remove_corners(grid, path, threshold):
    if len(path) < 3:
        return path.copy()
    result = [path[0]]
    previous = path[0]
    cost1 = math.inf if grid.is_blocked(path[0], path[1], threshold) else np.linalg.norm(path[0] - path[1])
    for index in range(1, len(path) - 1):
        first, second = path[index:index + 2]
        cost2 = math.inf if grid.is_blocked(first, second, threshold) else np.linalg.norm(first - second)
        cost3 = math.inf if grid.is_blocked(previous, second, threshold) else np.linalg.norm(previous - second)
        if cost3 < cost1 + cost2:
            cost1 = cost3
        else:
            result.append(first)
            cost1 = np.linalg.norm(first - second)
            previous = first
    return np.asarray(result + [path[-1]])


def simplify_path(grid, path, *, distance_map=False):
    path = _remove_line_points(path) if distance_map else path
    threshold = 1 if distance_map else 100
    path = _remove_corners(grid, path, threshold)
    path = _remove_corners(grid, path[::-1], threshold)[::-1]
    return path if distance_map else _remove_line_points(path)


class DistanceMapPlanner:
    def __init__(self, grid, potential_radius=(1.,1.,1.), search_radius=(2.,2.,2.),
                 heuristic_weight=0., cost_weight=.1, power=1, map_range=(0.,0.,0.)):
        self.grid = grid
        self.potential_radius = np.asarray(potential_radius, dtype=float)
        self.search_radius = np.asarray(search_radius, dtype=float)
        self.map_range = np.asarray(map_range, dtype=float)
        self.heuristic_weight, self.cost_weight, self.power = heuristic_weight, cost_weight, power
        self._potential_cache = None
        self.expanded = 0
        self.cost = math.inf

    def potential_map(self, position):
        global_map = not np.any(self.map_range)
        if global_map and self._potential_cache is not None:
            return self._potential_cache
        values = self.grid.values.copy()
        rn = math.ceil(self.potential_radius[0] / self.grid.resolution)
        hn = math.ceil(self.potential_radius[2] / self.grid.resolution)
        selected = values > 0
        if not global_map:
            low, high = self.grid.float_to_int(np.array([position-self.map_range,position+self.map_range]))
            low, high = np.clip(low,0,self.grid.shape-1),np.clip(high,0,self.grid.shape-1)
            region = np.zeros_like(selected)
            region[tuple(slice(a,b) for a,b in zip(low,high))] = True
            selected &= region
        values[selected] = 100
        if rn > 0 and hn > 0:
            shape = (2*rn+1,2*rn+1,2*hn+1)
            heights = np.zeros(shape,dtype=np.int16)
            footprint = np.zeros(shape,dtype=bool)
            for x in range(-rn,rn+1):
                for y in range(-rn,rn+1):
                    radius = math.hypot(x,y)
                    if radius > rn:
                        continue
                    for z in range(-hn,hn+1):
                        height = 100*((1-radius/rn)*(1-abs(z)/hn))**self.power
                        if height > 1e-3:
                            heights[x+rn,y+rn,z+hn] = int(height)
                            footprint[x+rn,y+rn,z+hn] = True
            # Max-plus dilation is the source's max of integer potential masks.
            source = np.where(selected,100,-1000).astype(np.int16)
            field = grey_dilation(source,footprint=footprint,structure=heights-100,
                                  mode="constant",cval=-1000)
            values = np.maximum(values,field).astype(np.int8)
        result = VoxelGrid(values,self.grid.origin,self.grid.resolution)
        if global_map:
            self._potential_cache = result
        return result

    def path_region(self, path, *, dense=False):
        radius = self.search_radius
        if radius[0] < 0 or radius[1] < 0:
            return np.ones_like(self.grid.values,dtype=bool)
        if dense:
            points = self.grid.float_to_int(path)
        else:
            points = []
            for first,second in zip(path[:-1],path[1:]):
                points.extend(self.grid.ray_trace(first,second))
                points.append(self.grid.float_to_int(second))
            points = np.asarray(points,dtype=np.int64).reshape(-1,3)
        region = np.zeros_like(self.grid.values,dtype=bool)
        points = points[((points>=0)&(points<self.grid.shape)).all(axis=1)]
        region[tuple(points.T)] = True
        rn = math.ceil(radius[0]/self.grid.resolution)
        hn = math.ceil(radius[2]/self.grid.resolution)
        xy = np.arange(-rn,rn+1)
        disk = (xy[:,None]**2+xy[None,:]**2)<=rn*rn
        region = binary_dilation(region,structure=disk[:,:,None])
        if hn:
            region = maximum_filter1d(region,size=2*hn+1,axis=2,mode="constant",cval=0)
        return region

    def plan(self, start, goal, prior_path):
        start, goal = np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
        self.distance_grid = self.potential_map(start)
        self.region = self.path_region(prior_path)
        first, last = self.grid.float_to_int(np.array([start, goal]))
        for point in (first, last):
            if np.any((point < 0) | (point >= self.grid.shape)) or self.distance_grid.values[tuple(point)] in (-1,100):
                raise SearchFailure("DMP start or goal is occupied, unknown, or outside the grid")
        # A one-cell blocked border removes per-neighbour coordinate/bound checks.
        costs_map = np.pad(self.distance_grid.values, 1, constant_values=100).ravel(order="F")
        allowed = np.pad(self.region, 1, constant_values=False).ravel(order="F")
        dims = tuple(self.grid.shape + 2)
        strides = (1, dims[0], dims[0]*dims[1])
        first_id = int(np.dot(first + 1, strides)); last_id = int(np.dot(last + 1, strides))
        edges = [(int(np.dot(d, strides)), math.sqrt(sum(v*v for v in d))) for d in _DIRECTIONS]
        n = len(costs_map)
        gs = np.full(n, np.inf)
        parents = np.full(n, -1, dtype=np.int64)
        closed = np.zeros(n, dtype=bool)
        gs[first_id] = float(costs_map[first_id])
        def heuristic(node):
            if self.heuristic_weight == 0:
                return 0.
            coordinate = np.array(np.unravel_index(node, dims, order="F")) - 1
            return self.heuristic_weight * np.linalg.norm(coordinate-last)
        queue = [_QueueItem(float(gs[first_id])+heuristic(first_id), float(gs[first_id]), first_id, 0)]
        serial = 1
        self.expanded = 0
        while queue:
            item = heapq.heappop(queue)
            current = item.node
            if item.g != gs[current] or closed[current]:
                continue
            closed[current] = True
            self.expanded += 1
            if current == last_id:
                path = [current]
                while path[-1] != first_id:
                    path.append(int(parents[path[-1]]))
                coordinates = np.column_stack(np.unravel_index(path[::-1], dims, order="F")) - 1
                self.raw_path = self.grid.int_to_float(coordinates)
                self.path = simplify_path(self.distance_grid, self.raw_path, distance_map=True)
                self.cost = item.g
                return self.raw_path
            for offset, length in edges:
                child = current + offset
                if costs_map[child] >= 100 or not allowed[child]:
                    continue
                candidate = item.g + length + self.cost_weight * float(costs_map[child])
                if candidate < gs[child]:
                    parents[child], gs[child] = current, candidate
                    if not closed[child]:
                        heapq.heappush(queue, _QueueItem(candidate+heuristic(child), candidate, child, serial))
                        serial += 1
        raise SearchFailure("DMP exhausted the open set")


@dataclass
class SearchResult:
    path: np.ndarray
    jps_path: np.ndarray
    dmp_path: np.ndarray
    cost: float
    expanded: dict[str, int]
    timings: dict[str, float]


def plan_path(grid: VoxelGrid, start, goal, *, heuristic_weight=None,
              potential_radius=(1.,1.,1.), search_radius=(2.,2.,2.)):
    """PlanCorridor's JPS→DMP stage; return the raw DMP path with exact endpoints."""
    start, goal = np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
    if np.linalg.norm(start-goal) <= 1e-3:
        return SearchResult(start[None], start[None], start[None], 0., {"jps":0,"dmp":0}, {"jps_s":0.,"dmp_s":0.})
    begin = time.perf_counter()
    jps = JPSPlanner(grid, grid.resolution if heuristic_weight is None else heuristic_weight)
    jps_path = jps.plan(start, goal)
    middle = time.perf_counter()
    dmp = DistanceMapPlanner(grid, potential_radius=potential_radius, search_radius=search_radius)
    dmp_path = dmp.plan(start, goal, jps_path)
    end = time.perf_counter()
    return SearchResult(np.vstack((start, dmp_path, goal)), jps_path, dmp_path, dmp.cost,
                        {"jps":jps.expanded,"dmp":dmp.expanded},
                        {"jps_s":middle-begin,"dmp_s":end-middle})
