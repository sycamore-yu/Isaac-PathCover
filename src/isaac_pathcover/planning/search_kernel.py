"""CPU compilation of SUPER's A* loop; algorithm and queue order are unchanged."""
from time import perf_counter

import numpy as np
from numba import njit, objmode, types
from numba.extending import intrinsic


# One search runs at a time in SUPER's planner callback. The object-mode call
# preserves the supplied simulation clock at every source timeout checkpoint.
active_clock = perf_counter


def read_clock():
    return float(active_clock())


@intrinsic
def read_shared_clock(typing_context, array_type):
    """An atomic load cannot be hoisted out of the source expansion loop."""
    def codegen(context, builder, signature, args):
        array = context.make_array(signature.args[0])(context, builder, args[0])
        return builder.load_atomic(array.data, "monotonic", 8)
    return types.float64(array_type), codegen


@njit(cache=True, inline="always")
def heuristic(index, goal, kind):
    dx, dy, dz = abs(index[0]-goal[0]), abs(index[1]-goal[1]), abs(index[2]-goal[2])
    if kind == 1:
        value = dx + dy + dz
    elif kind == 2:
        value = int(np.sqrt(dx*dx+dy*dy+dz*dz))
    else:
        a, b, c = sorted((dx, dy, dz))
        value = np.sqrt(3.)*a + np.sqrt(2.)*(b-a) + c-b
    return 1.00001 * value


@njit(cache=True, inline="always")
def push(heap, node, costs):
    heap.append(node)
    hole = len(heap)-1
    while hole > 0:
        parent = (hole-1)//2
        if costs[heap[parent]] <= costs[node]:
            break
        heap[hole] = heap[parent]
        hole = parent
    heap[hole] = node


@njit(cache=True, inline="always")
def pop(heap, costs):
    result = heap[0]
    value = heap.pop()
    length = len(heap)
    if length == 0:
        return result
    hole, second = 0, 0
    while second < (length-1)//2:
        second = 2*(second+1)
        if costs[heap[second]] > costs[heap[second-1]]:
            second -= 1
        heap[hole] = heap[second]
        hole = second
    if length % 2 == 0 and second == (length-2)//2:
        second = 2*(second+1)
        heap[hole] = heap[second-1]
        hole = second-1
    while hole > 0:
        parent = (hole-1)//2
        if costs[heap[parent]] <= costs[value]:
            break
        heap[hole] = heap[parent]
        hole = parent
    heap[hole] = value
    return result


@njit(cache=True, inline="always")
def grid_type(index, resolution, inflated, values, occupied, unknown, lower, upper,
              inf_lower, parameters, heights, unknown_inflation):
    # parameters: probability resolution, inflation resolution, free/occ odds,
    # virtual ground/ceiling, inflated ground/ceiling, inflated query margin.
    x = (index[0]+0.5)*resolution
    y = (index[1]+0.5)*resolution
    z = (index[2]+0.5)*resolution
    inverse = 1.0 / parameters[0]
    px, py, pz = int(np.floor(x*inverse)), int(np.floor(y*inverse)), int(np.floor(z*inverse))
    inside = lower[0] <= px <= upper[0] and lower[1] <= py <= upper[1] and lower[2] <= pz <= upper[2]
    if inflated:
        if not inside:
            return 2  # OUT_OF_MAP
        if z >= parameters[7]-parameters[8] or z <= parameters[6]+parameters[8]:
            return 3
        inverse = 1.0 / parameters[1]
        ix = int(np.floor(x*inverse))-inf_lower[0]
        iy = int(np.floor(y*inverse))-inf_lower[1]
        iz = int(np.floor(z*inverse))-inf_lower[2]
        if occupied[ix, iy, iz] > 0:
            return 3
        if unknown_inflation and unknown[ix, iy, iz] > 0:
            return 1
        return 4
    if z <= parameters[4] or z >= parameters[5] or pz <= heights[0] or pz >= heights[1]:
        return 3
    if not inside:
        return 2
    value = values[px-lower[0], py-lower[1], pz-lower[2]]
    return 3 if value >= parameters[3] else 4 if value < parameters[2] else 1


@njit(cache=True)
def run(start, goal, center, half_size, offsets, distances, fine_neighbors,
        resolution, inflated, use_inf_neighbor, unknown_as_occupied, escape,
        heuristic_kind, horizon, started, timeout, values, occupied, unknown,
        lower, upper, inf_lower, parameters, heights, unknown_inflation, lookup, clock_buffer):
    indices = [start]
    g = [0.0]
    f = [0.0 if escape else heuristic(start, goal, heuristic_kind)]
    parents = [-1]
    closed = [False]
    stride_y, stride_z = 2*half_size[1]+1, 2*half_size[2]+1
    start_hash = ((start[0]-center[0]+half_size[0])*stride_y
                  + start[1]-center[1]+half_size[1])*stride_z + start[2]-center[2]+half_size[2]
    lookup[start_hash] = 0
    used = [start_hash]
    heap = [0]
    expanded = 0
    while len(heap):
        current = pop(heap, f)
        index = indices[current]
        expanded += 1
        status = 0
        if escape:
            kind = grid_type(index, resolution, True, values, occupied, unknown,
                             lower, upper, inf_lower, parameters, heights, unknown_inflation)
            if kind != 3 and (not unknown_as_occupied or kind != 1):
                status = 2
        elif index == goal:
            status = 1
        elif horizon > 0.0 and g[current] > horizon / resolution:
            status = 2
        if status:
            length, parent = 0, current
            while parent >= 0:
                length += 1
                parent = parents[parent]
            path = np.empty((length, 3), dtype=np.int64)
            parent = current
            for i in range(length-1, -1, -1):
                path[i, 0], path[i, 1], path[i, 2] = indices[parent]
                parent = parents[parent]
            for address in used:
                lookup[address] = -1
            return path, status, expanded, g[current]*resolution
        closed[current] = True
        current_cost = g[current]
        for i in range(len(offsets)):
            neighbor = (index[0]+offsets[i, 0], index[1]+offsets[i, 1], index[2]+offsets[i, 2])
            if (abs(neighbor[0]-center[0]) > half_size[0]
                or abs(neighbor[1]-center[1]) > half_size[1]
                or abs(neighbor[2]-center[2]) > half_size[2]):
                continue
            address = ((neighbor[0]-center[0]+half_size[0])*stride_y
                       + neighbor[1]-center[1]+half_size[1])*stride_z + neighbor[2]-center[2]+half_size[2]
            node = lookup[address]
            if node == -2 or (node >= 0 and closed[node]):
                continue
            # The map and query flags are constant during one source search.
            # Existing nodes are already traversable; -2 memoizes a blocked cell.
            if node == -1:
                if use_inf_neighbor and not inflated:
                    kind, has_free = 0, False
                    for j in range(len(fine_neighbors)):
                        query = (neighbor[0]+fine_neighbors[j, 0], neighbor[1]+fine_neighbors[j, 1],
                                 neighbor[2]+fine_neighbors[j, 2])
                        if (abs(query[0]-center[0]) > half_size[0]
                            or abs(query[1]-center[1]) > half_size[1]
                            or abs(query[2]-center[2]) > half_size[2]):
                            continue
                        query_type = grid_type(query, resolution, False, values, occupied, unknown,
                                               lower, upper, inf_lower, parameters, heights, unknown_inflation)
                        if query_type == 3:
                            kind = 3
                            break
                        if query_type == 4:
                            has_free = True
                    if unknown_as_occupied and kind != 3:
                        kind = 4 if has_free else 1
                else:
                    kind = grid_type(neighbor, resolution, inflated, values, occupied, unknown,
                                     lower, upper, inf_lower, parameters, heights, unknown_inflation)
                if kind == 3 or kind == 2 or (unknown_as_occupied and kind == 1):
                    lookup[address] = -2
                    used.append(address)
                    continue
            cost = current_cost + distances[i]
            if node < 0:
                node = len(indices)
                lookup[address] = node
                used.append(address)
                indices.append(neighbor)
                g.append(cost)
                f.append(cost if escape else cost + heuristic(neighbor, goal, heuristic_kind))
                parents.append(current)
                closed.append(False)
                push(heap, node, f)
            elif cost < g[node]:
                parents[node], g[node] = current, cost
                f[node] = cost if escape else cost + heuristic(neighbor, goal, heuristic_kind)
        if timeout >= 0.0:
            if len(clock_buffer):
                now = read_shared_clock(clock_buffer)
            else:
                with objmode(now="float64"):
                    now = read_clock()
            if now - started > timeout:
                for address in used:
                    lookup[address] = -1
                return np.empty((0, 3), dtype=np.int64), 3, expanded, 0.0
    for address in used:
        lookup[address] = -1
    return np.empty((0, 3), dtype=np.int64), 0, expanded, 0.0
