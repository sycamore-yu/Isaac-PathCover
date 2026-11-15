"""CPU compilation of the scenario LiDAR's existing voxel traversal."""
import numpy as np
from numba import njit


@njit(cache=True, fastmath=False)
def cast_voxels(occupied, indices, step, delta, tmax, max_range, max_steps):
    """Keep the NumPy DDA's axis ties, addition order and range termination.

    Ray origins and boundary distances are prepared by Scene.lidar with the
    original NumPy operations. Rays are independent, so each completes the
    same sequence of cells without allocating per-step gather/scatter arrays.
    """
    distance = np.zeros(len(indices))
    hits = np.zeros(len(indices), dtype=np.bool_)
    nx, ny, nz = occupied.shape
    for ray in range(len(indices)):
        x, y, z = indices[ray]
        tx, ty, tz = tmax[ray]
        dx, dy, dz = delta[ray]
        sx, sy, sz = step[ray]
        for _ in range(max_steps):
            if x < 0 or y < 0 or z < 0 or x >= nx or y >= ny or z >= nz:
                break
            if occupied[x, y, z]:
                hits[ray] = True
                break
            # np.argmin chooses the first axis when distances are equal.
            if tx <= ty and tx <= tz:
                distance[ray] = tx
                tx += dx
                x += sx
            elif ty <= tz:
                distance[ray] = ty
                ty += dy
                y += sy
            else:
                distance[ray] = tz
                tz += dz
                z += sz
            if distance[ray] >= max_range:
                break
    return hits, distance
