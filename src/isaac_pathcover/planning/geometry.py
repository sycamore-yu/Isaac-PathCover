"""Convex half-space geometry for PathCover and GCOPTER.

Half-spaces use ``A @ x <= b``. Dual hull construction follows PathCover
``PathCover.hpp::noredund`` and GCOPTER ``geo_utils.hpp::enumerateVs`` at
8723a05d. GCOPTER's vertex quantization is applied before translation.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull


class GeometryError(ValueError):
    """A seed or half-space intersection cannot define the requested region."""


def box_bounds(lower, upper):
    """Return the source ordering [-I; I] for an axis-aligned workspace."""
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if lower.shape != upper.shape or np.any(lower >= upper):
        raise ValueError("workspace lower bounds must be below upper bounds")
    identity = np.eye(lower.size)
    return np.concatenate((-identity, identity)), np.concatenate((-lower, upper))


def remove_redundant(A, b, seed):
    """Retain vertices of the dual hull, as in the upstream noredund."""
    A, b, seed = np.asarray(A, dtype=float), np.asarray(b, dtype=float), np.asarray(seed, dtype=float)
    slack = b - A @ seed
    if np.any(slack <= 1e-8):
        raise GeometryError("seed is not strictly inside the polytope")
    dual = A / slack[:, None]
    retained = ConvexHull(dual).vertices
    return A[retained].copy(), b[retained].copy()


def chebyshev_center(A, b):
    """Maximize the inscribed-ball radius; return (center, radius).

    The Euclidean normal lengths make this independent of plane scaling.
    Empty and lower-dimensional intersections have no positive radius.
    """
    A, b = np.asarray(A, dtype=float), np.asarray(b, dtype=float)
    dimension = A.shape[1]
    normals = np.linalg.norm(A, axis=1)
    result = linprog(
        np.r_[np.zeros(dimension), -1.0],
        A_ub=np.column_stack((A, normals)), b_ub=b,
        bounds=[(None, None)] * dimension + [(0.0, None)], method="highs",
    )
    if not result.success:
        raise GeometryError(f"polytope has no bounded inscribed ball: {result.message}")
    return result.x[:-1], float(result.x[-1])


def vertices(A, b, interior=None):
    """Enumerate primal vertices using facets of the dual convex hull.

    Match GCOPTER ``enumerateVs/filterVs``: reconstruct each primal vertex
    from the dual triangle cross product, quantize relative to the interior
    at 1e-6, and retain the first vertex in each quantization cell. This rule
    is different from the corridor visualizer's con2vert duplicate filter.
    """
    A, b = np.asarray(A, dtype=float), np.asarray(b, dtype=float)
    if interior is None:
        interior, radius = chebyshev_center(A, b)
        if radius <= 1e-10:
            raise GeometryError("polytope has no strict interior")
    interior = np.asarray(interior, dtype=float)
    slack = b - A @ interior
    if np.any(slack <= 1e-10):
        raise GeometryError("vertex enumeration requires a strict interior point")
    dual = A / slack[:, None]
    triangles = dual[ConvexHull(dual).simplices]
    point = triangles[:, 1]
    normal = np.cross(point - triangles[:, 0], triangles[:, 2] - point)
    raw = normal / np.einsum("ij,ij->i", normal, point)[:, None]
    magnitude = np.max(np.abs(raw))
    resolution = magnitude * max(1e-6 / magnitude, np.finfo(float).eps)
    scaled = raw / resolution
    # Eigen's round rounds ties away from zero; NumPy round uses ties to even.
    keys = np.copysign(np.floor(np.abs(scaled) + 0.5), scaled).astype(np.int64)
    seen, keep = set(), []
    for index, key in enumerate(map(tuple, keys)):
        if key not in seen:
            seen.add(key)
            keep.append(index)
    return raw[keep] + interior


def segment_exit(A, b, p1, p2):
    """First plane intersection on [p1,p2], matching computePolyhedraIntersection."""
    A, b = np.asarray(A, dtype=float), np.asarray(b, dtype=float)
    p1, p2 = np.asarray(p1, dtype=float), np.asarray(p2, dtype=float)
    direction = p2 - p1
    denominator = A @ direction
    nonparallel = np.abs(denominator) >= 1e-10
    values = (b[nonparallel] - A[nonparallel] @ p1) / denominator[nonparallel]
    values = values[(values >= 0.0) & (values <= 1.0)]
    if not values.size:
        raise GeometryError("no plane intersection lies on the path segment")
    return p1 + np.min(values) * direction
