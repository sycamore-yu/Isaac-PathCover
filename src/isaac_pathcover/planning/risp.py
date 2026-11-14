"""Native NumPy port of PathCover.hpp RISP (upstream 8723a05d).

For a sampled obstacle p and seed y the separator is
``a=p-y; b=a@y+(1-alpha)*||a||**2``. Points with ``a@p>b`` are
removed, preserving their original order. No ellipsoid optimization or
post-hoc polytope shrink is introduced.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import GeometryError, remove_redundant


@dataclass
class RISPResult:
    A: np.ndarray
    b: np.ndarray
    remaining_counts: list[int]


def risp(points, seed, bounds, *, rng=None, alpha=0.01):
    """Partition obstacle points and append workspace bounds.

    ``rng`` follows NumPy's ``Generator.integers(high)`` interface. Passing
    one generator through a corridor retains a single random sequence,
    just as upstream uses the process-wide C random generator.
    """
    points = np.asarray(points, dtype=float)
    seed = np.asarray(seed, dtype=float)
    A_bound, b_bound = (np.asarray(value, dtype=float) for value in bounds)
    if not 0.0 < alpha < 1.0:
        raise ValueError("RISP alpha must lie in (0,1)")
    if points.shape != (points.shape[0], seed.size) or not np.all(np.isfinite(points)):
        raise ValueError("obstacle points must be finite with shape (N, dimension)")
    if np.any(b_bound - A_bound @ seed <= 1e-8):
        raise GeometryError("RISP seed must be strictly inside workspace bounds")
    if len(points) and np.any(np.all(points == seed, axis=1)):
        raise GeometryError("RISP seed coincides with an obstacle point")
    if rng is None:
        rng = np.random.default_rng(0)
    work = points.copy()
    normals, offsets = [], []
    remaining = [len(work)]
    while len(work):
        normal = work[int(rng.integers(len(work)))] - seed
        offset = float(normal @ seed + (1.0 - alpha) * (normal @ normal))
        normals.append(normal)
        offsets.append(offset)
        # A single-threaded GEMV and stable compaction preserve the selected
        # point sequence while avoiding boolean-index copying of each row.
        distances = work @ normal
        work = np.compress(distances <= offset, work, axis=0)
        remaining.append(len(work))
    A = np.concatenate((np.asarray(normals).reshape(-1, seed.size), A_bound))
    b = np.concatenate((np.asarray(offsets), b_bound))
    A, b = remove_redundant(A, b, seed)
    return RISPResult(A, b, remaining)
