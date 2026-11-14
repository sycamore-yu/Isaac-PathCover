"""PathCover's path traversal and RISP corridor construction.

Source: PathCover.hpp::pathCover, commit 8723a05d1993bf0199a453b47dc6ee167d493c23.
The source checks ``size > max_horizon`` after adding a region, therefore
returns up to max_horizon+1 regions. The terminal seed is the source's
published local goal; callers must not substitute the global goal when
the horizon ends early.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import segment_exit
from .risp import risp


@dataclass
class Corridor:
    polytopes: list[tuple[np.ndarray, np.ndarray]]
    seeds: np.ndarray
    remaining_counts: list[list[int]]
    source_start_index: int = 0

    @property
    def goal(self):
        return self.seeds[-1]


def find_start_polytope(position, polytopes):
    """Upstream trajectory_server's latest-containing/nearest-excess rule."""
    position = np.asarray(position, dtype=float)
    containing, nearest, nearest_excess = -1, 0, float("inf")
    for index, (A, b) in enumerate(polytopes):
        norms = np.linalg.norm(A, axis=1)
        valid = norms >= 1e-9
        distances = (A[valid] @ position - b[valid]) / norms[valid]
        if np.max(distances, initial=-np.inf) <= 0.0:
            containing = index
        excess = np.linalg.norm(np.maximum(distances, 0.0))
        if excess <= nearest_excess:
            nearest, nearest_excess = index, excess
    return containing if containing >= 0 else nearest


def path_cover(points, path, bounds, *, max_horizon=6, alpha=0.01, seed=0, rng=None):
    """Generate overlapping convex regions along a nonempty waypoint path.

    Array order in the input obstacle cloud is significant to randomized
    selection. Reuse ``rng`` for repeated calls to preserve its state.
    """
    points, path = np.asarray(points, dtype=float), np.asarray(path, dtype=float)
    if path.ndim != 2 or not len(path) or not np.all(np.isfinite(path)):
        raise ValueError("path must contain finite waypoints")
    if max_horizon < 1:
        raise ValueError("max_horizon must be positive")
    if rng is None:
        rng = np.random.default_rng(seed)
    result = risp(points, path[0], bounds, rng=rng, alpha=alpha)
    A, b = result.A, result.b
    polytopes = [(A, b)]
    seeds = [path[0].copy()]
    counts = [result.remaining_counts]
    intersection = path[0].copy()
    i = 1
    # With valid bounded workspace constraints the source's second while
    # condition (all planes violated at once) cannot hold. Explicit indexing
    # also reports invalid input instead of C++'s out-of-bounds access.
    while i < len(path):
        if np.all(A @ path[i] <= b):
            intersection = path[i].copy()
            i += 1
            continue
        intersection = segment_exit(A, b, intersection, path[i])
        result = risp(points, intersection, bounds, rng=rng, alpha=alpha)
        A, b = result.A, result.b
        polytopes.append((A, b))
        seeds.append(intersection.copy())
        counts.append(result.remaining_counts)
        if len(polytopes) > max_horizon:
            break
    # Retain the upstream .all() test and its local-goal publication rule.
    seeds.append(path[-1].copy() if np.all(A @ path[-1] > b) else intersection.copy())
    return Corridor(polytopes, np.asarray(seeds), counts)
