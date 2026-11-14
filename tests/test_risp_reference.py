"""Compare the Python RISP port with unmodified upstream C++ output."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from isaac_pathcover.planning.geometry import box_bounds, chebyshev_center, vertices
from isaac_pathcover.planning.risp import risp


REFERENCE = json.loads((Path(__file__).parent / "fixtures/risp_corridor_reference.json").read_text())


class ReferenceRandom:
    """Use the recorded C rand draws; modulo reproduces rand()%size exactly."""

    def __init__(self, values):
        self.values = iter(values)

    def integers(self, high):
        return next(self.values) % high


def normalized_planes(A, b):
    A, b = np.asarray(A), np.asarray(b)
    return np.column_stack((A, b)) / np.linalg.norm(A, axis=1)[:, None]


def plane_error(A, b, expected):
    actual = normalized_planes(A, b)
    reference = normalized_planes(expected["A"], expected["b"])
    assert actual.shape == reference.shape
    distances = np.max(np.abs(actual[:, None] - reference[None, :]), axis=2)
    rows, columns = linear_sum_assignment(distances)
    return float(distances[rows, columns].max())


def test_risp_matches_official_planes_and_discard_sequence():
    case = REFERENCE
    bounds = np.asarray(case["bounds"]["A"]), np.asarray(case["bounds"]["b"])
    expected = case["expected"]
    result = risp(case["points"], case["path"][0], bounds, rng=ReferenceRandom(expected["rand_values"]))
    assert result.remaining_counts == expected["risp_counts"]
    assert plane_error(result.A, result.b, expected["risp"]) < 2e-13
    assert np.all(result.A @ case["path"][0] < result.b)
    violations = np.asarray(case["points"]) @ result.A.T - result.b
    assert np.all(np.max(violations, axis=1) > 0.0)


def test_dual_hull_vertices_and_chebyshev_ball():
    A, b = box_bounds([-2.0, -3.0, -4.0], [2.0, 3.0, 4.0])
    center, radius = chebyshev_center(A, b)
    assert radius == pytest.approx(2.0)
    verts = vertices(A, b, center)
    assert verts.shape == (8, 3)
    np.testing.assert_allclose(np.abs(verts), np.tile([2.0, 3.0, 4.0], (8, 1)))
    assert np.all(A @ center + np.linalg.norm(A, axis=1) * radius <= b + 1e-12)


def test_risp_empty_cloud_and_invalid_seed():
    bounds = box_bounds([-1.0] * 3, [1.0] * 3)
    result = risp(np.empty((0, 3)), np.zeros(3), bounds)
    assert result.remaining_counts == [0]
    assert len(result.A) == 6
    with pytest.raises(ValueError, match="coincides"):
        risp(np.zeros((1, 3)), np.zeros(3), bounds)
    with pytest.raises(ValueError, match="workspace"):
        risp(np.empty((0, 3)), [1.0, 0.0, 0.0], bounds)
