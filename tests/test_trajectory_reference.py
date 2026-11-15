"""Numerical checks against the unmodified upstream C++ header executable."""
import json
from pathlib import Path

import numpy as np
import pytest

from isaac_pathcover.trajectory import MINCO
from isaac_pathcover.trajectory.super_opt import inverse_t
from isaac_pathcover.trajectory.numerics import positive_time
from isaac_pathcover.trajectory.minco import derivative_basis

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "trajectory_reference.json").read_text())


def make_minco(order):
    a = np.zeros((3, order))
    b = a.copy()
    b[:, 0] = [3, 2, 1]
    return MINCO(a, b, 3, order).set_parameters(
        [[1, .5, .3], [2, 1.5, .7]], [1.1, 1.4, 1.2])


@pytest.mark.parametrize("order", [3, 4])
def test_minco_upstream_coefficients_energy_and_adjoint(order):
    m = make_minco(order)
    expected = FIXTURE["expected"][f"minco{order}"]
    energy, gc, gt = m.energy_with_partials()
    gp, time_grad = m.propagate_gradient(gc, gt)
    np.testing.assert_allclose(m.coefficients.reshape(-1, 3), expected["coefficients"], rtol=2e-11, atol=2e-12)
    np.testing.assert_allclose(energy, expected["energy"], rtol=2e-12)
    np.testing.assert_allclose(gc.reshape(-1, 3), expected["grad_c"], rtol=2e-11, atol=1e-9)
    np.testing.assert_allclose(gt, expected["partial_grad_t"], rtol=2e-11, atol=1e-9)
    np.testing.assert_allclose(gp, np.array(expected["grad_points"]).T, rtol=2e-11, atol=1e-9)
    np.testing.assert_allclose(time_grad, expected["grad_times"], rtol=2e-11, atol=2e-9)


@pytest.mark.parametrize("order", [3, 4])
def test_minco_boundary_and_full_optimal_continuity(order):
    m = make_minco(order)
    trajectory = m.trajectory()
    for d in range(order):
        np.testing.assert_allclose(trajectory.evaluate(0, d), m.start[:, d], atol=2e-10)
        np.testing.assert_allclose(trajectory.evaluate(trajectory.duration, d), m.end[:, d], atol=2e-10)
    for i in range(m.pieces - 1):
        for d in range(2 * order - 1):
            left = derivative_basis(m.durations[i], m.width - 1, d) @ m.coefficients[i]
            right = derivative_basis(0, m.width - 1, d) @ m.coefficients[i + 1]
            np.testing.assert_allclose(left, right, atol=2e-9)


@pytest.mark.parametrize("order", [3, 4])
def test_minco_adjoint_finite_differences(order):
    m = make_minco(order)
    x = np.r_[np.array([[1, .5, .3], [2, 1.5, .7]]).ravel(), m.durations]
    def objective(y):
        m.set_parameters(y[:6].reshape(2, 3), y[6:])
        energy, gc, gt = m.energy_with_partials()
        gp, gt = m.propagate_gradient(gc, gt)
        return energy, np.r_[gp.ravel(), gt]
    _, analytical = objective(x)
    numerical = finite_difference(objective, x, 1e-5)
    np.testing.assert_allclose(analytical, numerical, rtol=5e-7, atol=2e-5)


def test_positive_time_parameterization_roundtrip():
    times = np.geomspace(.001, 100, 51)
    actual = np.array([positive_time(t) for t in inverse_t(times)])
    np.testing.assert_allclose(actual, times, rtol=4e-15)


def finite_difference(objective, x, step):
    gradient = np.empty_like(x)
    for i in range(len(x)):
        xp, xm = x.copy(), x.copy()
        xp[i] += step
        xm[i] -= step
        gradient[i] = (objective(xp)[0] - objective(xm)[0]) / (2 * step)
    return gradient
