"""Numerical checks against the unmodified upstream C++ header executable."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from isaac_pathcover.planning.geometry import box_bounds
from isaac_pathcover.trajectory import FlatnessMap, GCopter, MINCO, TrajectoryConfig
from isaac_pathcover.trajectory.gcopter import forward_t, inverse_t
from isaac_pathcover.trajectory.minco import derivative_basis

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "trajectory_reference.json").read_text())


def reference_solver(config=None):
    start = np.zeros((3, 3))
    end = start.copy()
    start[:, 0], end[:, 0] = [-2, 0, 1], [2, 0, 1]
    return GCopter([box_bounds([-3, -2, 0], [1, 2, 3]),
                    box_bounds([-1, -2, 0], [3, 2, 3])], start, end, config)


def shared_coordinate_solver():
    solver = reference_solver()
    # Hull libraries may order equal vertices differently. The C++ harness and
    # Python use these identical vertex coordinates to define the same x.
    solver.vpolys = [np.array(v) for v in FIXTURE["gcopter_vertex_coordinates"]]
    solver.point_polys = [solver.vpolys[i] for i in solver.vpoly_indices]
    solver.initial_x = np.array(FIXTURE["gcopter_initial_x"])
    return solver


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


def test_flatness_nonzero_drag_upstream():
    model = FlatnessMap((.752, 9.8066, .1, .15, .08, .001))
    thrust, quat, omega = model.forward(np.array([.8, -.3, .2]), np.array([.5, .2, .8]),
                                        np.array([.2, .7, -.4]), .3, -.2)
    gradients = model.backward(np.array([.2, -.1, .4]), np.array([.1, .3, -.2]), .7,
                               np.array([.3, -.6, .2, .1]), np.array([.4, -.2, .5]))
    expected = FIXTURE["expected"]["flatness"]
    for name, value in zip(["thrust", "quat", "omega", "grad_pos", "grad_vel", "grad_acc",
                            "grad_jer", "grad_yaw", "grad_yaw_rate"],
                           [thrust, quat, omega, *gradients]):
        np.testing.assert_allclose(value, expected[name], rtol=2e-14, atol=2e-15)


def test_flatness_batch_reverse_matches_scalar():
    rng = np.random.default_rng(91)
    vel, acc, jer = rng.normal(size=(3, 2, 4, 3))
    pg, vg, og = rng.normal(size=(3, 2, 4, 3))
    qg, tg = rng.normal(size=(2, 4, 4)), rng.normal(size=(2, 4))
    parameters = (.752, 9.8066, .1, .15, .08, .001)
    batch = FlatnessMap(parameters)
    output = batch.forward(vel, acc, jer, .2, .1)
    gradient = batch.backward(pg, vg, tg, qg, og)
    for i in range(2):
        for j in range(4):
            scalar = FlatnessMap(parameters)
            one = scalar.forward(vel[i, j], acc[i, j], jer[i, j], .2, .1)
            grad = scalar.backward(pg[i, j], vg[i, j], tg[i, j], qg[i, j], og[i, j])
            for a, b in zip([*output, *gradient], [*one, *grad]):
                np.testing.assert_allclose(a[i, j], b, atol=2e-14, rtol=2e-14)


def test_positive_time_parameterization_roundtrip():
    times = np.geomspace(.001, 100, 51)
    np.testing.assert_allclose(forward_t(inverse_t(times)), times, rtol=4e-15)


def test_gcopter_objective_and_gradient_upstream():
    solver = shared_coordinate_solver()
    cost, gradient = solver.cost_gradient(solver.initial_x)
    expected = FIXTURE["expected"]["gcopter"]
    np.testing.assert_allclose(cost, expected["initial_cost"], rtol=3e-12)
    np.testing.assert_allclose(gradient, expected["initial_gradient"], rtol=4e-11, atol=1e-5)


def finite_difference(objective, x, step):
    gradient = np.empty_like(x)
    for i in range(len(x)):
        xp, xm = x.copy(), x.copy()
        xp[i] += step
        xm[i] -= step
        gradient[i] = (objective(xp)[0] - objective(xm)[0]) / (2 * step)
    return gradient


@pytest.mark.parametrize("order", [3, 4])
def test_gcopter_analytic_gradient_finite_differences(order):
    solver = reference_solver(TrajectoryConfig(cost_order=order))
    x = solver.initial_x.copy()
    x[:2] = [.2, -.1]
    x[2:] += np.linspace(-.05, .05, len(x) - 2)
    _, analytical = solver.cost_gradient(x)
    numerical = finite_difference(solver.cost_gradient, x, 2e-6)
    np.testing.assert_allclose(analytical, numerical, rtol=2e-6, atol=2e-3)


def test_gcopter_original_lbfgs_convergence():
    solver = shared_coordinate_solver()
    result = solver.optimize()
    expected = FIXTURE["expected"]["gcopter"]
    assert result.success
    assert result.status_code == expected["status"]
    assert result.iterations == expected["iterations"]
    np.testing.assert_allclose(result.cost, expected["cost"], rtol=2e-8, atol=2e-5)
    np.testing.assert_allclose(solver.last_result.x, expected["x"], atol=2e-5)
    trajectory = result.trajectory
    np.testing.assert_allclose(trajectory.evaluate(0), [-2, 0, 1], atol=1e-12)
    np.testing.assert_allclose(trajectory.evaluate(trajectory.duration), [2, 0, 1], atol=1e-12)


def test_gcopter_iteration_limit_discards_candidate():
    solver = reference_solver(replace(TrajectoryConfig(), max_iterations=1))
    result = solver.optimize()
    assert not result.success
    assert result.status_code == -1008
    assert result.trajectory is None
    assert np.isinf(result.cost)


def test_polynomial_constraint_extrema_between_samples():
    from isaac_pathcover.trajectory import PiecewisePolynomial
    from isaac_pathcover.trajectory.diagnostics import trajectory_diagnostics
    coefficients = np.zeros((1, 6, 3))
    coefficients[0, 0, 2] = 1.0
    coefficients[0, 2, 0], coefficients[0, 3, 0] = .75, -.25
    trajectory = PiecewisePolynomial(np.array([2.0]), coefficients)
    corridor = [box_bounds([-1, -1, 0], [.9, 1, 2])]
    result = trajectory_diagnostics(trajectory, corridor, TrajectoryConfig(), samples_per_piece=2)
    # Speed peaks at t=1, between the only two flatness sample times.
    np.testing.assert_allclose(result["max_speed"], .75, atol=1e-14)
    np.testing.assert_allclose(result["max_acceleration"], 1.5, atol=1e-14)
    np.testing.assert_allclose(result["max_corridor_residual_m"], .1, atol=1e-14)
