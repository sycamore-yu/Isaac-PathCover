"""Cooperative solve deadlines and forward-handoff semantics."""

from time import perf_counter
from types import SimpleNamespace

import numpy as np

from isaac_pathcover.control.execution import TrajectoryExecution
from isaac_pathcover.control.async_planner import handoff_horizon
from isaac_pathcover.planning.pipeline import Planner
from isaac_pathcover.trajectory.lbfgs import LBFGSParameters, minimize
from isaac_pathcover.trajectory.minco import PiecewisePolynomial


def test_expired_optimizer_budget_evaluates_no_objectives():
    calls = []

    def objective(x):
        calls.append(x.copy())
        return float(x @ x), 2 * x

    result = minimize(objective, [3., -4.], LBFGSParameters(max_iterations=20),
                      deadline=perf_counter() - 1.)
    assert result.message == "wall-clock budget exceeded"
    assert result.status == -1013 and result.evaluations == 0
    assert calls == []


def test_deadline_does_not_modify_unbudgeted_lbfgs_result():
    objective = lambda x: (float(x @ x), 2 * x)
    regular = minimize(objective, [3., -4.], LBFGSParameters(max_iterations=10))
    budgeted = minimize(objective, [3., -4.], LBFGSParameters(max_iterations=10),
                        deadline=perf_counter() + 10.)
    np.testing.assert_array_equal(regular.x, budgeted.x)
    assert regular.status == budgeted.status


def test_forward_shift_covers_budget_and_observed_clock_rate():
    assert handoff_horizon(1.5, .8, 1.) == 1.5
    assert handoff_horizon(1.5, .8, 2.) == 3.
    np.testing.assert_allclose(handoff_horizon(1.5, .8, .8), 1.2)
    assert handoff_horizon(.2, .8, .25) == .8


def test_planner_expired_deadline_does_not_refresh_map():
    class Scene:
        lower = np.array([-1., -1., 0.])
        upper = np.array([1., 1., 2.])
        goal = np.array([.5, .5, 1.])

        def point_cloud(self):
            return np.array([[0., 0., 0.]])

    planner = Planner(Scene())
    result = planner.plan(np.column_stack((np.array([0., 0., 1.]), np.zeros((3, 2)))),
                          deadline=perf_counter() - 1.)
    assert not result.success and result.trajectory is None
    assert "PlanningDeadline" in result.status
    assert planner.grid is None


def test_predicted_start_matches_old_active_reference_at_handoff():
    previous = TrajectoryExecution(np.zeros(3))
    polynomial = np.zeros((1, 6, 3))
    polynomial[0, 1, 0] = 1.
    old = PiecewisePolynomial([2.], polynomial)
    previous.replace(SimpleNamespace(success=True, trajectory=old), 0.,
                     np.zeros((3, 3)), validation={"accepted": True})
    target = .8
    predicted = previous.reference(target)[:, :3]
    np.testing.assert_allclose(predicted[:, 0], [.8, 0., 0.])
    before = previous.reference(.79)
    np.testing.assert_allclose(before[:, 0], [.79, 0., 0.])
    candidate_coeff = np.zeros((1, 6, 3))
    candidate_coeff[0, 0] = predicted[:, 0]
    candidate_coeff[0, 1] = predicted[:, 1]
    new = PiecewisePolynomial([1.], candidate_coeff)
    outcome = previous.replace(SimpleNamespace(success=True, trajectory=new), target,
                               predicted, validation={"accepted": True})
    assert outcome["accepted"]
    np.testing.assert_allclose(outcome["reference_jump"][:2], [0., 0.], atol=1e-12)
