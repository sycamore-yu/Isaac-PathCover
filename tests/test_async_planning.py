"""Unbounded source optimizer and measured-progress handoff semantics."""

from types import SimpleNamespace

import numpy as np

from isaac_pathcover.control.execution import TrajectoryExecution
from isaac_pathcover.control.async_planner import handoff_horizon
from isaac_pathcover.trajectory.lbfgs import LBFGSParameters, minimize
from isaac_pathcover.trajectory.minco import PiecewisePolynomial


def test_source_optimizer_converges_without_external_wall_deadline():
    objective = lambda x: (float(x @ x), 2 * x)
    result = minimize(objective, [3., -4.], LBFGSParameters(max_iterations=10))
    assert result.success and result.evaluations > 0
    np.testing.assert_allclose(result.x, np.zeros(2), atol=1e-12)


def test_forward_shift_uses_measured_progress_without_budget_cutoff():
    assert handoff_horizon(.8, 1.2) == 1.2
    assert handoff_horizon(.8, 3.0) == 3.0
    assert handoff_horizon(.8, .25) == .8


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
