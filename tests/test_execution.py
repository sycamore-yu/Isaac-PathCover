from types import SimpleNamespace
import numpy as np

from isaac_pathcover.control.execution import TrajectoryExecution, execution_check
from isaac_pathcover.trajectory.gcopter import TrajectoryConfig
from isaac_pathcover.trajectory.minco import PiecewisePolynomial


def test_failed_candidate_keeps_active_reference():
    execution = TrajectoryExecution(np.zeros(3))
    coefficients = np.zeros((1, 6, 3))
    coefficients[0, 1, 0] = 1.
    trajectory = PiecewisePolynomial(durations=np.array([2.]), coefficients=coefficients)
    result = SimpleNamespace(success=True, trajectory=trajectory)
    measured = np.column_stack((np.zeros(3), [1., 0., 0.], np.zeros(3)))
    execution.replace(result, 0., measured, validation={"accepted": True})
    before = execution.reference(.5)
    outcome = execution.replace(SimpleNamespace(success=False), .5, measured, validation=None)
    assert outcome == {"accepted": False, "retained_previous": True}
    np.testing.assert_array_equal(execution.reference(.5), before)
    assert execution.start_time == 0.


def test_first_failure_holds_initial_position():
    execution = TrajectoryExecution(np.array([1., 2., 3.]))
    outcome = execution.replace(SimpleNamespace(success=False), 1., np.zeros((3, 3)), validation=None)
    assert not outcome["accepted"] and not outcome["retained_previous"]
    np.testing.assert_array_equal(execution.reference(2.)[:, 0], [1., 2., 3.])
    np.testing.assert_array_equal(execution.reference(2.)[:, 1:], 0.)


def test_observed_converged_but_infeasible_plan_is_not_executed():
    # Actual medium t=3.5 s candidate: upstream C++ also converged to a bad
    # trajectory for these inputs. Keep optimizer success distinct from handoff.
    diagnostics = {"max_speed": 5.670156106934268, "max_body_rate": 13.566524906451011,
                   "max_tilt": 2.866186856156229, "min_thrust": .2147150626519353,
                   "max_thrust": 13.323697984030261, "max_corridor_residual_m": -.012903883257923576,
                   "flatness_samples_per_piece": 101}
    validation = execution_check(diagnostics, TrajectoryConfig())
    assert set(validation["violations"]) == {"speed_squared", "body_rate_squared", "tilt", "thrust_squared"}
    execution = TrajectoryExecution(np.array([1., 2., 3.]))
    result = SimpleNamespace(success=True, trajectory=None)
    outcome = execution.replace(result, 3.5, np.zeros((3, 3)), validation=validation)
    assert result.success and not outcome["accepted"]
    np.testing.assert_array_equal(execution.reference(4.)[:, 0], [1., 2., 3.])


def test_execution_check_accepts_observed_nominal_plan_and_rejects_nonfinite():
    diagnostics = {"max_speed": 3.0007596420856, "max_body_rate": 1.215096151307942,
                   "max_tilt": .30091666139828677, "min_thrust": 7.152839515859484,
                   "max_thrust": 7.963606350711862, "max_corridor_residual_m": -.12049398640999609,
                   "flatness_samples_per_piece": 101}
    assert execution_check(diagnostics, TrajectoryConfig())["accepted"]
    diagnostics["max_tilt"] = np.nan
    check = execution_check(diagnostics, TrajectoryConfig())
    assert not check["accepted"] and check["residuals"]["tilt"] is None
    diagnostics["max_tilt"] = .3
    diagnostics["max_thrust"] = np.nan
    assert not execution_check(diagnostics, TrajectoryConfig())["accepted"]
