"""Acceptance metrics must not turn terminal failures into successful flights."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from isaac_pathcover.simulation.metrics import body_collisions, summarize


def make_scene():
    return SimpleNamespace(
        name="test", provenance="metric fixture", goal=np.array([1., 0., 1.]),
        lower=np.array([-5., -5., 0.]), upper=np.array([5., 5., 5.]),
        body_size=np.array([.2, .2, .2]),
        box_lower=np.array([[2., -.5, .5]]), box_upper=np.array([[3., .5, 1.5]]),
    )


def state(position=(1., 0., 1.), velocity=(0., 0., 0.)):
    return np.r_[position, [1., 0., 0., 0.], velocity, np.zeros(3)]


def report(states, *, arrived=False, substeps=None):
    states = np.asarray(states)
    references = np.zeros((len(states), 3, 4))
    references[:, :, 0] = make_scene().goal
    return summarize(
        make_scene(), .01 * np.arange(1, len(states) + 1), states, references,
        states if substeps is None else substeps, np.zeros(len(states)),
        np.zeros((len(states), 4)), [], .1, arrived=arrived,
    )


def test_runner_arrival_decision_rejects_fast_pass_and_short_dwell():
    fast_pass = report([state(velocity=(2., 0., 0.))], arrived=False)
    short_dwell = report([state()], arrived=False)
    settled = report([state()] * 50, arrived=True)
    assert fast_pass["goal_error_m"] == short_dwell["goal_error_m"] == 0.
    assert not fast_pass["arrived"] and not fast_pass["collision_free_arrival"]
    assert not short_dwell["arrived"]
    assert settled["arrived"] and settled["collision_free_arrival"]


def test_last_integrated_pose_is_checked_for_collision():
    before = np.array([state(position=(1., 0., 1.))])
    after = np.array([state(position=(2.5, 0., 1.))])
    assert not body_collisions(make_scene(), before).any()
    result = report(after, substeps=before)
    assert result["contact_force_max_N"] == 0.
    assert result["collision"]
    assert result["collision_physics_samples"] == 1
    assert result["collision_check_complete"]


def test_nonfinite_quaternion_preserves_trace_and_writes_failure_report(tmp_path):
    states = np.array([state(), state()])
    states[-1, 3:7] = np.nan
    trace = states.copy()
    np.savez_compressed(tmp_path / "flight.npz", states=states)
    result = report(states, arrived=True, substeps=states[:1])
    (tmp_path / "result.json").write_text(json.dumps(result, allow_nan=False))
    assert not result["finite"] and not result["arrived"]
    assert not result["collision_free_arrival"]
    assert not result["collision_check_complete"]
    assert result["collision_unchecked_physics_samples"] == 1
    assert result["invalid_state_samples"] == 1
    assert result["tracking_samples"] == 1
    assert result["max_tilt_rad"] == pytest.approx(0.)
    np.testing.assert_array_equal(states, trace)
    with np.load(tmp_path / "flight.npz") as saved:
        np.testing.assert_array_equal(saved["states"], trace)


@pytest.mark.parametrize("invalid_quaternion", [np.full(4, np.nan), np.zeros(4)])
def test_no_valid_orientation_reports_null_statistics(invalid_quaternion):
    states = np.array([state()])
    states[0, 3:7] = invalid_quaternion
    result = report(states, arrived=True)
    json.dumps(result, allow_nan=False)
    assert not result["arrived"] and not result["collision_check_complete"]
    assert result["valid_state_samples"] == 0
    assert result["max_tilt_rad"] is None
    assert result["tracking_rmse_m"] is None
    assert result["max_speed_m_s"] is None
    assert result["max_body_rates_rad_s"] == [None, None, None]
    assert result["height_range_m"] == [None, None]
