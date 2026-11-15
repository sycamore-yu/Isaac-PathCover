"""Speed timelines distinguish startup, planned braking, and tracking lag."""
import numpy as np

from isaac_pathcover.simulation.metrics import summarize_timeline


def test_intermediate_pause_keeps_reference_and_planning_evidence():
    times = np.arange(1, 201) * .01
    states = np.zeros((200, 13))
    states[:, 0] = 2.
    references = np.zeros((200, 3, 4))
    states[30:, 7] = references[30:, 0, 1] = 1.
    states[75:125, 7] = references[75:125, 0, 1] = .05
    arrays = {"time": times, "states": states, "references": references,
              "on_backup": (times >= .76) & (times <= 1.25), "fsm_state": np.full(200, 4)}
    plans = [
        {"id": 1, "t": .01, "result_t": .2, "activation_t": .2,
         "operation": "plan_from_rest",
         "status": "SUCCESS_WITH_BACKUP", "backup_status": "SUCCESS", "worker_s": .19,
         "timings": {"total_s": .19},
         "handoff": {"accepted": True, "backup_available": True, "reference_jump": [0.] * 4}},
        {"id": 2, "t": .6, "result_t": 1.4, "activation_t": None,
         "operation": "replan_once",
         "status": "REPLAN_OVERTIME", "backup_status": "SUCCESS", "worker_s": .8,
         "timings": {"total_s": .8}, "handoff": {"accepted": False}},
    ]
    result = summarize_timeline(arrays, plans, np.array([10., 0., 0.]))
    assert result["first_motion_s"] == .31
    assert len(result["intermediate_slow_intervals"]) == 1
    interval = result["intermediate_slow_intervals"][0]
    np.testing.assert_allclose([interval["start_s"], interval["end_s"]], [.75, 1.25])
    assert interval["last_committed_plan"] == 1
    assert interval["concurrent_plans"][0]["status"] == "REPLAN_OVERTIME"
    assert interval["on_backup_samples"] == 50
    assert result["backup_entries"] == 1
    assert result["committed_with_backup"] == 1


def test_no_flight_is_reported_as_startup_without_intermediate_pause():
    arrays = {"time": np.arange(1, 101) * .01, "states": np.zeros((100, 13)),
              "references": np.zeros((100, 3, 4)), "on_backup": np.zeros(100, bool),
              "fsm_state": np.full(100, 3)}
    result = summarize_timeline(arrays, [], np.array([10., 0., 0.]))
    assert result["first_motion_s"] is None
    assert result["intermediate_slow_intervals"] == []
    assert result["backup_entries"] == 0
