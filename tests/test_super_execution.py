"""Numerical/state fixtures produced by unmodified SUPER source methods."""
from pathlib import Path
import json
from types import SimpleNamespace

import numpy as np
import pytest

from isaac_pathcover.control.execution import (
    BackupTraj, CmdTraj, CommandSample, ExecutionFSM, ExpTraj, MachineState,
    RetCode, TrajectoryExecution,
)
from isaac_pathcover.planning.pipeline import (
    PlanResult, Planner, PlannerConfig, _forward_sample_times, simple_pm_time_allocator,
)
from isaac_pathcover.trajectory.minco import PiecewisePolynomial


@pytest.fixture(scope="module")
def reference():
    return json.loads((Path(__file__).parent / "fixtures/super_execution.json").read_text())


def polynomial(durations, shift):
    coefficients = np.empty((len(durations), 8, 3))
    for i in range(len(durations)):
        for d in range(3):
            for k in range(8):
                coefficients[i, k, d] = (i + 1) * (d + 1) * .01 * (k + 1) + (shift if k == 0 else 0)
    return PiecewisePolynomial(durations, coefficients)


def trajectories():
    exp = ExpTraj(polynomial([2., 1.5], 1.), polynomial([2., 1.5], .1), 7.,
                  first_backup_start=.2, first_backup_end=.4)
    backup = BackupTraj(polynomial([.8], 5.), polynomial([.8], .7), 8.25, 1.25)
    return exp, backup


def test_cmd_prefix_backup_and_boundary_sampling_cpp(reference):
    exp, backup = trajectories()
    command = CmdTraj.from_trajectories(exp, backup)
    for row in reference["command"]:
        sample = command.sample(7.0 + row[0])
        assert sample.on_backup == bool(row[1])
        assert sample.finished == bool(row[2])
        np.testing.assert_allclose(sample.state, np.asarray(row[3:15]).reshape(3, 4), rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose([sample.yaw, sample.yaw_rate], row[15:17], rtol=2e-12, atol=2e-12)
    exp_only = CmdTraj.from_trajectories(exp)
    expected = reference["exp_only"][0]
    assert [exp_only.duration, exp_only.start_time, exp_only.backup_available,
            exp_only.is_on_backup(.3)] == expected


def test_shifted_polynomial_prefix_cpp(reference):
    exp, _ = trajectories()
    prefix = exp.position.cut(.35, 2.7)
    for row in reference["prefix"]:
        np.testing.assert_allclose(prefix.state(row[0]), np.asarray(row[1:]).reshape(3, 4),
                                   rtol=2e-12, atol=2e-12)


def test_source_guide_time_allocator_cpp(reference):
    for v0, total, distance, time, velocity in reference["allocator"]:
        np.testing.assert_allclose(simple_pm_time_allocator(15., 8., v0, total, distance),
                                   [time, velocity], rtol=2e-14, atol=2e-14)


def test_batched_sample_times_preserve_original_cpp_addition_order(reference):
    for start, stop, step, *times in reference["sampling"]:
        np.testing.assert_array_equal(_forward_sample_times(start, stop, step), times)


def test_source_fsm_cpp(reference):
    fsm = ExecutionFSM([0., 0., 0.], [10., 0., 0.])
    observations = []
    now, plan_calls, replan_calls = 0., 0, 0

    def main(result=RetCode.SUCCESS):
        nonlocal plan_calls
        action = fsm.main_tick(fsm.position, now)
        if action == "plan_from_rest":
            plan_calls += 1
            fsm.accept_result(PlanResult(code=result), now)

    def replan(result=RetCode.SUCCESS):
        nonlocal replan_calls
        if fsm.replan_tick() == "replan_once":
            replan_calls += 1
            fsm.accept_result(PlanResult(code=result), now)

    def command(position):
        sample = CommandSample(np.zeros((3, 4)), 0., 0., False, True)
        fsm.observe_command(sample, position, now)

    def emit(event):
        nonlocal now
        observations.append({"event": event, "state": int(fsm.state), "new_goal": bool(fsm.new_goal),
                             "finish_plan": fsm.finish_plan, "plan_from_rest": fsm.plan_from_rest,
                             "plan_calls": plan_calls, "replan_calls": replan_calls})
        now += .01

    main(); emit("init")
    main(); emit("goal")
    main(RetCode.FAILED); emit("rest_failed")
    main(); emit("rest_success")
    replan(); emit("skip_first_replan")
    replan(RetCode.FAILED); emit("replan_failed")
    replan(); emit("replan_success")
    replan(RetCode.NEW_TRAJ); emit("replan_new")
    main(RetCode.FINISH); emit("rest_finish")
    replan(); emit("finished_replan_skipped")
    command([0., 0., 0.]); emit("local_end")
    main(); emit("rest_after_local_end")
    replan(); emit("second_skip")
    replan(RetCode.EMER); emit("emergency")
    main(); emit("emergency_wait")
    fsm.new_goal = True
    main(); emit("next_goal")
    main(); emit("next_rest")
    # The C++ fixture directly assigns its MainFsm snapshot at this event.
    fsm.position = np.array([9.95, 0., 0.])
    command([9.95, 0., 0.]); emit("goal_end")
    fsm.new_goal = True
    fsm.position = np.zeros(3)
    main(); emit("race_goal")
    main(); emit("race_rest")
    replan(); emit("race_skip")
    assert fsm.replan_tick() == "replan_once"
    replan_calls += 1
    command([0., 0., 0.])
    fsm.accept_result(PlanResult(code=RetCode.SUCCESS), now)
    emit("replan_crosses_command_end")
    assert observations == reference["fsm"]


@pytest.mark.parametrize("index", [0, 1], ids=["outside_to_inside", "inside_to_outside"])
def test_pubcmd_during_replan_uses_main_position_snapshot_cpp(reference, index):
    case = reference["fsm_snapshots"][index]
    fsm = ExecutionFSM([0., 0., 0.], [10., 0., 0.])
    fsm.state = MachineState.FOLLOW_TRAJ
    fsm.new_goal = False
    observation = np.array(case["main_position"])
    fsm.main_tick(observation, now=0.)
    assert fsm.replan_tick() == "replan_once"
    observation[:] = case["pub_position"]
    sample = CommandSample(np.zeros((3, 4)), 0., 0., False, True)
    fsm.observe_command(sample, observation, now=.01)
    fsm.accept_result(PlanResult(code=RetCode.SUCCESS), now=.02)
    assert int(fsm.state) == case["state_after_pub"]
    np.testing.assert_array_equal(fsm.position, case["cached_position_after_pub"])
    fsm.main_tick(observation, now=.03)
    assert int(fsm.state) == case["state_after_main"]
    np.testing.assert_array_equal(fsm.position, case["cached_position_after_main"])


def test_handoff_samples_original_clock_and_retains_previous_prefix():
    exp, backup = trajectories()
    command = CmdTraj.from_trajectories(exp, backup)
    executor = TrajectoryExecution([0., 0., 0.])
    executor.install(command, now=7.5)
    assert executor.start_time == 7.
    np.testing.assert_allclose(executor.reference(7.5), exp.position.state(.5))
    assert not executor.sample(8.).on_backup
    assert executor.sample(8.3).on_backup


def test_replan_no_need_preserves_committed_backup(monkeypatch):
    planner = Planner.__new__(Planner)
    planner.scene = SimpleNamespace(goal=np.array([10., 0., 1.]))
    planner.config = PlannerConfig()
    planner.map = SimpleNamespace(version=1)
    planner.state = np.zeros((3, 4))
    planner.clock = lambda: 0.
    exp, backup = trajectories()
    planner.command = CmdTraj.from_trajectories(exp, backup)
    previous = planner.command
    planner._generate_exp = lambda _: (RetCode.NO_NEED, exp)
    planner._generate_backup = lambda _: (RetCode.NO_NEED, BackupTraj())
    result = planner.replan_once()
    assert result.success and result.command is None
    assert planner.command is previous
    assert planner.last_exp is exp


def test_replan_preserves_source_overtime_rejection(monkeypatch):
    import isaac_pathcover.planning.pipeline as pipeline
    clock = [0.]
    monkeypatch.setattr(pipeline, "perf_counter", lambda: clock[0])
    planner = Planner.__new__(Planner)
    planner.scene = SimpleNamespace(goal=np.array([10., 0., 1.]))
    planner.config = PlannerConfig()
    planner.map = SimpleNamespace(version=1)
    planner.state = np.zeros((3, 4))
    planner.clock = lambda: 0.
    exp, backup = trajectories()
    planner._generate_exp = lambda _: (RetCode.SUCCESS, exp)

    def generate_backup(_):
        clock[0] = .091
        return RetCode.SUCCESS, backup

    planner._generate_backup = generate_backup
    result = planner.replan_once()
    assert result.code == RetCode.FAILED
    assert result.status == "REPLAN_OVERTIME"
    assert result.command is None


def test_backup_captures_odometry_after_exp_optimization():
    planner = Planner.__new__(Planner)
    planner.config = PlannerConfig()
    planner.command = CmdTraj()
    planner.state = np.zeros((3, 4))
    measured = np.zeros((3, 4))
    measured[:, 0] = [1., 2., 3.]
    measured[:, 1] = [4., 0., 0.]
    planner.state_reader = lambda: (measured, np.array([1., 0., 0., 0.]), .4)
    planner.clock = lambda: 10.5
    exp, _ = trajectories()
    code, backup = planner._generate_backup(exp)
    assert code == RetCode.NO_NEED
    np.testing.assert_array_equal(backup.robot_position, measured[:, 0])
    np.testing.assert_array_equal(planner.state, measured)
    assert planner.yaw == .4


def test_backup_flag_changes_only_from_published_command_or_commit():
    planner = Planner.__new__(Planner)
    planner.config = PlannerConfig()
    planner.state_reader = None
    planner.backup_state_reader = lambda: True
    planner.clock = lambda: 12.
    planner.last_exp, backup = trajectories()
    planner.command = CmdTraj.from_trajectories(planner.last_exp, backup)
    planner.robot_on_backup = False
    planner.new_goal = False
    planner._result = PlanResult()
    measured = np.zeros((3, 4))
    planner.set_robot_state(measured)
    # A held command in GENERATE does not advance its source backup flag on odometry.
    assert planner.robot_on_backup is False
    code, _ = planner._generate_exp(measured[:, 0])
    assert code == RetCode.FAILED
    assert planner._result.status == "ON_BACKUP_WAIT_FOR_REST"
    assert planner.robot_on_backup is True


def test_close_goal_marks_output_copy_until_successful_commit():
    planner = Planner.__new__(Planner)
    planner.config = PlannerConfig()
    planner.state_reader = planner.backup_state_reader = None
    planner.clock = lambda: 7.
    planner.last_exp, backup = trajectories()
    planner.last_exp.corridor = SimpleNamespace(regions=[1, 2])
    planner.command = CmdTraj.from_trajectories(planner.last_exp, backup)
    planner.robot_on_backup = planner.new_goal = False
    planner.goal = planner.last_exp.position.evaluate(.1)
    planner._result = PlanResult()
    planner.set_robot_state(np.zeros((3, 4)))
    code, exp = planner._generate_exp(np.zeros(3))
    assert code == RetCode.NO_NEED and exp.connected_goal
    assert not planner.last_exp.connected_goal
