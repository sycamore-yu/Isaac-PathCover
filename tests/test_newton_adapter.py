"""SUPER callback publication and wall-clock phase in the Newton adapter."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from isaac_pathcover.control.async_planner import AsyncPlanningWorker
from isaac_pathcover.control.execution import CmdTraj, ExecutionFSM, MachineState, TrajectoryExecution
from isaac_pathcover.demo import FlightTrace, next_wall_deadline
from isaac_pathcover.trajectory.minco import PiecewisePolynomial


def command(offset):
    coefficients = np.zeros((1, 8, 3))
    coefficients[0, 0, 0] = offset
    coefficients[0, 1, 0] = 1.
    position = PiecewisePolynomial([1.], coefficients)
    yaw = PiecewisePolynomial([1.], np.zeros((1, 8, 3)))
    return CmdTraj(position, yaw)


def test_async_result_in_generate_does_not_publish_until_follow():
    execution = TrajectoryExecution(np.zeros(3))
    flight = FlightTrace(SimpleNamespace(step_dt=.01), None, execution)
    fsm = ExecutionFSM(np.zeros(3), [20., 0., 0.])
    state = np.r_[np.zeros(3), [1., 0., 0., 0.], np.zeros(6)]
    execution.install(command(0.))
    fsm.state = MachineState.FOLLOW_TRAJ
    assert flight.publish_command(.3, state, fsm) is False
    np.testing.assert_allclose(flight.command.state[:, 0], [.3, 0., 0.])

    # PubCmd can finish while Replan is computing. Its later result is installed
    # in GENERATE, but the source callback keeps the last published command.
    fsm.state = MachineState.GENERATE_TRAJ
    execution.install(command(10.))
    assert flight.publish_command(.4, state, fsm) is None
    np.testing.assert_allclose(flight.command.state[:, 0], [.3, 0., 0.])
    assert flight.command_time == .3

    fsm.state = MachineState.FOLLOW_TRAJ
    assert flight.publish_command(.5, state, fsm) is False
    np.testing.assert_allclose(flight.command.state[:, 0], [10.5, 0., 0.])
    assert flight.command_time == .5
    fsm.stop = True
    assert flight.publish_command(.6, state, fsm) is None
    assert flight.command_time == .5


def test_delayed_periodic_callback_retains_original_phase():
    # A 66 ms timer delayed until 200 ms next expires at 264 ms, not 266 ms.
    assert next_wall_deadline(66_000_000, 200_000_000, 66_000_000) == 264_000_000


def test_worker_error_survives_cleanup_after_child_closes_pipe():
    worker = AsyncPlanningWorker.__new__(AsyncPlanningWorker)
    worker.ready, worker.busy = True, False
    worker._connection = Mock()
    worker._connection.poll.return_value = True
    worker._connection.recv.return_value = {"kind": "error", "traceback": "initialization failed"}
    worker._connection.send.side_effect = BrokenPipeError("child already closed its pipe")
    worker._process = Mock()
    worker._process.is_alive.return_value = True
    with pytest.raises(RuntimeError, match="initialization failed"):
        try:
            worker.poll()
        finally:
            worker.close()
    worker._connection.send.assert_not_called()
    worker._process.terminate.assert_called_once()
