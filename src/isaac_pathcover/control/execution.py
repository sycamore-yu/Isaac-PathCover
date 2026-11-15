"""SUPER trajectory containers, command sampling, and execution state machine.

Port of SUPER 2ad3419 data_structure/{exp,backup,cmd}_traj.h, fsm.cpp,
and fsm_ros2.hpp. Copyright 2025 Yunfan REN, MaRS Lab, HKU; LGPL-3.0-or-later.
The Newton adapter publishes the same PVAJ/yaw command on its control clock.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from ..trajectory.minco import PiecewisePolynomial


class RetCode(IntEnum):
    FAILED = 0
    NO_NEED = 1
    SUCCESS = 2
    FINISH = 3
    NEW_TRAJ = 4
    EMER = 5
    OPT_FAILED = 6
    INIT_ERROR = 7


@dataclass
class ExpTraj:
    position: PiecewisePolynomial | None = None
    yaw: PiecewisePolynomial | None = None
    start_time: float = 0.0
    corridor: object = None
    connected_goal: bool = False
    whole_known_free: bool = False
    first_backup_start: float = -1.0
    first_backup_end: float = -1.0

    @property
    def empty(self):
        return self.position is None

    @property
    def duration(self):
        return 0.0 if self.empty else self.position.duration


@dataclass
class BackupTraj:
    position: PiecewisePolynomial | None = None
    yaw: PiecewisePolynomial | None = None
    start_time: float = 0.0
    start_tt: float = -1.0
    corridor: object = None
    robot_position: np.ndarray | None = None

    @property
    def empty(self):
        return self.position is None

    @property
    def duration(self):
        return 0.0 if self.empty else self.position.duration


@dataclass
class CommandSample:
    state: np.ndarray
    yaw: float
    yaw_rate: float
    on_backup: bool
    finished: bool


@dataclass
class CmdTraj:
    position: PiecewisePolynomial | None = None
    yaw: PiecewisePolynomial | None = None
    start_time: float = 0.0
    backup_start: float = 99999999.0
    backup_available: bool = False
    first_backup_start: float = -1.0
    first_backup_end: float = -1.0
    last_yaw: float = 0.0

    @property
    def empty(self):
        return self.position is None

    @property
    def duration(self):
        return 0.0 if self.empty else self.position.duration

    @classmethod
    def from_trajectories(cls, exp: ExpTraj, backup: BackupTraj | None = None):
        position, yaw = exp.position, exp.yaw
        if backup is not None and not backup.empty:
            position = position.cut(0.0, backup.start_tt).concatenate(backup.position)
            yaw = yaw.cut(0.0, backup.start_tt).concatenate(backup.yaw)
        return cls(position, yaw, exp.start_time,
                   backup.start_tt if backup is not None and not backup.empty else 99999999.0,
                   backup is not None and not backup.empty,
                   exp.first_backup_start, exp.first_backup_end)

    def is_on_backup(self, trajectory_time):
        if not self.backup_available:
            return False
        first_valid = (self.first_backup_start >= 0.0 and self.first_backup_end >= 0.0
                       and self.first_backup_end >= self.first_backup_start)
        if first_valid and self.first_backup_start < trajectory_time < self.first_backup_end:
            return True
        return trajectory_time > self.backup_start

    def sample(self, now):
        elapsed = float(now) - self.start_time
        finished = elapsed > self.duration
        time = self.duration if finished else elapsed
        state = self.position.state(time)
        yaw = float(self.yaw.evaluate(time)[0])
        yaw_rate = float(self.yaw.evaluate(time, derivative=1)[0])
        if np.isnan(yaw):
            yaw, yaw_rate = self.last_yaw, 0.0
        else:
            self.last_yaw = yaw
        if np.isnan(yaw_rate):
            yaw_rate = 0.0
        return CommandSample(state, yaw, yaw_rate, self.is_on_backup(time), finished)


class TrajectoryExecution:
    def __init__(self, initial_position):
        self.command = None
        self.hover_position = np.asarray(initial_position, dtype=float).copy()

    @property
    def trajectory(self):
        return None if self.command is None else self.command.position

    @property
    def start_time(self):
        return 0.0 if self.command is None else self.command.start_time

    def sample(self, now):
        if self.command is None:
            return CommandSample(np.column_stack((self.hover_position, np.zeros((3, 3)))),
                                 0.0, 0.0, False, False)
        return self.command.sample(now)

    def reference(self, now):
        return self.sample(now).state

    def install(self, command, now=0.0):
        previous = self.reference(now)
        if self.command is not None:
            command.last_yaw = self.command.last_yaw
        self.command = command
        current = self.reference(now)
        return {"accepted": True, "start_time": command.start_time,
                "installed_time": float(now), "backup_available": command.backup_available,
                "backup_start_tt": command.backup_start,
                "reference_jump": np.linalg.norm(current - previous, axis=0).tolist()}


class MachineState(IntEnum):
    INIT = 0
    WAIT_GOAL = 1
    YAWING = 2
    GENERATE_TRAJ = 3
    FOLLOW_TRAJ = 4
    EMER_STOP = 5


class ExecutionFSM:
    """The source FSM; the caller dispatches planning requests asynchronously.

    The planner callback is mutually exclusive. This object only generates a
    request when the adapter calls it with the planning worker available.
    """
    def __init__(self, initial_position, goal, goal_yaw=float("nan")):
        self.position = np.asarray(initial_position, dtype=float).copy()
        self.goal = np.asarray(goal, dtype=float).copy()
        self.goal_yaw = float(goal_yaw)
        self.state = MachineState.INIT
        self.new_goal = np.linalg.norm(self.position - self.goal) >= 0.1
        self.started = self.new_goal
        self.finish_plan = False
        self.plan_from_rest = False
        self.stop = False
        self.transitions = []
        self.fsm_start_time = None
        self.last_print_time = 0.0
        self.pending_operation = None

    def _change(self, state, now, callback):
        self.transitions.append({"time": float(now), "from": self.state.name,
                                 "to": state.name, "callback": callback})
        self.state = state

    def main_tick(self, position, now=0.0, odom_received=True, odom_time=None):
        if self.stop:
            return None
        self.position = np.asarray(position, dtype=float).copy()
        if self.fsm_start_time is None:
            self.fsm_start_time = float(now)
        elapsed = float(now) - self.fsm_start_time
        if elapsed - self.last_print_time > 1.0:
            self.last_print_time = elapsed
            if not odom_received or (odom_time is not None and now - odom_time > 0.1):
                return None
        if self.state == MachineState.INIT:
            if self.started:
                self._change(MachineState.WAIT_GOAL, now, "MainFsmCallback")
        elif self.state == MachineState.WAIT_GOAL:
            if self.new_goal:
                self._change(MachineState.GENERATE_TRAJ, now, "MainFsmCallback")
        elif self.state == MachineState.GENERATE_TRAJ:
            if np.linalg.norm(self.position - self.goal) < 0.1:
                self._change(MachineState.WAIT_GOAL, now, "MainFsmCallback")
                self.new_goal, self.finish_plan = False, True
            else:
                self.pending_operation = "plan_from_rest"
                return "plan_from_rest"
        elif self.state == MachineState.EMER_STOP:
            self._change(MachineState.WAIT_GOAL, now, "MainFsmCallback")
        return None

    def replan_tick(self):
        if self.stop or self.state != MachineState.FOLLOW_TRAJ or self.finish_plan:
            return None
        if self.plan_from_rest:
            self.plan_from_rest = False
            return None
        self.pending_operation = "replan_once"
        return "replan_once"

    def accept_result(self, result, now=0.0):
        code = RetCode(result.code)
        operation, self.pending_operation = self.pending_operation, None
        if operation == "plan_from_rest":
            if not result.goal_valid:
                self._change(MachineState.WAIT_GOAL, now, "MainFsmCallback")
            elif code in (RetCode.SUCCESS, RetCode.FINISH):
                self.new_goal = False
                self.plan_from_rest = True
                self.finish_plan = code == RetCode.FINISH
                self._change(MachineState.FOLLOW_TRAJ, now, "MainFsmCallback")
        elif operation == "replan_once":
            if code == RetCode.EMER:
                self._change(MachineState.EMER_STOP, now, "ReplanTimerCallback")
            elif code == RetCode.NEW_TRAJ:
                self._change(MachineState.GENERATE_TRAJ, now, "ReplanTimerCallback")
            elif code in (RetCode.SUCCESS, RetCode.FINISH):
                self.new_goal = False

    def observe_command(self, sample, position, now=0.0):
        # Source PubCmd calls closeToGoal on the last MainFsm odometry snapshot.
        # Planning can block MainFsm while newer physical observations arrive.
        if self.state not in (MachineState.FOLLOW_TRAJ, MachineState.EMER_STOP):
            return
        if sample.finished:
            next_state = (MachineState.WAIT_GOAL if np.linalg.norm(self.position - self.goal) < 0.1
                          else MachineState.GENERATE_TRAJ)
            self._change(next_state, now, "PubCmdCallback")
