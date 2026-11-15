"""SUPER planning with ROG-Map, A*, RISP corridors, and S4 trajectories.

Port of SUPER 2ad3419 super_core/super_planner.cpp and config/static_dense.yaml.
Copyright 2025 Yunfan REN, MaRS Lab, HKU; LGPL-3.0-or-later.
Only the geometric polytope generator is replaced by PathCover/RISP.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from time import perf_counter

import numpy as np
from scipy.spatial import QhullError
from scipy.spatial.transform import Rotation

from ..control.execution import BackupTraj, CmdTraj, ExpTraj, RetCode
from ..mapping.rog_map import GridType, ROGMap, ROGMapConfig
from ..trajectory.super_opt import BackupTrajOpt, ExpTrajOpt, YawTrajOpt
from .corridor import CorridorGenerator
from .geometry import GeometryError, chebyshev_center
from .search import AStar


@dataclass
class PlannerConfig:
    """The active source static_dense configuration, in SI units."""
    resolution: float = 0.05
    goal_vel_en: bool = False
    goal_yaw_en: bool = False
    use_fov_cut: bool = False
    frontend_in_known_free: bool = False
    corridor_bound_dis: float = 0.8
    corridor_line_max_length: float = 1.0
    safe_corridor_line_max_length: float = 6.0
    replan_forward_dt: float = 0.1
    planning_horizon: float = 7.0
    sensing_horizon: float = -1.0
    receding_dis: float = 1.0
    robot_r: float = 0.2
    yaw_dot_max: float = 3.0
    replan_rate: float = 15.0


@dataclass
class PlanResult:
    trajectory: object = None
    corridor: object = None
    search: object = None
    optimization: object = None
    timings: dict[str, float] = field(default_factory=dict)
    success: bool = False
    status: str = "FAILED"
    map_version: int = 0
    code: RetCode = RetCode.FAILED
    command: CmdTraj | None = None
    exp: ExpTraj | None = None
    backup: BackupTraj | None = None
    exp_status: str = "FAILED"
    backup_status: str = "NO_NEED"
    goal_valid: bool = True
    guide_path: np.ndarray | None = None
    guide_times: np.ndarray | None = None
    backup_optimization: object = None
    events: dict = field(default_factory=dict)


def simple_pm_time_allocator(a_max, v_max, v0, total_distance, distance):
    """Literal source simplePMTimeAllocator, including its quadratic formula."""
    def solve(a, b, c):
        return (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)
    t_to_vmax, t_to_v0 = v_max / a_max, v0 / a_max
    dis_to_vmax, dis_to_v0 = 0.5 * a_max * t_to_vmax ** 2, 0.5 * a_max * t_to_v0 ** 2
    dec_time = (v_max - v0) / a_max
    dec_distance = 0.5 * (v_max + v0) * dec_time
    if total_distance <= dis_to_v0:
        t = np.sqrt(2 * distance / a_max)
        return float(t), float(a_max * t)
    if total_distance <= dis_to_vmax + dec_distance:
        t_acc = solve(2 * a_max, -(a_max - v0), -(v0 * v0 / a_max + 2 * total_distance))
        dis_acc, current_vmax = 0.5 * a_max * t_acc ** 2, a_max * t_acc
        if distance <= dis_acc:
            t = np.sqrt(2 * distance / a_max)
            return float(t), float(a_max * t)
        t2 = solve(-a_max, 2 * current_vmax, -2 * (distance - dis_acc))
        return float(t_acc + t2), float(current_vmax - t2 * a_max)
    if distance < dis_to_vmax:
        t = np.sqrt(2 * distance / a_max)
        return float(t), float(a_max * t)
    if distance < total_distance - dec_distance:
        return float(t_to_vmax + (distance - dis_to_vmax) / v_max), float(v_max)
    constant_distance = total_distance - dec_distance - dis_to_vmax
    remaining_distance = distance - dis_to_vmax - constant_distance
    t_dec = solve(-a_max, 2 * v_max, -2 * remaining_distance)
    return float(t_to_vmax + constant_distance / v_max + t_dec), float(v_max - t_dec * a_max)


def _forward_sample_times(start, stop, step):
    """Batch the source ``t += dt`` loop without changing its rounding order."""
    if start >= stop:
        return np.empty(0)
    increments = np.full(int(np.ceil((stop - start) / step)) + 1, step)
    increments[0] = start
    times = np.cumsum(increments)
    return times[times < stop]


class Planner:
    def __init__(self, scene, seed=0, *, config=None, clock=None, state_reader=None,
                 clock_buffer=None, backup_state_reader=None):
        started = perf_counter()
        self.scene = scene
        self.config = config or PlannerConfig()
        self.clock = perf_counter if clock is None else clock
        self.state_reader = state_reader
        self.backup_state_reader = backup_state_reader
        cfg = self.config
        self.map = ROGMap(scene.lower, scene.upper, config=ROGMapConfig(resolution=cfg.resolution))
        self.map.set_occupied(scene.point_cloud())
        self.initialization_timings = {"point_cloud_mapping_s": perf_counter() - started}
        step = int(np.ceil(cfg.robot_r / cfg.resolution))
        offsets = np.stack(np.meshgrid(*([np.arange(-step, step + 1)] * 3), indexing="ij"), axis=-1).reshape(-1, 3)
        squared = np.sum(offsets * offsets, axis=1)
        offsets, squared = offsets[squared <= step * step], squared[squared <= step * step]
        self.line_neighbors = offsets[np.argsort(squared, kind="stable")]
        raycast_started = perf_counter()
        self.map.is_line_free(scene.start, scene.start, neighbors=self.line_neighbors)
        self.map.first_blocked_line(scene.start, np.asarray([scene.start]), 1.0, self.line_neighbors)
        self.initialization_timings["raycast_initialization_s"] = perf_counter() - raycast_started
        search_started = perf_counter()
        self.astar = AStar(self.map, clock=self.clock, clock_buffer=clock_buffer)
        self.initialization_timings["search_initialization_s"] = perf_counter() - search_started
        self.astar.set_fine_inf_neighbors(int(np.floor(cfg.robot_r / cfg.resolution)))
        self.corridor_generator = CorridorGenerator(
            self.map, bound_distance=cfg.corridor_bound_dis,
            line_max_length=cfg.corridor_line_max_length,
            robot_radius=cfg.robot_r, line_neighbors=self.line_neighbors, seed=seed)
        self.exp_optimizer = ExpTrajOpt()
        self.backup_optimizer = BackupTrajOpt()
        self.yaw_optimizer = YawTrajOpt(cfg.yaw_dot_max)
        self.state = np.zeros((3, 4))
        self.state[:, 0] = scene.start
        self.yaw = 0.0
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        self.odom_received = False
        self.goal = np.asarray(scene.goal, dtype=float).copy()
        self.goal_yaw = float("nan")
        self.new_goal = True
        self.goal_valid = True
        self.last_exp = ExpTraj()
        self.command = CmdTraj()
        self.robot_on_backup = False
        self.shifted_start = np.full(3, 9999.0)
        self._result = PlanResult()
        self._wall_start = 0.0

    @property
    def sample_traj_dt(self):
        return self.config.resolution / self.exp_optimizer.config.max_vel

    def set_robot_state(self, state, yaw=0.0, quaternion=None):
        self.state = np.asarray(state, dtype=float).copy()
        self.yaw = float(yaw)
        if quaternion is not None:
            self.quaternion = np.asarray(quaternion, dtype=float).copy()
        self.odom_received = True

    def _refresh_robot_state(self):
        if self.state_reader is not None:
            state, quaternion, yaw = self.state_reader()
            self.set_robot_state(state, yaw=yaw, quaternion=quaternion)

    def update_map(self, origin, endpoints):
        started, previous = perf_counter(), self.map.version
        self.map.update_map(origin, endpoints)
        return {"fusion_s": perf_counter() - started,
                "occupancy_changed": previous != self.map.version, "map_version": self.map.version}

    def _begin(self, goal, goal_yaw, new_goal):
        self._wall_start = perf_counter()
        self.goal = np.asarray(self.scene.goal if goal is None else goal, dtype=float).copy()
        self.goal_yaw, self.new_goal, self.goal_valid = float(goal_yaw), bool(new_goal), True
        self._result = PlanResult(map_version=self.map.version)
        self._result.events["planning_start_sim"] = float(self.clock())

    def _finish(self, code, status=None):
        out = self._result
        out.code = RetCode(code)
        out.success = code in (RetCode.SUCCESS, RetCode.FINISH)
        out.status = status or out.code.name
        out.goal_valid = self.goal_valid
        out.timings["total_s"] = perf_counter() - self._wall_start
        out.events["planning_end_sim"] = float(self.clock())
        return out

    def plan_from_rest(self, goal=None, goal_yaw=float("nan"), new_goal=True):
        self._begin(goal, goal_yaw, new_goal)
        self._refresh_robot_state()
        if not self.odom_received:
            return self._finish(RetCode.FAILED, "NO_ODOM")
        local_start = self.map.nearest_cell_not_occupied(self.state[:, 0], 3.0)
        if local_start is None:
            return self._finish(RetCode.FAILED, "NO_START_POINT")
        self.last_exp = ExpTraj()
        code, exp = self._generate_exp(local_start)
        self._result.exp_status = code.name
        self._result.exp = exp
        if code == RetCode.FAILED:
            return self._finish(RetCode.FAILED, self._result.status)
        code, backup = self._generate_backup(exp)
        self._result.backup, self._result.backup_status = backup, code.name
        if code == RetCode.SUCCESS:
            self._commit(exp, backup)
            return self._finish(RetCode.SUCCESS, "SUCCESS_WITH_BACKUP")
        if code in (RetCode.FINISH, RetCode.NO_NEED):
            self._commit(exp)
            return self._finish(RetCode.SUCCESS, "SUCCESS_NO_BACKUP")
        return self._finish(RetCode.FAILED, self._result.status)

    def replan_once(self, goal=None, goal_yaw=float("nan"), new_goal=False):
        self._begin(goal, goal_yaw, new_goal)
        started = perf_counter()
        code, exp = self._generate_exp(self.state[:, 0])
        self._result.timings["generate_exp_s"] = perf_counter() - started
        self._result.exp, self._result.exp_status = exp, code.name
        if code in (RetCode.FAILED, RetCode.NEW_TRAJ, RetCode.EMER):
            return self._finish(code, self._result.status)
        started = perf_counter()
        code, backup = self._generate_backup(exp)
        self._result.timings["generate_backup_s"] = perf_counter() - started
        self._result.backup, self._result.backup_status = backup, code.name
        # Source ReplanOnce, lines 233-237: this is its original acceptance rule.
        if perf_counter() - self._wall_start > self.config.replan_forward_dt * 0.9:
            return self._finish(RetCode.FAILED, "REPLAN_OVERTIME")
        if code == RetCode.SUCCESS:
            self._commit(exp, backup)
            return self._finish(RetCode.SUCCESS, "SUCCESS_WITH_BACKUP")
        if code == RetCode.NO_NEED:
            # The source updates last_exp but deliberately retains CmdTraj here.
            self.last_exp, self.robot_on_backup, self.new_goal = exp, False, False
            return self._finish(RetCode.SUCCESS, "SUCCESS_NO_BACKUP_RETAINED_COMMAND")
        if code == RetCode.FINISH:
            self._commit(exp)
            return self._finish(RetCode.SUCCESS, "SUCCESS_NO_BACKUP")
        return self._finish(RetCode.FAILED, self._result.status)

    def _commit(self, exp, backup=None):
        command = CmdTraj.from_trajectories(exp, backup)
        command.last_yaw = self.yaw if self.command.empty else self.command.last_yaw
        self.command = command
        self.last_exp, self.robot_on_backup, self.new_goal = exp, False, False
        self._result.command = self.command

    def _path_search(self, start, horizon):
        if horizon <= 0.0:
            return None
        if self.map.get_grid_type(start) in (GridType.OCCUPIED, GridType.OUT_OF_MAP):
            return None
        unknown = self.config.frontend_in_known_free
        escaped = self.astar.escape_path_search(start, unknown_as_occupied=unknown)
        self._result.search = escaped
        if escaped.status not in ("NO_NEED", "REACH_HORIZON", "REACH_GOAL"):
            return None
        shifted_start = start if escaped.status == "NO_NEED" else escaped.path[-1]
        result = self.astar.search(shifted_start, self.goal, horizon=horizon,
                                   inflated=True, unknown_as_occupied=unknown)
        if result.status == "INIT_ERROR":
            self.goal_valid = False
            return None
        if result.status == "NO_PATH":
            result = self.astar.search(shifted_start, self.goal, horizon=horizon,
                                      inflated=False, unknown_as_occupied=unknown,
                                      use_inf_neighbor=True)
        self._result.search = result
        if not result.success:
            return None
        path = result.path
        if escaped.status != "NO_NEED":
            path = np.vstack((escaped.path, path))
        if not len(path):
            return None
        path = np.vstack((start, path))
        if result.status == "REACH_GOAL":
            path = np.vstack((path, self.goal))
        return path

    def _exp_failure(self, reason, exp=None):
        self._result.status = reason
        return RetCode.FAILED, ExpTraj() if exp is None else exp

    def _generate_exp(self, local_start):
        frontend_start = perf_counter()
        cfg = self.config
        # Source reads the live odometry yaw before recording replan start time.
        self._refresh_robot_state()
        now = float(self.clock())
        init = np.zeros((3, 4))
        init_yaw, final_yaw = np.array([self.yaw, 0.0, 0.0, 0.0]), np.zeros(4)
        guide, stamps, end_vel = [], [], 0.0
        old = self.last_exp
        process_tt = replan_tt = -1.0
        if old.empty:
            init[:, 0] = local_start
        else:
            process_tt = now - old.start_time
            replan_tt = process_tt + cfg.replan_forward_dt
            self._result.events.update(predicted_start_tt=replan_tt, retained_prefix_s=cfg.replan_forward_dt)
            early = (replan_tt >= self.command.duration or replan_tt >= old.duration
                     or (not self.new_goal and len(old.corridor.regions) == 1 and old.connected_goal))
            if not early and not self.new_goal:
                early = np.linalg.norm(self.goal - old.position.evaluate(replan_tt)) < cfg.resolution * 3
                if early:
                    old = replace(old, connected_goal=True)
            if early:
                # Source reads the last published command flag; odometry never updates it.
                if self.backup_state_reader is not None:
                    self.robot_on_backup = bool(self.backup_state_reader())
                if self.robot_on_backup:
                    return self._exp_failure("ON_BACKUP_WAIT_FOR_REST", old)
                self._result.trajectory, self._result.corridor = old.position, old.corridor
                return RetCode.NO_NEED, old
            trajectory = self.command.position
            positions, times, velocities = [], [], []
            last_sample = trajectory.evaluate(replan_tt)
            old.whole_known_free = True
            sampled_times = _forward_sample_times(replan_tt + self.sample_traj_dt, trajectory.duration,
                                                   self.sample_traj_dt)
            sampled_positions = trajectory.evaluate(sampled_times)
            sampled_velocities = np.linalg.norm(trajectory.evaluate(sampled_times, derivative=1), axis=1)
            sampled_types = self.map.get_inf_grid_type(sampled_positions)
            for t, point, velocity, grid_type in zip(sampled_times, sampled_positions, sampled_velocities, sampled_types):
                if np.linalg.norm(point - last_sample) < cfg.resolution * 0.8:
                    continue
                if grid_type in (GridType.OCCUPIED, GridType.OUT_OF_MAP):
                    old.whole_known_free = False
                    break
                positions.append(point)
                times.append(float(t))
                velocities.append(float(velocity))
                last_sample = point
            split_distance = cfg.receding_dis
            if old.whole_known_free and not self.new_goal and cfg.receding_dis > 0.0:
                split_distance = np.finfo(float).max
            init = trajectory.state(replan_tt)
            if split_distance > 0.0:
                while positions and (self.map.is_occupied_inflate(positions[-1])
                                     or np.linalg.norm(positions[-1] - init[:, 0]) > split_distance):
                    positions.pop()
                    times.pop()
                    velocities.pop()
            else:
                positions = []
            if positions:
                guide = positions
                stamps = [t - times[0] for t in times]
                end_vel = velocities[-1]
            else:
                guide, stamps = [init[:, 0]], [0.0]
                self._refresh_robot_state()
                end_vel = float(np.linalg.norm(self.state[:, 1]))
        length = float(np.linalg.norm(np.diff(guide, axis=0), axis=1).sum()) if len(guide) > 1 else 0.0
        horizon = cfg.planning_horizon - length
        if not guide or np.linalg.norm(guide[0] - init[:, 0]) > 1e-2:
            guide.insert(0, init[:, 0])
            stamps.insert(0, 0.0)
        if horizon > cfg.resolution * 2:
            if np.linalg.norm(guide[-1] - self.goal) < cfg.resolution * 5:
                stamps.append(stamps[-1] + np.linalg.norm(guide[-1] - self.goal) / self.exp_optimizer.config.max_vel)
                guide.append(self.goal.copy())
            else:
                search_start = perf_counter()
                new_path = self._path_search(guide[-1], horizon)
                self._result.timings["search_s"] = perf_counter() - search_start
                if new_path is None or len(new_path) < 2:
                    return self._exp_failure("PATH_SEARCH_FAILED")
                distances = np.zeros(len(new_path))
                total = 0.0
                for i in range(len(new_path) - 2, -1, -1):
                    total += np.linalg.norm(new_path[i + 1] - new_path[i])
                    distances[i + 1] = total
                total += np.linalg.norm(new_path[0] - guide[-1])
                distances[0] = total
                dt = np.zeros(len(new_path))
                last_stamp = 0.0
                for i in range(len(distances) - 1, -1, -1):
                    stamp, _ = simple_pm_time_allocator(self.exp_optimizer.config.max_acc,
                                                       self.exp_optimizer.config.max_vel,
                                                       end_vel, total, distances[i])
                    dt[i], last_stamp = stamp - last_stamp, stamp
                stamp = stamps[-1]
                for i in range(1, len(new_path)):
                    stamp += dt[i]
                    guide.append(new_path[i])
                    stamps.append(stamp)
        guide, stamps = np.asarray(guide), np.asarray(stamps)
        connected = np.linalg.norm(guide[-1, :2] - self.goal[:2]) < cfg.resolution * 2
        self._result.guide_path, self._result.guide_times = guide, stamps
        corridor_start = perf_counter()
        self.shifted_start = np.full(3, 9999.0)
        try:
            corridor = self.corridor_generator.search_polytope_on_path(guide, cut_first_poly=cfg.use_fov_cut)
        except (GeometryError, QhullError) as error:
            self._result.events["corridor_error"] = str(error)
            return self._exp_failure("CORRIDOR_FAILED")
        self._result.timings["corridor_s"] = perf_counter() - corridor_start
        self._result.corridor = corridor
        self.shifted_start = corridor.shifted_start
        self._result.timings["exp_frontend_s"] = perf_counter() - frontend_start
        end = np.zeros((3, 4))
        end[:, 0] = guide[-1]
        if cfg.goal_vel_en:
            self._refresh_robot_state()
        distance = np.linalg.norm(self.goal - self.state[:, 0])
        if cfg.goal_vel_en and distance > cfg.planning_horizon / 2:
            end[:, 1] = (self.goal - self.state[:, 0]) / distance * self.exp_optimizer.config.max_vel / 2
        if np.linalg.norm(end[:, 0] - self.goal) < cfg.resolution * 2:
            end[:, 0], end[:, 1] = self.goal, 0.0
        optimize_start = perf_counter()
        optimum = self.exp_optimizer.optimize(init, end, guide, stamps, corridor.regions)
        self._result.timings["exp_opt_s"] = perf_counter() - optimize_start
        self._result.optimization = optimum
        self._result.trajectory = optimum.trajectory
        if not optimum.success:
            return self._exp_failure("EXP_OPT_FAILED")
        # Source generateExpTraj lines 757-760 uses getSimTime, unlike ReplanOnce.
        if float(self.clock()) - now > cfg.replan_forward_dt:
            return self._exp_failure("EXP_SIM_OVERTIME")
        position = optimum.trajectory
        if not old.empty:
            process_tt = now - self.command.start_time
            try:
                prefix = self.command.position.cut(process_tt, replan_tt)
            except ValueError:
                return self._exp_failure("EXP_PREFIX_FAILED")
            position = prefix.concatenate(position)
            init_yaw = self.command.yaw.state(replan_tt)[0]
        free_goal = not (cfg.goal_yaw_en and np.isfinite(self.goal_yaw) and connected)
        if not free_goal:
            final_yaw[0] = self.goal_yaw
        yaw_start = perf_counter()
        yaw = self.yaw_optimizer.optimize(init_yaw, final_yaw, optimum.trajectory,
                                          order=3, free_start=False, free_goal=free_goal)
        self._result.timings["exp_yaw_s"] = perf_counter() - yaw_start
        if yaw is None:
            return self._exp_failure("EXP_YAW_FAILED")
        if not old.empty:
            try:
                prefix = self.command.yaw.cut(process_tt, replan_tt)
            except ValueError:
                return self._exp_failure("EXP_YAW_PREFIX_FAILED")
            yaw = prefix.concatenate(yaw)
        backup_start = backup_end = -1.0
        if not old.empty and replan_tt > self.command.backup_start:
            backup_start = self.command.backup_start - process_tt
            backup_end = replan_tt - process_tt
        exp = ExpTraj(position, yaw, now, corridor, connected, False, backup_start, backup_end)
        self._result.trajectory = position
        return RetCode.SUCCESS, exp

    def _backup_failure(self, reason, backup, code=RetCode.FAILED):
        self._result.status = reason
        return code, backup

    def _cut_backup_corridor(self, poly, seed):
        cfg = self.config
        robot = self.state[:, 0]
        direction = seed - robot
        if cfg.use_fov_cut:
            center, radius = chebyshev_center(poly.A, poly.b)
            if radius < 0.1:
                raise GeometryError("source FOV cut requires interior distance >= 0.1")
            rotation = Rotation.from_quat(self.quaternion[[1, 2, 3, 0]]).as_matrix()
            body = rotation.T @ direction
            yaw = np.arctan2(body[1], body[0])
            rotation = rotation @ Rotation.from_euler("z", yaw).as_matrix()
            origin = robot - 0.2 * direction / np.linalg.norm(direction)
            inner = rotation[:, 0] + origin
            normals, bounds = [], []
            # Source FOVChecker constructs these points with its raw angle args.
            for angle in (35.0, -35.0):
                r, z = 5.0 * np.cos(angle), 5.0 * np.sin(angle)
                a, b = rotation @ np.array([r, -3.0, z]) + origin, rotation @ np.array([r, 3.0, z]) + origin
                normal = np.cross(b - a, origin - a)
                normal /= np.linalg.norm(normal)
                bound = float(normal @ a)
                if normal @ inner > bound:
                    normal, bound = -normal, -bound
                normals.append(normal)
                bounds.append(bound)
            poly.planes = np.vstack((poly.planes, np.column_stack((normals, -np.asarray(bounds)))))
        if cfg.sensing_horizon > 0:
            direction /= np.linalg.norm(direction)
            point = robot + direction * cfg.sensing_horizon
            poly.planes = np.vstack((poly.planes, np.r_[direction, -direction @ point]))
            _, radius = chebyshev_center(poly.A, poly.b)
            if radius <= 0.0:
                raise GeometryError("sensing horizon cut has no interior")
        return poly

    def _generate_backup(self, exp):
        cfg = self.config
        # Source lines 829-831 take a fresh robot-position snapshot after Exp.
        self._refresh_robot_state()
        started = perf_counter()
        backup = BackupTraj(robot_position=self.state[:, 0].copy())
        total = exp.duration
        start_tt = float(self.clock()) - exp.start_time
        if start_tt > total - 0.01:
            return RetCode.NO_NEED, backup
        trajectory = exp.position
        last_position = trajectory.evaluate(start_tt)
        evaluated, times = [], []
        all_visible = True
        limit = (min(cfg.sensing_horizon, cfg.safe_corridor_line_max_length)
                 if cfg.sensing_horizon > 0 else cfg.safe_corridor_line_max_length)
        out_t = start_tt
        sampled_times = _forward_sample_times(start_tt, total, self.sample_traj_dt)
        sampled_positions = trajectory.evaluate(sampled_times)
        for out_t, point in zip(sampled_times, sampled_positions):
            if np.linalg.norm(last_position - point) < cfg.resolution * 0.8:
                continue
            last_position = point
            evaluated.append(point)
            times.append(float(out_t))
        blocked = self.map.first_blocked_line(backup.robot_position, evaluated, limit, self.line_neighbors)
        if blocked < len(evaluated):
            all_visible = False
            evaluated, times = evaluated[:blocked + 1], times[:blocked + 1]
            out_t = times[-1]
        if all_visible:
            try:
                backup.corridor = self.corridor_generator.generate_from_line(backup.robot_position,
                                                                            trajectory.evaluate(total))
                backup.corridor.known_free = True
            except (GeometryError, QhullError):
                pass  # Source ignores this visualization-only corridor failure.
            self._result.timings["backup_frontend_s"] = perf_counter() - started
            return RetCode.FINISH, backup
        invisible_point = evaluated[-1]
        reverse_times = []
        while out_t > start_tt:
            out_t -= self.sample_traj_dt
            reverse_times.append(out_t)
        if reverse_times:
            reverse_points = trajectory.evaluate(reverse_times)
            beyond = np.flatnonzero(np.linalg.norm(reverse_points - invisible_point, axis=1) > cfg.robot_r)
            if len(beyond):
                out_t = reverse_times[beyond[0]]
        seed_time = max(start_tt, out_t)
        seed = trajectory.evaluate(seed_time)
        self._refresh_robot_state()
        shifted_robot = self.state[:, 0] if np.linalg.norm(self.shifted_start) > 999 else self.shifted_start
        shifted_robot = self.map.nearest_cell_not_occupied(shifted_robot, 3.0)
        if shifted_robot is None:
            return self._backup_failure("BACKUP_NO_START_POINT", backup)
        try:
            poly = self.corridor_generator.generate_from_line(shifted_robot, seed)
            _, radius = chebyshev_center(poly.A, poly.b)
            if radius <= 0.0:
                return self._backup_failure("BACKUP_NO_INTERIOR", backup)
            if cfg.use_fov_cut or cfg.sensing_horizon > 0:
                self._refresh_robot_state()
            poly = self._cut_backup_corridor(poly, seed)
        except (GeometryError, QhullError) as error:
            self._result.events["backup_corridor_error"] = str(error)
            return self._backup_failure("BACKUP_CORRIDOR_FAILED", backup)
        backup.corridor = poly
        backup.corridor.known_free = True
        eval_t, last_position = times[-1] + self.sample_traj_dt, evaluated[-1]
        continuation_times = []
        while eval_t < total:
            continuation_times.append(eval_t)
            eval_t += self.sample_traj_dt
        continuation_points = trajectory.evaluate(continuation_times)
        for eval_t, point in zip(continuation_times, continuation_points):
            if not poly.contains(evaluated[-1]):
                break
            if np.linalg.norm(point - last_position) < cfg.resolution * 0.8:
                continue
            evaluated.append(point)
            times.append(float(eval_t))
            last_position = point
        evaluated.pop()
        times.pop()
        if not evaluated:
            # The C++ implementation dereferences back() here; expose failure.
            return self._backup_failure("BACKUP_EMPTY_VISIBLE_INTERVAL", backup)
        t0, te = float(self.clock()) - exp.start_time + 0.01, times[-1]
        velocity = np.linalg.norm(trajectory.evaluate(te, derivative=1))
        heu_ts = max((t0 + te) / 2, te - velocity / self.backup_optimizer.config.max_acc)
        heu_duration = te - heu_ts
        self._result.events["backup_interval"] = {"t0": t0, "te": te, "heuristic_ts": heu_ts}
        self._result.timings["backup_frontend_s"] = perf_counter() - started
        started = perf_counter()
        optimum = self.backup_optimizer.optimize(trajectory, t0, te, heu_ts,
                                                 evaluated[-1], heu_duration, poly)
        self._result.timings["backup_opt_s"] = perf_counter() - started
        self._result.backup_optimization = optimum
        if not optimum.success:
            return self._backup_failure("BACKUP_OPT_FAILED", backup, RetCode.OPT_FAILED)
        switch_time = float(optimum.switch_time)
        init_yaw = exp.yaw.state(switch_time)[0]
        goal_yaw = np.zeros(4)
        free_goal = not (cfg.goal_yaw_en and np.isfinite(self.goal_yaw))
        if not free_goal:
            goal_yaw[0] = self.goal_yaw
        started = perf_counter()
        yaw = self.yaw_optimizer.optimize(init_yaw, goal_yaw, optimum.trajectory,
                                          order=3, free_start=False, free_goal=free_goal)
        self._result.timings["backup_yaw_s"] = perf_counter() - started
        if yaw is None or switch_time < t0:
            return self._backup_failure("BACKUP_YAW_OR_START_FAILED", backup, RetCode.OPT_FAILED)
        new_start = exp.start_time + switch_time
        # Preserve the source comparison, including its WT/TT naming mismatch.
        if self.command.backup_start < self.command.duration and new_start < self.command.backup_start:
            return self._backup_failure("BACKUP_BEFORE_COMMITTED_START", backup, RetCode.OPT_FAILED)
        backup.position, backup.yaw = optimum.trajectory, yaw
        backup.start_time, backup.start_tt = new_start, switch_time
        return RetCode.SUCCESS, backup
