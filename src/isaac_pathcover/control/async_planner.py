"""A persistent SUPER planning process sharing Newton's advancing simulation clock."""
from __future__ import annotations

import multiprocessing as mp
from time import perf_counter
import traceback

import numpy as np


def _planning_process(connection, scene_name, seed, sim_time, observation):
    from isaac_pathcover.planning.pipeline import Planner
    from isaac_pathcover.simulation.scene import Scene
    from isaac_pathcover.trajectory.super_opt import warmup

    def measured():
        with observation.get_lock():
            values = np.asarray(observation[:], dtype=float)
        return values[:12].reshape(3, 4), values[12:16], float(values[16])

    def published_backup():
        with observation.get_lock():
            return bool(observation[17])

    try:
        scene = Scene(scene_name)
        clock_buffer = np.frombuffer(sim_time.get_obj(), dtype=np.float64)
        planner = Planner(scene, seed=seed, clock=lambda: sim_time.value,
                          state_reader=measured, clock_buffer=clock_buffer,
                          backup_state_reader=published_backup)
        warmup_started = perf_counter()
        warmup()
        planner.initialization_timings["numeric_initialization_s"] = perf_counter() - warmup_started
        lidar_started = perf_counter()
        scene.lidar(scene.start, [1., 0., 0., 0.])
        planner.initialization_timings["lidar_initialization_s"] = perf_counter() - lidar_started
        planning_goal = planner.map.nearest_inf_not_occupied(scene.goal, 3.)
        connection.send({"kind": "ready", "initialization": planner.initialization_timings,
                         "resolved_goal": planning_goal})
        while True:
            request = connection.recv()
            if request["kind"] == "stop":
                break
            started = perf_counter()
            mapping = None
            if request["scan"]:
                state, quaternion, _ = measured()
                scan_started = perf_counter()
                origin, endpoints = scene.lidar(state[:, 0], quaternion)
                sensor_elapsed = perf_counter() - scan_started
                update = planner.update_map(origin, endpoints)
                mapping = {"t": sim_time.value, "points": len(endpoints),
                           "sensor_s": sensor_elapsed, "sensor_origin": origin.tolist(),
                           "sensor_position": state[:, 0].tolist(),
                           "sensor_quaternion_wxyz": quaternion.tolist(), **update}
            state, quaternion, yaw = measured()
            planner.set_robot_state(state, yaw=yaw, quaternion=quaternion)
            plan_start = sim_time.value
            if request["operation"] == "plan_from_rest":
                result = planner.plan_from_rest(goal=planning_goal, new_goal=request["new_goal"])
            else:
                shifted = planner.map.nearest_inf_not_occupied(planning_goal, 3.)
                if shifted is not None:
                    planning_goal = shifted
                result = planner.replan_once(goal=planning_goal, new_goal=request["new_goal"])
            result.events["resolved_goal"] = planning_goal.tolist()
            connection.send({"kind": "result", "id": request["id"], "result": result,
                             "plan_start_t": plan_start, "plan_end_t": sim_time.value,
                             "mapping": mapping, "elapsed_wall_s": perf_counter() - started})
    except EOFError:
        pass
    except Exception:
        connection.send({"kind": "error", "traceback": traceback.format_exc()})
    finally:
        connection.close()


class AsyncPlanningWorker:
    """The ROS callback mutex becomes one in-flight CPU request; control never waits."""

    def __init__(self, scene_name, seed):
        context = mp.get_context("spawn")
        parent, child = context.Pipe(duplex=True)
        self._sim_time = context.Value("d", 0.)
        self._observation = context.Array("d", 18)
        self._connection = parent
        self._process = context.Process(
            target=_planning_process,
            args=(child, scene_name, seed, self._sim_time, self._observation),
            daemon=True, name=f"super-risp-{scene_name}-planner",
        )
        self._process.start()
        child.close()
        self.ready = False
        self.busy = False
        self.initialization_timings = None
        self.resolved_goal = None

    def observe(self, now, state, quaternion, yaw):
        self._sim_time.value = now
        with self._observation.get_lock():
            self._observation[:17] = np.r_[np.asarray(state).ravel(), quaternion, yaw]

    def set_backup_flag(self, on_backup):
        with self._observation.get_lock():
            self._observation[17] = float(on_backup)

    def advance_clock(self, now):
        self._sim_time.value = now

    def poll(self):
        if self._connection.poll():
            message = self._connection.recv()
            if message["kind"] == "error":
                self.ready = False
                raise RuntimeError("SUPER planning process failed:\n" + message["traceback"])
            if message["kind"] == "ready":
                self.ready = True
                self.initialization_timings = message["initialization"]
                self.resolved_goal = message["resolved_goal"]
            elif message["kind"] == "result":
                self.busy = False
                return message
        if not self._process.is_alive() and not self._connection.poll():
            raise RuntimeError(f"SUPER planning process exited: code {self._process.exitcode}")
        return None

    def submit(self, request):
        if not self.ready or self.busy:
            raise RuntimeError("planning worker is not ready")
        self.busy = True
        self._connection.send(request)

    def close(self):
        if self._process.is_alive():
            if self.ready and not self.busy:
                self._connection.send({"kind": "stop"})
                self._process.join(timeout=1.)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join()
        self._connection.close()
