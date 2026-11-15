"""Persistent CPU planning process; Newton and its CUDA context stay in the flight process.

The process receives one snapshot at a time, never blocks the physics loop,
and returns only fully solved candidates. A monotonic wall-clock deadline is
enforced cooperatively inside the unchanged GCOPTER objective/line search.
"""

from __future__ import annotations

import multiprocessing as mp
from time import perf_counter
import traceback


def handoff_horizon(budget_wall_s, minimum_sim_s, observed_sim_per_wall):
    """SUPER-style forward shift, scaled to the measured Newton clock rate.

    Allow 20% schedule slack and 0.1 s simulation time for IPC and control.
    The measured ratio is advisory; unexpectedly fast simulation can still
    produce a stale result, which the flight runner must reject explicitly.
    """
    return max(minimum_sim_s, 1.2 * budget_wall_s * max(1., observed_sim_per_wall) + .1)


def _planning_process(connection, scene_name, seed):
    from isaac_pathcover.control.execution import execution_check
    from isaac_pathcover.planning.pipeline import Planner
    from isaac_pathcover.simulation.scene import Scene
    from isaac_pathcover.trajectory.diagnostics import trajectory_diagnostics

    try:
        scene = Scene(scene_name)
        planner = Planner(scene, seed=seed)
        connection.send({"kind": "ready", "initialization": planner.initialization_timings})
        while True:
            request = connection.recv()
            if request["kind"] == "stop":
                break
            started = perf_counter()
            mapping = None
            if request["scan"]:
                scan_started = perf_counter()
                origin, endpoints = scene.lidar(request["position"], request["quaternion"])
                sensor_elapsed = perf_counter() - scan_started
                update = planner.update_map(origin, endpoints)
                mapping = {"t": request["t"], "points": len(endpoints),
                           "sensor_s": sensor_elapsed, "sensor_origin": origin.tolist(), **update}
            result = planner.plan(request["start_state"], guide=request["guide"],
                                  guide_elapsed=request["guide_elapsed"], deadline=request["deadline"])
            diagnostics = validation = None
            if result.success and perf_counter() < request["deadline"]:
                diagnostics = trajectory_diagnostics(result.trajectory, result.corridor.polytopes,
                                                     planner.config)
                validation = execution_check(diagnostics, planner.config)
            ended = perf_counter()
            connection.send({"kind": "result", "id": request["id"], "result": result,
                             "mapping": mapping, "diagnostics": diagnostics,
                             "validation": validation, "elapsed_wall_s": ended - started,
                             "deadline_exceeded": ended > request["deadline"]})
    except EOFError:
        pass
    except Exception:
        try:
            connection.send({"kind": "error", "traceback": traceback.format_exc()})
        except (BrokenPipeError, OSError):
            pass
    finally:
        connection.close()


class AsyncPlanningWorker:
    """One in-flight request, no queued backlog, and no CUDA object transfer."""

    def __init__(self, scene_name, seed):
        context = mp.get_context("spawn")
        parent, child = context.Pipe(duplex=True)
        self._connection = parent
        self._process = context.Process(target=_planning_process, args=(child, scene_name, seed),
                                        daemon=True, name=f"pathcover-{scene_name}-planner")
        self._process.start()
        child.close()
        self.ready = False
        self.busy = False
        self.initialization_timings = None

    def poll(self):
        if self._connection.poll():
            message = self._connection.recv()
            if message["kind"] == "error":
                raise RuntimeError("asynchronous planner failed:\n" + message["traceback"])
            if message["kind"] == "ready":
                self.ready = True
                self.initialization_timings = message["initialization"]
            elif message["kind"] == "result":
                self.busy = False
                return message
        if not self._process.is_alive() and not self._connection.poll():
            raise RuntimeError(f"asynchronous planner exited: code {self._process.exitcode}")
        return None

    def submit(self, request):
        if not self.ready or self.busy:
            raise RuntimeError("planning worker is not ready for a new request")
        self.busy = True
        self._connection.send(request)

    def close(self):
        if self._process.is_alive():
            if not self.busy:
                self._connection.send({"kind": "stop"})
                self._process.join(timeout=0.5)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=2.)
        self._connection.close()
