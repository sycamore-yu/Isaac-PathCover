"""Flight metrics from recorded states, independent of the planning objective."""
import numpy as np
from collections import Counter
from scipy.spatial.transform import Rotation


def _valid_poses(states):
    return np.isfinite(states[:, :7]).all(axis=1) & (np.linalg.norm(states[:, 3:7], axis=1) > 0.)


def _finite_extreme(values, operation):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return float(operation(values)) if values.size else None


def body_collisions(scene, states):
    """Test valid poses against the solids; invalid poses remain unchecked."""
    states = np.asarray(states)
    checked = np.flatnonzero(_valid_poses(states))
    collision = np.zeros(len(states), dtype=bool)
    if not len(checked):
        return collision
    rotation = Rotation.from_quat(states[checked][:, [4, 5, 6, 3]]).as_matrix()
    centers = (scene.box_lower + scene.box_upper) / 2
    half = (scene.box_upper - scene.box_lower) / 2
    body_half = scene.body_size / 2
    identity = np.eye(3)
    for i, r in zip(checked, rotation):
        state = states[i]
        extent = np.abs(r) @ body_half
        candidates = np.all(np.abs(centers - state[:3]) <= half + extent, axis=1)
        if not candidates.any():
            continue
        axes = np.concatenate((identity, r.T, np.cross(identity[:, None], r.T[None]).reshape(-1, 3)))
        axes = axes[np.linalg.norm(axes, axis=1) > 1e-12]
        radius_body = np.abs(axes @ r) @ body_half
        radius_obstacle = half[candidates] @ np.abs(axes).T
        projections = np.abs((centers[candidates] - state[:3]) @ axes.T)
        collision[i] = np.any(np.all(projections <= radius_body + radius_obstacle + 1e-12, axis=1))
    return collision


def summarize(scene, times, states, references, substeps, contact_force, motor_states, plans, wall_time, *, arrived):
    """Report the runner's settled-arrival decision and all observed failures.

    ``substeps`` contains poses before each integration. The final recorded state
    supplies the last endpoint. Invalid samples remain in the caller's trace;
    partial statistics use valid samples and expose their counts below.
    """
    states = np.asarray(states)
    references = np.asarray(references)
    physics_states = np.concatenate((np.asarray(substeps).reshape(-1, states.shape[1]), states[-1:]))
    valid = np.isfinite(states).all(axis=1) & _valid_poses(states)
    valid_states = states[valid]
    valid_tracking = valid & np.isfinite(references[:, :, 0]).all(axis=1)
    error = np.linalg.norm(states[valid_tracking, :3] - references[valid_tracking, :, 0], axis=1)
    valid_physics_poses = _valid_poses(physics_states)
    valid_dynamics = physics_states[valid_physics_poses & np.isfinite(physics_states).all(axis=1)]
    collisions = body_collisions(scene, physics_states)
    rate = np.abs(valid_dynamics[:, 10:13])
    tilt = np.empty(0)
    if len(valid_dynamics):
        rotation = Rotation.from_quat(valid_dynamics[:, [4, 5, 6, 3]]).as_matrix()
        tilt = np.arccos(np.clip(rotation[:, 2, 2], -1., 1.))
    finite = all(np.isfinite(value).all() for value in (states, references, physics_states, contact_force, motor_states))
    arrived = bool(arrived and finite and valid.all() and valid_physics_poses.all())
    max_contact_force = _finite_extreme(contact_force, np.max)
    collision = bool(collisions.any() or (max_contact_force is not None and max_contact_force > 1e-6))
    valid_positions = np.isfinite(physics_states[:, :3]).all(axis=1)
    outside = valid_positions & np.any((physics_states[:, :3] < scene.lower) | (physics_states[:, :3] > scene.upper), axis=1)
    durations = np.asarray([p["timings"]["total_s"] for p in plans]) if plans else np.empty(0)
    return {
        "scene": scene.name, "scene_provenance": scene.provenance,
        "arrived": arrived, "collision": collision, "collision_free_arrival": arrived and not collision,
        "goal_error_m": _finite_extreme(np.linalg.norm(states[-1, :3] - scene.goal), np.max),
        "duration_sim_s": float(times[-1]), "wall_time_s": wall_time,
        "tracking_rmse_m": float(np.sqrt(np.mean(error ** 2))) if len(error) else None,
        "tracking_max_m": _finite_extreme(error, np.max),
        "tracking_samples": int(valid_tracking.sum()),
        "valid_state_samples": int(valid.sum()), "invalid_state_samples": int((~valid).sum()),
        "max_speed_m_s": _finite_extreme(np.linalg.norm(valid_dynamics[:, 7:10], axis=1), np.max),
        "max_body_rates_rad_s": [_finite_extreme(rate[:, i], np.max) for i in range(3)],
        "max_body_rate_norm_rad_s": _finite_extreme(np.linalg.norm(rate, axis=1), np.max),
        "max_tilt_rad": _finite_extreme(tilt, np.max),
        "contact_force_max_N": max_contact_force,
        "collision_physics_samples": int(collisions.sum()),
        "collision_check_complete": bool(valid_physics_poses.all()),
        "collision_unchecked_physics_samples": int((~valid_physics_poses).sum()),
        "out_of_bounds": bool(outside.any()), "out_of_bounds_physics_samples": int(outside.sum()),
        "height_range_m": [_finite_extreme(valid_dynamics[:, 2], np.min), _finite_extreme(valid_dynamics[:, 2], np.max)],
        "motor_state_min": _finite_extreme(motor_states, np.min), "motor_state_max": _finite_extreme(motor_states, np.max),
        "planning_attempts": len(plans), "planning_successes": sum(p["success"] for p in plans),
        "planning_failures": sum(not p["success"] for p in plans),
        "trajectory_handoffs": sum(p["handoff"]["accepted"] for p in plans),
        "cold_start_commits": sum(p["handoff"]["accepted"] and p["operation"] == "plan_from_rest" for p in plans),
        "replan_commits": sum(p["handoff"]["accepted"] and p["operation"] == "replan_once" for p in plans),
        "plans_without_commit": sum(not p["handoff"]["accepted"] for p in plans),
        "planning_p50_s": float(np.median(durations)) if len(durations) else None,
        "planning_p95_s": float(np.quantile(durations, .95)) if len(durations) else None,
        "planning_max_s": float(durations.max()) if len(durations) else None,
        "physics_hz": 400, "tracking_hz": 100,
        "finite": bool(finite),
    }


def _intervals(mask, times):
    indices = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)))
    dt = float(np.median(np.diff(times))) if len(times) > 1 else .01
    return [(int(start), int(end), float(times[start] - dt), float(times[end - 1]))
            for start, end in indices.reshape(-1, 2)]


def summarize_timeline(arrays, plans, goal):
    """Locate intermediate slowdowns without using them as trajectory acceptance gates."""
    times, states, references = arrays["time"], arrays["states"], arrays["references"]
    actual_speed = np.linalg.norm(states[:, 7:10], axis=1)
    reference_speed = np.linalg.norm(references[:, :, 1], axis=1)
    goal_distance = np.linalg.norm(states[:, :3] - goal, axis=1)
    moved = np.flatnonzero(actual_speed > .5)
    first_motion = float(times[moved[0]]) if len(moved) else None
    intermediate = ((goal_distance > 1.) & (times > first_motion)) if len(moved) else np.zeros(len(times), bool)
    episodes = []
    for start, end, t0, t1 in _intervals(intermediate & (actual_speed < .2), times):
        if t1 - t0 < .2 - 1e-9:
            continue
        concurrent = [p for p in plans if p["t"] <= t1 and p["result_t"] >= t0]
        prior = [p for p in plans if p["handoff"]["accepted"] and p["activation_t"] <= t0]
        episodes.append({
            "start_s": t0, "end_s": t1, "duration_s": t1 - t0,
            "actual_speed_mean_m_s": float(np.mean(actual_speed[start:end])),
            "reference_speed_mean_m_s": float(np.mean(reference_speed[start:end])),
            "reference_speed_max_m_s": float(np.max(reference_speed[start:end])),
            "goal_distance_min_m": float(np.min(goal_distance[start:end])),
            "on_backup_samples": int(np.count_nonzero(arrays["on_backup"][start:end])),
            "fsm_states": np.unique(arrays["fsm_state"][start:end]).tolist(),
            "last_committed_plan": prior[-1]["id"] if prior else None,
            "concurrent_plans": [{"id": p["id"], "status": p["status"], "t": p["t"],
                                  "result_t": p["result_t"], "worker_s": p["worker_s"]}
                                 for p in concurrent],
        })
    backup_intervals = [{"start_s": t0, "end_s": t1, "duration_s": t1 - t0,
                         "start_position_m": states[start, :3].tolist(),
                         "end_position_m": states[end - 1, :3].tolist()}
                        for start, end, t0, t1 in _intervals(arrays["on_backup"], times)]
    modules = {}
    for key in sorted({key for plan in plans for key in plan["timings"]}):
        values = np.asarray([p["timings"][key] for p in plans if key in p["timings"]])
        modules[key] = {"count": len(values), "p50_s": float(np.median(values)),
                        "p95_s": float(np.quantile(values, .95)), "max_s": float(np.max(values))}
    jumps = np.asarray([p["handoff"]["reference_jump"] for p in plans if p["handoff"]["accepted"]])
    replan_jumps = np.asarray([p["handoff"]["reference_jump"] for p in plans
                              if p["handoff"]["accepted"] and p["operation"] == "replan_once"])
    motion_mask = times >= first_motion if first_motion is not None else np.zeros(len(times), bool)
    return {
        "diagnostic_definition": {"slow_speed_m_s": .2, "minimum_interval_s": .2,
                                  "minimum_goal_distance_m": 1., "motion_onset_speed_m_s": .5},
        "first_motion_s": first_motion, "intermediate_slow_intervals": episodes,
        "intermediate_slow_total_s": sum(item["duration_s"] for item in episodes),
        "backup_entries": len(backup_intervals), "backup_intervals": backup_intervals,
        "committed_with_backup": sum(p["handoff"].get("backup_available", False) for p in plans),
        "backup_status_counts": dict(Counter(p["backup_status"] for p in plans)),
        "planning_status_counts": dict(Counter(p["status"] for p in plans)),
        "module_timings": modules,
        "reference_jump_max_pvaj": jumps.max(axis=0).tolist() if len(jumps) else None,
        "replan_reference_jump_max_pvaj": replan_jumps.max(axis=0).tolist() if len(replan_jumps) else None,
        "actual_speed_p50_m_s": float(np.median(actual_speed)),
        "actual_speed_p95_m_s": float(np.quantile(actual_speed, .95)),
        "reference_speed_p50_m_s": float(np.median(reference_speed)),
        "reference_speed_p95_m_s": float(np.quantile(reference_speed, .95)),
        "motion_to_end_actual_speed_p50_m_s": float(np.median(actual_speed[motion_mask])) if motion_mask.any() else None,
        "motion_to_end_actual_speed_p95_m_s": float(np.quantile(actual_speed[motion_mask], .95)) if motion_mask.any() else None,
        "motion_to_end_reference_speed_p50_m_s": float(np.median(reference_speed[motion_mask])) if motion_mask.any() else None,
        "motion_to_end_reference_speed_p95_m_s": float(np.quantile(reference_speed[motion_mask], .95)) if motion_mask.any() else None,
    }
