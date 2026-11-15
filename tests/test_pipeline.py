"""Complete planning flow and the failure/cache contract used by Newton."""

import numpy as np
import pytest

from isaac_pathcover.planning.geometry import box_bounds
from isaac_pathcover.planning.pipeline import Planner


class SmallScene:
    lower = np.array([-3.0, -2.0, 0.0])
    upper = np.array([3.0, 2.0, 2.5])
    start = np.array([-2.0, -0.5, 1.0])
    goal = np.array([2.0, -0.5, 1.0])

    def point_cloud(self):
        return np.array([[x, y, z] for x in [-0.2, 0.2]
                         for y in [0.8, 1.2] for z in [0.5, 1.5]])

    def bounds_halfspaces(self):
        return box_bounds(self.lower, self.upper)


def initial_state(scene):
    result = np.zeros((3, 3))
    result[:, 0] = scene.start
    return result


def test_point_cloud_to_executable_minco_and_expected_search_failure():
    scene = SmallScene()
    planner = Planner(scene, seed=3)
    state = initial_state(scene)
    result = planner.plan(state)
    assert result.success, result.status
    trajectory = result.trajectory
    assert result.search.path.shape[1] == 3
    assert len(result.corridor.polytopes) >= 1
    assert result.optimization.success
    assert result.timings["total"] > 0.0
    for order in range(3):
        np.testing.assert_allclose(trajectory.evaluate(0.0, order), state[:, order], atol=2e-9)
    np.testing.assert_allclose(trajectory.evaluate(trajectory.duration), result.corridor.goal, atol=2e-9)
    # A failed new request returns no candidate and leaves the successful
    # result object usable by the flight loop's retain-old-trajectory rule.
    grid = planner.grid
    failed = planner.plan(state, goal=[4.0, 0.0, 1.0])
    assert not failed.success and failed.trajectory is None
    assert "SearchFailure" in failed.status
    assert planner.grid is grid
    assert failed.timings["inflation_s"] == failed.timings["grid_s"] == 0.0
    np.testing.assert_allclose(trajectory.evaluate(0.0), scene.start, atol=2e-9)


def test_map_fusion_invalidates_planning_data_only_for_changed_occupancy():
    scene = SmallScene()
    planner = Planner(scene)
    initial = planner.map.version
    hit = np.array([[2.5, 1.5, 1.0]])
    update = planner.update_map(hit[0] - [0.1, 0.0, 0.0], hit)
    assert update["occupancy_changed"]
    assert planner.map.version > initial
    assert planner.map.is_occupied(hit).all()
    assert update["fusion_s"] >= 0.0
    result = planner.plan(initial_state(scene), goal=[4.0, 0.0, 1.0])
    assert result.map_version == planner.map.version
    assert result.timings["grid_s"] > 0.0


def test_invalid_caller_state_is_not_hidden_as_planning_failure():
    with pytest.raises(ValueError, match="start_state"):
        Planner(SmallScene()).plan(np.zeros(3))


def test_replan_can_use_previous_trajectory_without_reusing_old_polytopes():
    scene = SmallScene()
    planner = Planner(scene, seed=3)
    previous = planner.plan(initial_state(scene))
    assert previous.success
    elapsed = min(0.1, previous.trajectory.duration / 4)
    measured = np.column_stack([previous.trajectory.evaluate(elapsed, derivative=k)
                                for k in range(3)])
    result = planner.plan(measured, guide=previous.trajectory, guide_elapsed=elapsed)
    assert result.success, result.status
    assert result.optimization.initialization in ("cold", "guide")
    np.testing.assert_allclose(result.trajectory.evaluate(0), measured[:, 0], atol=2e-9)


def test_receding_search_reuses_only_the_free_suffix():
    scene = SmallScene()
    planner = Planner(scene, seed=3)
    first = planner.plan(initial_state(scene))
    assert first.success
    state = initial_state(scene)
    state[:, 0] = (first.search.path[0] + first.search.path[1]) / 2
    second = planner.plan(state)
    assert second.timings.get("search_reused") == 1.0
    np.testing.assert_allclose(second.search.path[0], state[:, 0])
    np.testing.assert_allclose(second.search.path[-1], scene.goal)
    # A new goal must force a fresh search instead of using an obsolete tail.
    different = planner.plan(state, goal=[2.0, 0.5, 1.0])
    assert different.timings.get("search_reused") is None
