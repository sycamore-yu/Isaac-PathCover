import numpy as np
import pytest
import torch

from isaac_pathcover.control.actuator import RotorActuator
from isaac_pathcover.control.rate import BodyRateController
from isaac_pathcover.dynamics import load_params, rotor_thrust_mean, motor_step_closed_form
from isaac_pathcover.simulation.scene import Scene


def test_hover_wrench_preserves_motor_equilibrium():
    params = load_params(dtype=torch.float64)
    actuator = RotorActuator(params)
    command = torch.tensor([[.752 * 9.8066, 0., 0., 0.]], dtype=torch.float64)
    actuator.command(command)
    force, torque = actuator.step(.0025)
    torch.testing.assert_close(force[:, 2], command[:, 0], atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(torque, torch.zeros_like(torque), atol=1e-13, rtol=0)
    torch.testing.assert_close(actuator.motor_state, actuator.hover_state)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_recorded_zero_collective_preserves_torque_direction(dtype):
    # Medium seed2, recorded step ends 4.51 s and 4.97 s, before first contact.
    params = load_params(dtype=dtype, num_envs=2)
    actuator = RotorActuator(params)
    wrench = torch.tensor([
        [0., -.05256457254290581, .15454986691474915, -.033958569169044495],
        [3.4720371786534088e-6, .0964922234416008, -.12167281657457352, -.15397004783153534],
    ], dtype=dtype)
    actuator.command(wrench)
    assert (actuator.allocated_wrench[:, 1:] * wrench[:, 1:] > 0.).all()
    torch.testing.assert_close(actuator.allocated_wrench[:, 1:], wrench[:, 1:])
    assert (actuator.allocated_thrust >= actuator.lower).all()
    assert (actuator.allocated_thrust <= actuator.upper).all()


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_feasible_wrench_keeps_rotor_allocation(dtype):
    params = load_params(dtype=dtype, num_envs=2)
    actuator = RotorActuator(params)
    thrust = torch.tensor([[1., 2., 3., 4.], [8.5, 0., 1.5, 4.]], dtype=dtype)
    wrench = torch.einsum("nij,nj->ni", actuator.allocation, thrust)
    actuator.command(wrench)
    torch.testing.assert_close(actuator.allocated_thrust, thrust)
    torch.testing.assert_close(actuator.allocated_wrench, wrench)
    assert (actuator.allocated_thrust >= actuator.lower).all()
    assert (actuator.allocated_thrust <= actuator.upper).all()


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_infeasible_torque_scales_all_axes_together(dtype):
    params = load_params(dtype=dtype, num_envs=2)
    actuator = RotorActuator(params)
    wrench = torch.tensor([[34., 2., -1.5, .7], [0., -2., 3., -.9]], dtype=dtype)
    actuator.command(wrench)
    ratio = actuator.allocated_wrench[:, 1:] / wrench[:, 1:]
    assert ((ratio > 0.) & (ratio < 1.)).all()
    torch.testing.assert_close(ratio, ratio[:, :1].expand_as(ratio))
    assert (actuator.allocated_thrust >= actuator.lower).all()
    assert (actuator.allocated_thrust <= actuator.upper).all()


def test_rate_damping_uses_measured_motion_and_resets():
    params = load_params(dtype=torch.float64)
    actuator = RotorActuator(params)
    controller = BodyRateController(params, actuator.upper.sum(-1))
    command = torch.tensor([[9.8066, 1., 0., 0.]], dtype=torch.float64)
    stationary = torch.zeros((1, 3), dtype=torch.float64)
    first = controller.wrench(command, stationary, .0025)
    faster_command = command.clone()
    faster_command[:, 1] = 2.
    faster = controller.wrench(faster_command, stationary, .0025)
    torch.testing.assert_close(faster[:, 1:], 2 * first[:, 1:])
    moving = torch.tensor([[.05, 0., 0.]], dtype=torch.float64)
    damped = controller.wrench(command, moving, .0025)
    assert 0. < damped[0, 1] < first[0, 1] / 2
    controller.reset()
    torch.testing.assert_close(controller.wrench(command, stationary, .0025), first)


def test_motor_impulse_matches_quadrature():
    params = load_params(dtype=torch.float64)
    state = torch.tensor([[.2, .8, .3, .7]], dtype=torch.float64)
    target = torch.tensor([[.8, .2, .9, .1]], dtype=torch.float64)
    dt = .02
    time = torch.linspace(0., dt, 20001, dtype=torch.float64)[:, None, None]
    speed = target + (state - target) * torch.exp(-time / .033)
    expected = torch.trapezoid(8.5 * speed.square(), x=time[:, 0, 0], dim=0) / dt
    torch.testing.assert_close(rotor_thrust_mean(params, state, target, dt), expected, atol=2e-9, rtol=1e-9)
    torch.testing.assert_close(motor_step_closed_form(params, state, target, dt), speed[-1])


def test_exact_existing_scenario_assets_and_lidar():
    for name, count in [("medium", 59870), ("dense", 103876)]:
        scene = Scene(name)
        assert int(scene.occupied.sum()) == count
        assert not scene.contains_obstacle(np.array([scene.start, scene.goal])).any()
        origin, cloud = scene.lidar(scene.start, [1., 0., 0., 0.], horizontal_rays=32)
        np.testing.assert_allclose(origin, scene.start + [0., 0., .25])
        assert len(cloud) > 0
        assert scene.contains_obstacle(cloud).all()
