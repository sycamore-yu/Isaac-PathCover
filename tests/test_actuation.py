import numpy as np
import torch

from isaac_pathcover.control.actuator import RotorActuator
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
