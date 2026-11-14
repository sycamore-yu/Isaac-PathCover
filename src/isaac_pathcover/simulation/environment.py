"""Isaac Lab 3.0 Newton/MJWarp plant with the extracted L2F motor equations."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

from isaac_pathcover.control.actuator import RotorActuator
from isaac_pathcover.dynamics import load_params


def make_scene_cfg(scene):
    params = load_params()
    fields = {
        "__module__": __name__,
        "__annotations__": {"robot": RigidObjectCfg, "contact": ContactSensorCfg},
        "robot": RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.CuboidCfg(
                size=tuple(scene.body_size),
                rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
                collision_props=sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True),
                mass_props=sim_utils.MassCfg(mass=float(params.mass[0])),
                activate_contact_sensors=True,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.1, .3, .6)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(scene.start), rot=(1., 0., 0., 0.)),
        ),
        "contact": ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot", update_period=0., history_length=4),
    }
    for index, (lower, upper) in enumerate(zip(scene.box_lower, scene.box_upper)):
        name = f"obstacle_{index:04d}"
        fields["__annotations__"][name] = AssetBaseCfg
        fields[name] = AssetBaseCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{name}",
            spawn=sim_utils.CuboidCfg(
                size=tuple(upper - lower),
                collision_props=sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.45, .55, .45)),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=tuple((lower + upper) / 2)),
        )
    cls = configclass(type("FlightSceneCfg", (InteractiveSceneCfg,), fields))
    return cls(num_envs=1, env_spacing=50., replicate_physics=True, clone_in_fabric=True)


@configclass
class FlightEnvCfg(DirectRLEnvCfg):
    seed = 0
    decimation = 4
    episode_length_s = 300.
    action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
    observation_space = 13
    state_space = 0
    scene: InteractiveSceneCfg = None
    start: tuple = (0., 0., 1.)
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=.0025, render_interval=4, gravity=(0., 0., -9.8066),
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(njmax=128, nconmax=32, cone="pyramidal", impratio=1, integrator="implicitfast"),
            num_substeps=1, debug_mode=False, use_cuda_graph=True,
        ),
    )


def make_env_cfg(scene, device="cuda:0", seed=0):
    cfg = FlightEnvCfg()
    cfg.scene = make_scene_cfg(scene)
    cfg.start = tuple(scene.start)
    cfg.seed = seed
    cfg.sim.device = device
    return cfg


class FlightEnv(DirectRLEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.robot = self.scene["robot"]
        self.params = load_params(device=self.device, num_envs=self.num_envs)
        self.actuator = RotorActuator(self.params)
        self.robot.set_masses_index(masses=self.params.mass[:, None].contiguous())
        self.robot.set_inertias_index(inertias=torch.linalg.inv(self.params.J_inv).reshape(self.num_envs, 1, 9))
        self.sim.physics_manager.set_decimation(1)
        self._physics_handles_decimation = False
        self.command = torch.zeros((self.num_envs, 4), device=self.device)
        self.command[:, 0] = -self.params.mass * self.params.gravity[:, 2]
        self.wrench_updates = 0
        self.substep_states = []

    def physical_state(self):
        data = self.robot.data
        return torch.cat((data.root_com_pos_w.torch - self.scene.env_origins,
                          data.root_link_quat_w.torch[:, [3, 0, 1, 2]],
                          data.root_com_lin_vel_w.torch, data.root_com_ang_vel_b.torch), dim=-1)

    def _pre_physics_step(self, actions):
        self.command.copy_(actions)
        self.actuator.command(self.command)
        self.substep_states = []

    def _apply_action(self):
        force, torque = self.actuator.step(self.physics_dt)
        self.robot.permanent_wrench_composer.set_forces_and_torques_index(
            forces=force[:, None, :], torques=torque[:, None, :], body_ids=[0], is_global=False,
        )
        self.substep_states.append(self.physical_state()[0].clone())
        self.wrench_updates += 1

    def _get_observations(self):
        return {"policy": self.physical_state()}

    def _get_rewards(self):
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self):
        # The flight runner records the terminal state before resetting.
        done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return done, done.clone()

    def contact_force(self):
        return self.scene["contact"].data.net_forces_w_history.torch.norm(dim=-1).amax()

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self.actuator.reset(ids)
        self.command[ids] = 0.
        self.command[ids, 0] = -self.params.mass[ids] * self.params.gravity[ids, 2]
        pose = torch.zeros((len(ids), 7), device=self.device)
        pose[:, :3] = torch.tensor(self.cfg.start, device=self.device) + self.scene.env_origins[ids]
        pose[:, 6] = 1.
        self.robot.write_root_com_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        self.robot.write_root_com_velocity_to_sim_index(
            root_velocity=torch.zeros((len(ids), 6), device=self.device), env_ids=env_ids,
        )
