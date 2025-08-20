import numpy as np
import torch
import math
import genesis as gs
from PIL import Image
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sys
sys.path.append("/workspace/lerobot")
sys.path.append("/workspace/lerobot/src")
sys.path.append("/workspace/lerobot/src/lerobot")

try:
    from lerobot.src.lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy # type: ignore
except:
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


import logging

logging.basicConfig(
    level=logging.INFO,
)

logger = logging.getLogger("Env logger")


@dataclass
class CamConfig:
    lookat: tuple[int, int, int] = (0, 0, 0)
    pos: tuple[int, int, int] = (1, 1, 1)
    res: tuple[int, int] = (640, 360)
    fov: int = 60
    GUI: bool = True


@dataclass
class EnvConfig:
    urdf_path: str | Path
    cams: list[CamConfig]
    show: bool = True
    n_envs: int = 1
    action_scale: float = 1.0
    task: str = "Pick up the object."

    @classmethod
    def make_so_config(cls):
        cam_poss = [
            CamConfig(),
            CamConfig() # this one should be on the robot end effector
        ]
        return cls(
            urdf_path="/workspace/SO-100-arm/models/so_100_arm_5dof/so_100_arm_5dof.urdf",
            cams=cam_poss,
            show=True)

    @classmethod
    def make_franka_config(cls):
        cam_poss = [CamConfig(), CamConfig(pos=(-1,1,1))]
        return cls(urdf_path="../assets/xml/franka_emika_panda/panda.xml",
                   cams=cam_poss,
                   show=True)

    @classmethod
    def make_braccio_config(cls):
        cam_poss = [CamConfig()]
        return cls(urdf_path="/workspace/custom_scripts/braccio_description/urdf/braccio.urdf",
                   cams=cam_poss,
                   show=True)

    @classmethod
    def make_ur10_config(cls):
        cam_poss = [CamConfig()]
        return cls(urdf_path="/workspace/custom_scripts/urdf_files_dataset/urdf_files/random/robot-assets/ur10/ur10_robot.urdf",
                   cams=cam_poss,
                   show=True)

    @classmethod
    def make_external_panda_config(cls):
        cam_poss = [CamConfig()]
        return cls(urdf_path="/workspace/custom_scripts/urdf_files_dataset/urdf_files/random/robot-assets/franka_panda/panda.urdf",
                   cams=cam_poss,
                   show=True)

    @property
    def num_cams(self) -> int:
        return len(self.cams)


class ReachTask:
    def __init__(self, config: EnvConfig, policy: SmolVLAPolicy | None | Any = None):
        if not hasattr(gs, "device"):
            logging.warning("Initializing Genesis on device %s & precision %s", gs.gpu, "f32") # type: ignore
            gs.init(backend=gs.gpu, precision="32") # type: ignore
        self.config = config
        self.policy = policy
        self.num_envs = config.n_envs

        self.num_obs = -1
        self.num_actions = -1  # 7 dof + 2 gripper fingers
        self.num_goal_dim = 3  # goal position in 3D space
        self.device = gs.device

        self.dt = 1.00 / 60.00
        self.max_episode_length = math.ceil(10.0 / self.dt)

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(self.num_envs))),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=config.show,
        )
        self.plane = self.scene.add_entity(
            gs.morphs.Plane(),
        )
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(file=self.config.urdf_path)
        )
        self.goal = self.scene.add_entity(
            gs.morphs.Sphere(radius=0.05, fixed=True, visualization=True, collision=False, pos=(0.5, 0, 0.3))
        )
        self.cams = [self.scene.add_camera(
            **conf.__dict__
        ) for conf in self.config.cams]

        self.scene.build(n_envs=self.num_envs, env_spacing=(1.0, 1.0))

        link_names = [link.name for link in self.robot.links] # type: ignore
        joint_names = [link.name for link in self.robot.joints] # type: ignore
        logger.info(
            "LINK NAMES %s NUM LINKS %s JOINT NAMES %s "
            "NUM JOINTS %s NUM JOINTS %s NUM QS %s NUM DOFS %s",
            link_names, len(link_names), joint_names,
            len(joint_names), self.robot.n_joints, # type: ignore
            self.robot.n_qs, self.robot.n_dofs) # type: ignore
        assert self.robot.n_joints == self.robot.n_qs == self.robot.n_dofs, f"{self.robot.n_joints}, {self.robot.n_qs}, {self.robot.n_dofs}" # type: ignore

        self.robot.set_qpos(torch.rand(self.num_envs, self.robot.n_qs)) # type: ignore

        # robot observes its own joints
        self.num_actions = self.robot.n_dofs # type: ignore
        self.num_obs = self.robot.n_dofs # type: ignore
        self.action_scale = config.action_scale

        # initialize buffers
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.dof_pos = torch.zeros_like(self.actions, device=self.device, dtype=gs.tc_float)
        self.dof_vel = torch.zeros_like(self.actions, device=self.device, dtype=gs.tc_float)
        self.dof_targets = torch.zeros_like(self.actions, device=self.device, dtype=gs.tc_float)
        self.dof_lower_limits, self.dof_upper_limits = self.robot.get_dofs_limit() # type: ignore

        self.envs_idx = torch.arange(self.num_envs, device=self.device) # tensor([0, 1, 2, ..., num_envs])
        self.reset_buf = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        # speed scales for the robot's dofs
        self.dof_speed_scales = torch.ones((self.num_envs, self.robot.n_qs), device=self.device) # type: ignore

        self._init_robot()
        self.observations = self._get_observations()

    def render(self) -> Any:
        for cam in self.cams:
            cam.render(rgb=True)

    def _init_robot(self, envs_idx=None):
        """
        Initialize the robot's joint positions.
        If envs_idx is None, it initializes for all environments.
        """
        if envs_idx is None:
            envs_idx = self.envs_idx  # tensor([0, 1, 2, ..., num_envs])

        self.robot.set_qpos(self.dof_targets[envs_idx], envs_idx=envs_idx) # type: ignore

    def forbidden_contact(self):
        """Mask to reset envs that have collided with anything."""
        # get contact info from the robot
        contact_info = self.robot.get_contacts() # type: ignore
        valid_mask = contact_info.get("valid_mask", None)

        envs_with_contacts = valid_mask.any(dim=1)
        return envs_with_contacts

    def force_reset(self):
        """
        Force reset of all environments, regardless of done status.
        """
        envs_to_reset = self.envs_idx  # all env indices

        self._init_robot(envs_to_reset)
        self.episode_length_buf[envs_to_reset] = 0
        self.reset_buf[envs_to_reset] = False

        return self.observations, None

    def step(self, actions: torch.Tensor):
        ### Act (set the PD controller targets)
        self._act(actions)

        ### Step the scene
        self.scene.step()

        ### Get observations and reward
        self.observations = self._get_observations()
        self.episode_length_buf += 1

        ### Check if any envs are done
        # reset the environments that are done
        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).reshape((-1,)))

        return self.observations, self.rew_buf, self.reset_buf

    def _act(self, actions: torch.Tensor):
        actions = actions.to(self.device)
        self.actions = actions.clone().clamp(-1.0, 1.0)
        logger.info("robot dof: %s actions: %s", self.dof_speed_scales.shape[1], self.actions.shape[1])
        target_q = self.dof_targets + self.dof_speed_scales * self.dt * self.actions * self.action_scale
        self.dof_targets[:] = torch.clamp(target_q, self.dof_lower_limits, self.dof_upper_limits)
        self.robot.control_dofs_position(self.dof_targets, envs_idx=self.envs_idx) # type: ignore

    def _get_observations(self) -> dict[str, str | torch.Tensor]:
        obs_dof: gs.Tensor = self._get_dof_observations()
        assert self.policy
        images: list[torch.Tensor] = [ torch.from_numpy(np.ascontiguousarray(cam.render(rgb=True)[0])).permute(2, 0, 1).unsqueeze(0).to(device=gs.device, dtype=torch.float32).div_(255.0) for cam in self.cams]
        observations: dict[str, str | torch.Tensor] = dict(zip(self.policy.config.input_features.keys(), [obs_dof, *images]))
        observations["task"] = self.config.task
        return observations # type: ignore

    def _get_dof_observations(self) -> gs.Tensor:
        dofs_pos: gs.Tensor = self.robot.get_dofs_position(envs_idx=self.envs_idx) # type: ignore
        obs_dof = dofs_pos.clone() # type: ignore
        obs_dof: gs.Tensor
        assert obs_dof.shape[1] == self.num_obs,(
            f"Observation buffer shape mismatch: expected {self.num_obs}, got {obs_dof.shape[1]}")
        return obs_dof

    def reset_idx(self, envs_to_reset):
        """
        Reset the environments specified by envs_to_reset.
        This is a helper function to reset specific environments.
        """
        if envs_to_reset.numel() == 0:
            return

        self.robot.zero_all_dofs_velocity(envs_to_reset) # type: ignore
        self._init_robot(envs_idx=envs_to_reset)
        self.episode_length_buf[envs_to_reset] = 0
        self.reset_buf[envs_to_reset] = False

    def reset(self):
        self.reset_buf[:] = False
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        return self.observations, None

if __name__ == "__main__":
    from argparse import ArgumentParser
    gs.init(backend=gs.gpu, precision="32", logging_level='info') # type: ignore
    # URDF: git@github.com:brukg/SO-100-arm.git
    # env_config = EnvConfig.make_so_config()
    env_config = EnvConfig.make_franka_config()
    # env_config = EnvConfig.make_braccio_config()
    # env_config = EnvConfig.make_ur10_config()
    # env_config = EnvConfig.make_external_panda_config()
    # smolvla_path = "Hartvi/smolvla"
    smolvla_path = "Calvert0921/smolvla_franka_liftcube_200"

    policy = SmolVLAPolicy.from_pretrained(smolvla_path)
    policy.eval()

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    env = ReachTask(env_config, policy)
    env.reset()

    for res in range(5):
        print(f"Resetting environment {res + 1}")
        observations, _ = env.force_reset()
        for _ in range(500):
            action = torch.randn((env.num_envs, env.robot.n_qs), device=env.device) # type: ignore
            # input_features = dict(input_features=observations, image_features=observations)
            """
            ERROR:genesis:ValueError: All image features are missing from the batch. At least one expected.
            (batch: dict_keys(['input_features']))
            (image_features:{'observation.images.laptop': PolicyFeature(type=<FeatureType.VISUAL: 'VISUAL'>, shape=(3, 480, 640)), 'observation.images.phone': PolicyFeature(type=<FeatureType.VISUAL: 'VISUAL'>, shape=(3, 480, 640))})
            """
            print(observations.keys())
            action: torch.Tensor = env.policy.select_action(observations) # type: ignore
            observations, rew, done = env.step(action)
            # env.render()
            # if _ % 10 == 0:
            #     im = Image.fromarray(env.render())
            #     with open(f"im_{_}.png", "wb") as f:
            #         im.save(f, format="png")
            if done.any():
                print(f"\n\nResetting naturally.\n\n")
                env.reset()
