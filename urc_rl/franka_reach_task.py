import torch
import math
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat


class FrankaReachTask:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        self.num_envs = num_envs
        # self.num_obs = obs_cfg["num_obs"]
        # self.num_actions = env_cfg["num_actions"]
        # self.num_commands = command_cfg["num_commands"]

        self.num_obs = 34 # actions (9), dofs_pos (9), dofs_vel (9) gripper_pos (3), goal_pos (3), distance_to_goal (1), together num of observations is 9 + 9 + 9 + 3 + 3 + 1 = 34
        self.num_actions = 9  # 7 dof + 2 gripper fingers
        self.num_goal_dim = 3  # goal position in 3D space
        self.device = gs.device
        self.joint_names = [
            'joint1',
            'joint2',
            'joint3',
            'joint4',
            'joint5',
            'joint6',
            'joint7',
            'finger_joint1',
            'finger_joint2',
        ]

        # TODO simulate_action_latency is not implemented yet
        # self.simulate_action_latency = True  # there is a 1 step latency on real robot
        self.dt = 0.02  # control frequency on real robot is 50hz
        # self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)
        self.max_episode_length = math.ceil(10.0 / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        # initialize buffers
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=gs.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.dof_pos = torch.zeros_like(self.actions, device=gs.device, dtype=gs.tc_float)
        self.extras = dict()  # extra information for logging
        self.extras["observations"] = dict()
        self.extras["log"] = dict()

        # self.obs_scales = obs_cfg["obs_scales"]
        # self.reward_scales = reward_cfg["reward_scales"]

        # create franka scene
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
            show_viewer=show_viewer,
        )

        # add plane
        self.plane = self.scene.add_entity(
            gs.morphs.Plane(),
        )

        # add robot
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(file="../assets/xml/franka_emika_panda/panda.xml")
        )

        self.dofs_idx = [self.robot.get_joint(name).dof_idx_local for name in self.joint_names]
        print(f"[DEBUG] dofs_idx: {self.dofs_idx}")  # dofs_idx: [0, 1, 2, 3, 4, 5, 6, 7, 8]


        # goal visualization
        self.goal = self.scene.add_entity(
            gs.morphs.Sphere(radius=0.05, fixed=True, visualization=True, pos=(0.5, 0, 0.3))
        )

        self.scene.build(n_envs=self.num_envs, env_spacing=(1.0, 1.0))
        self.envs_idx = torch.arange(self.num_envs, device=self.device)

        self.reset_buf = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device)
        self.goal_pos = torch.zeros((self.num_envs, 3), device=self.device)

        self._init_robot()
        self._sample_goal_positions()

    # def _init_robot(self):
    #     qpos = torch.tensor(
    #         [
    #             -0.5, 0.5, 0.0, -1.5, 0.0, 1.5, 0.5, 0.04, 0.04
    #         ],
    #         device=self.device).repeat(self.num_envs, 1)
    #     self.robot.set_qpos(qpos, envs_idx=self.envs_idx)
    #     print(f"[DEBUG] shape of dofs_pos is {self.robot.get_dofs_position().shape}, expected {(self.num_envs, 9)}")  # [N, 9]
    #     print(f"[DEBUG] shape of dofs_vel is {self.robot.get_dofs_velocity().shape}, expected {(self.num_envs, 9)}")  # [N, 9]
    #     print(f"[DEBUG] dofs limits: {self.robot.get_dofs_limit()}")

    def _normal_sample_from_limits(self, num_envs: int) -> torch.Tensor:
        """
        Sample joint positions from a truncated normal distribution centered at the midpoint
        of the robot's joint limits. Two standard deviations cover the full joint range.

        Args:
            num_envs (int): number of environments to sample for

        Returns:
            Tensor: sampled joint positions, shape [num_envs, num_dofs]
        """
        lower, upper = self.robot.get_dofs_limit()  # [num_dofs], [num_dofs]
        mid = (lower + upper) / 2
        half_range = (upper - lower) / 2
        std = half_range / 2  # 2σ = full range

        samples = torch.normal(
            mean=mid.expand(num_envs, -1),
            std=std.expand(num_envs, -1)
        )
        return torch.clamp(samples, lower, upper)

    def _init_robot(self, envs_idx=None):
        """
        Initialize the robot's joint positions.
        If envs_idx is None, it initializes for all environments.
        """
        if envs_idx is None:
            envs_idx = self.envs_idx
        # qpos = torch.tensor(
        #     [-0.5, 0.5, 0.0, -1.5, 0.0, 1.5, 0.5, 0.04, 0.04],
        #     device=self.device
        # ).repeat(envs_idx.shape[0], 1)
        qpos = self._normal_sample_from_limits(envs_idx.shape[0])
        self.robot.set_qpos(qpos, envs_idx=envs_idx)

        # dofs limits:
        # (tensor([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973,  0.0000, 0.0000], device='cuda:0'),
        #  tensor([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973,  0.0400, 0.0400], device='cuda:0'))


    # def _sample_circle(self, radius_range: tuple[float, float] = (0.2, 0.7), z_range: tuple[float, float]=(0.0, 1.0), num_points: int=100):
    #     # first sample the radius uniformly from the given range
    #     radius = torch.rand(num_points, device=self.device) * (radius_range[1] - radius_range[0]) + radius_range[0]
        
    #     # sample N points on a circle of given radius
    #     angles = torch.linspace(0, 2 * math.pi, num_points, device=self.device)
    #     x = radius * torch.cos(angles)
    #     y = radius * torch.sin(angles)

    #     # sample z values within the specified range
    #     z_min, z_max = z_range
    #     z = z_min + (z_max - z_min) * torch.rand(num_points, device=self.device)
        
    #     stacked_points = torch.stack((x, y, z), dim=1)

    #     # reschuffle points
    #     indices = torch.randperm(num_points, device=self.device)
    #     shuffled_points = stacked_points[indices]

    #     return shuffled_points 

    # def _sample_hemisphere(
    #     self,
    #     radius_range=(0.2, 0.7),
    #     num_points=100
    # ):
    #     r = torch.empty(num_points, device=self.device).uniform_(radius_range[0], radius_range[1])
    #     phi = torch.empty(num_points, device=self.device).uniform_(0, math.pi * 2)  # azimuth
    #     theta = torch.empty(num_points, device=self.device).uniform_(0, math.pi / 2)

    #     x = r * torch.sin(theta) * torch.cos(phi)
    #     y = r * torch.sin(theta) * torch.sin(phi)
    #     z = r * torch.cos(theta)  # bias up to avoid below-table goals

    #     offset = torch.tensor([0.0, 0.0, 0.1], device=self.device)  # offset above table
    #     goal_pos = torch.stack((x, y, z), dim=1) + offset

    #     return goal_pos

    # cosine correction
    def _sample_hemisphere(self, num_points=1000, radius_range=(0.2, 0.7)):
        phi = torch.rand(num_points, device=self.device) * 2 * math.pi
        z = torch.rand(num_points, device=self.device)  # z ∈ [0, 1]
        radius = torch.empty(num_points, device=self.device).uniform_(radius_range[0], radius_range[1])
        theta = torch.acos(z)  # corrected inclination

        x = torch.sin(theta) * torch.cos(phi)
        y = torch.sin(theta) * torch.sin(phi)
        z = torch.cos(theta)

        points = torch.stack([x, y, z], dim=1) *  radius[:, None]
        return points
    
    def _sample_goal_positions(self):

        # self.goal_pos = self._sample_circle(radius_range=(0.2, 0.7), z_range=(0.0, 1.0), num_points=self.num_envs)
        self.goal_pos = self._sample_hemisphere(num_points=self.num_envs, radius_range=(0.5, 0.7))
        self.goal.set_pos(self.goal_pos, envs_idx=self.envs_idx)

    def reached_goal(self):
        """
        Check if the gripper has reached the goal position.
        Returns a boolean tensor of shape (num_envs,).
        """
        gripper_pos = self.robot.get_link("hand").get_pos(envs_idx=self.envs_idx)
        distance_to_goal = torch.norm(gripper_pos - self.goal_pos, dim=1)
        reached = distance_to_goal < 0.1  # threshold for reaching the goal
        return reached

    def joint_centering_penalty(self, scale=-0.1):
        lower, upper = self.robot.get_dofs_limit()
        midpoint = (lower + upper) / 2
        range_ = (upper - lower) / 2
        penalty = torch.norm((self.dof_pos - midpoint) / range_, dim=1)  # [N,], positive
        return scale * penalty

    def action_penalty(self, scale=-0.01):
        action_penalty = torch.sum(self.actions ** 2, dim=1)  # [N,]
        # clamp to avoid large jumps in reward
        action_penalty = torch.clamp(action_penalty, min=0.0, max= 1.0)  # [N,]
        return scale * action_penalty

    def distance_to_goal_penalty(self, scale=-1.0):
        """
        Compute the distance to goal penalty.
        Returns a tensor of shape (num_envs,).
        """
        d_goal = self.extras["observations"]["distance_to_goal"].squeeze(1)  # [N,]
        # clamp to avoid jumps in reward
        d_goal = torch.clamp(d_goal, min=0.0, max=1.0)  # [N,]
        # scale the distance to goal to be in the
        return scale * d_goal
    
    def contact_penalty(self, scale=-0.1):
                # Get contact info from robot
        contact_info = self.robot.get_contacts()
        valid_mask = contact_info.get("valid_mask", None)

        # Count per-env contacts
        if valid_mask is not None:
            contact_count = valid_mask.sum(dim=1).float()
        else:
            contact_count = torch.tensor(
                [len(contact_info["position"])] * self.num_envs,
                dtype=torch.float32,
                device=self.device
            )

        # Optional logging
        penalty = scale * contact_count  # [N,]
        # clamp to avoid large jumps in reward
        penalty = torch.clamp(penalty, min=0.0, max=1.0)  # [N,]

        self.extras["log"]["obs/contact_count"] = contact_count
        self.extras["log"]["rewards/contact_penalty"] = penalty

        return penalty

    def get_reward(self):
        """
        Compute the reward for the current step.
        Returns a tensor of shape (num_envs,).
        """
        distance_penalty = self.distance_to_goal_penalty(scale=-1.0)  # distance to goal penalty
        joint_centering_penalty = self.joint_centering_penalty(scale=-0.1 * self.dt)  # joint centering penalty
        action_penalty = self.action_penalty(scale=-0.01 * self.dt)  # action penalty
        reached_goal = self.reached_goal()  # [N,]

        self.extras["log"] = {
            "rewards/distance_to_goal": distance_penalty,
            "rewards/joint_centering_penalty": joint_centering_penalty,
            "rewards/action_penalty": action_penalty,
            "rewards/reached_goal": reached_goal,
        }
        
        reward = distance_penalty
        reward += joint_centering_penalty
        reward += action_penalty

        reward[reached_goal] += 1.0  # give a bonus for reaching

        self.extras["log"]["rewards/total"] = reward

        return reward
    
        
    def reset(self):
        envs_to_reset = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        # reset only the environments that are done
        if envs_to_reset.numel() > 0:
            self._init_robot(envs_to_reset)
            new_goals = self._sample_hemisphere(num_points=envs_to_reset.shape[0])
            self.goal_pos[envs_to_reset] = new_goals
            self.goal.set_pos(new_goals, envs_idx=envs_to_reset)
            self.episode_length_buf[envs_to_reset] = 0
            self.reset_buf[envs_to_reset] = False

        return self.obs_buf, None
    
    def force_reset(self):
        """
        Force reset of all environments, regardless of done status.
        """
        envs_to_reset = self.envs_idx  # all env indices

        self._init_robot(envs_to_reset)
        new_goals = self._sample_hemisphere(num_points=envs_to_reset.shape[0])
        self.goal_pos[envs_to_reset] = new_goals
        self.goal.set_pos(new_goals, envs_idx=envs_to_reset)
        self.episode_length_buf[envs_to_reset] = 0
        self.reset_buf[envs_to_reset] = False

        return self.obs_buf, None

    def get_dones(self):
        """
        Compute the done condition for the current step.
        Returns a tensor of shape (num_envs,).
        """
        reached_goal = self.reached_goal()
        dones = self.episode_length_buf >= self.max_episode_length  # [N,]
        return dones | reached_goal  # return True if either condition is met
    
    def act(self, actions):
        self.actions = actions.to(self.device)
        current_q = self.robot.get_dofs_position(envs_idx=self.envs_idx)
        target_q = current_q + 0.05 * torch.tanh(actions)

        # sets the PD targets
        self.robot.control_dofs_position(target_q, envs_idx=self.envs_idx)

    def step(self, actions):

        ### Act (set the PD controller targets)
        self.act(actions)

        ### Step the scene
        self.scene.step()

        ### Get observations and reward
        self.obs_buf, self.extras = self.get_observations()
        self.rew_buf = self.get_reward()
        self.episode_length_buf += 1

        ### Check if any envs are done
        self.reset_buf = self.get_dones()
  
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def normalize_dof_vel(self, dofs_vel):
        # according to the franka documentation
        # https://www.generationrobots.com/media/panda-franka-emika-datasheet.pdf
        # the limits are roughly
        # convert limits to rad/s and m/s
        # originally in degrees/s and mm/s
        # joints 1–4: 150 degrees/s = 2.618 rad/s
        # joints 5–7: 180 degrees/s = 3.142 rad/s
        # fingers: 50 mm/s = 0.05 m/s
        max_vel = torch.tensor([
            2.618, 2.618, 2.618, 2.618,  # joints 1–4
            3.142, 3.142, 3.142,         # joints 5–7
            0.05, 0.05                  # fingers
        ], device=self.device)
        return torch.clamp(dofs_vel / max_vel, -1.0, 1.0)

    def normalize_dof_pos(self, dofs_pos):
        lower, upper = self.robot.get_dofs_limit()
        midpoint = (lower + upper) / 2
        half_range = (upper - lower) / 2
        return torch.clamp((dofs_pos - midpoint) / half_range, -1.0, 1.0)

    def get_observations(self):
        actions = self.actions  # [N, 9]\

        raw_dofs_pos = self.robot.get_dofs_position(envs_idx=self.envs_idx) # [N, 9]
        raw_dofs_vel = self.robot.get_dofs_velocity(envs_idx=self.envs_idx) # [N, 9]

        dofs_pos = self.normalize_dof_pos(raw_dofs_pos)  # normalize dofs_pos to [-1, 1]
        dofs_vel = self.normalize_dof_vel(raw_dofs_vel)  # normalize dofs_vel to [-1, 1]

        gripper_pos = self.robot.get_link("hand").get_pos()  # [N, 3]
        goal_pos = self.goal_pos # [N, 3]
        distance_to_goal = torch.norm(gripper_pos - goal_pos, dim=1, keepdim=True) # [N, 1]

        # normalize dofs_pos and dofs_vel

        self.obs_buf = torch.cat(
            [   
                actions,
                dofs_pos,
                dofs_vel,
                gripper_pos,
                goal_pos,
                distance_to_goal,
            ],
            axis=-1
        )
        self.extras["observations"]["critic"] = self.obs_buf
        self.extras["observations"]["actions"] = actions
        self.extras["observations"]["dofs_pos"] = dofs_pos
        self.extras["observations"]["dofs_vel"] = dofs_vel
        self.extras["observations"]["gripper_pos"] = gripper_pos
        self.extras["observations"]["goal_pos"] = goal_pos
        self.extras["observations"]["distance_to_goal"] = distance_to_goal

        self.extras["log"]["obs/distance_to_goal"] = distance_to_goal.squeeze(1)

        return self.obs_buf, self.extras

if __name__ == "__main__":
    gs.init(backend=gs.gpu, precision="32", logging_level='info')
    env = FrankaReachTask(num_envs=8, env_cfg={}, obs_cfg={}, reward_cfg={}, command_cfg={}, show_viewer=True)
    env.reset()

    for res in range(5):
        print(f"Resetting environment {res + 1}")
        env.force_reset()
        for _ in range(100):
            actions = torch.randn((env.num_envs, 9), device=env.device)
            obs, rew, done, extras = env.step(actions)
            # print everything in a formatted way
            # print(f"Obs: {obs}, Reward: {rew}, Done: {done}, Extras: {extras}")
            if done.any():
                env.reset()    