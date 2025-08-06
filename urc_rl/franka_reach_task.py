import torch
import math
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
from reward_functions import (
    action_penalty,
    joint_acc_penalty,
    joint_velocity_penalty,
    grasp_proximity_reward,
)


class FrankaReachTask:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        self.num_envs = num_envs
        # self.num_obs = obs_cfg["num_obs"]
        # self.num_actions = env_cfg["num_actions"]
        # self.num_commands = command_cfg["num_commands"]

        # self.num_obs = 34 # actions (9), dofs_pos (9), dofs_vel (9) gripper_pos (3), goal_pos (3), distance_to_goal (1), together num of observations is 9 + 9 + 9 + 3 + 3 + 1 = 34
        self.num_obs = 30
        self.num_actions = 9  # 7 dof + 2 gripper fingers
        self.num_goal_dim = 3  # goal position in 3D space
        self.device = gs.device
        self.joint_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
            "finger_joint1",
            "finger_joint2",
        ]

        # TODO simulate_action_latency is not implemented yet
        # self.simulate_action_latency = True  # there is a 1 step latency on real robot
        # self.dt = 0.02  # control frequency on real robot is 50hz
        self.dt = 1.00 / 60.00
        # self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)
        self.max_episode_length = math.ceil(10.0 / self.dt)

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
        # goal visualization
        self.goal = self.scene.add_entity(
            gs.morphs.Sphere(
                radius=0.05, fixed=True, visualization=True, collision=False, pos=(0.5, 0, 0.3)
            )
        )

        self.scene.build(n_envs=self.num_envs, env_spacing=(1.0, 1.0))

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg
        self.action_scale = self.env_cfg["action_scale"]

        # initialize buffers
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float
        )
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.actions = torch.zeros(
            (self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float
        )
        self.dof_pos = torch.zeros_like(self.actions, device=self.device, dtype=gs.tc_float)
        self.dof_vel = torch.zeros_like(self.actions, device=self.device, dtype=gs.tc_float)
        self.dof_targets = torch.zeros_like(self.actions, device=self.device, dtype=gs.tc_float)
        self.dof_lower_limits, self.dof_upper_limits = self.robot.get_dofs_limit()

        # init raw dof buffers
        self.raw_dofs_acc = torch.zeros((self.num_envs, 9), device=self.device, dtype=gs.tc_float)
        self.raw_dofs_vel = torch.zeros((self.num_envs, 9), device=self.device, dtype=gs.tc_float)
        self.raw_dofs_pos = torch.zeros((self.num_envs, 9), device=self.device, dtype=gs.tc_float)

        # init curriculum buffers
        self.goal_reach_threshold = 0.05
        self.curriculum_goal_reached_once = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        # self.curriculum_goal_success_rate_ema = 0.0
        # self.curriculum_goal_success_smoothing = 0.95
        self.curriculum_radius_from_ee = 0.1  # initial radius from end-effector for sampling
        self.steps_from_last_curriculum = 0

        self.extras = dict()  # extra information for logging
        self.extras["observations"] = dict()
        self.extras["log"] = dict()

        # self.obs_scales = obs_cfg["obs_scales"]
        # self.reward_scales = reward_cfg["reward_scales"]

        self.dofs_idx = [self.robot.get_joint(name).dof_idx_local for name in self.joint_names]
        print(f"[DEBUG] dofs_idx: {self.dofs_idx}")  # dofs_idx: [0, 1, 2, 3, 4, 5, 6, 7, 8]

        self.envs_idx = torch.arange(
            self.num_envs, device=self.device
        )  # tensor([0, 1, 2, ..., num_envs])

        self.reset_buf = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.goal_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)

        # speed scales for the robot's dofs
        self.dof_speed_scales = torch.ones((self.num_envs, 9), device=self.device)
        # set the last two dofs (fingers) to a lower speed scale
        self.dof_speed_scales[:, 7:] = 0.1  # fingers

        self._init_robot()
        self._sample_goal_positions()
        self.num_steps = 0

    # --------------0-----------0--------
    #               0           0
    #      |============================|
    #      |                            |
    #      |  Sampling / Normalization  |
    #      |                            |
    #      |============================|

    def normalize_dof_vel(self, dofs_vel):
        # according to the franka documentation
        # https://www.generationrobots.com/media/panda-franka-emika-datasheet.pdf
        # the limits are roughly
        # convert limits to rad/s and m/s
        # originally in degrees/s and mm/s
        # joints 1–4: 150 degrees/s = 2.618 rad/s
        # joints 5–7: 180 degrees/s = 3.142 rad/s
        # fingers: 50 mm/s = 0.05 m/s
        max_vel = torch.tensor(
            [
                2.618,
                2.618,
                2.618,
                2.618,  # joints 1–4
                3.142,
                3.142,
                3.142,  # joints 5–7
                0.05,
                0.05,  # fingers
            ],
            device=self.device,
        )
        return torch.clamp(dofs_vel / max_vel, -1.0, 1.0)

    def normalize_dof_pos(self, dofs_pos):

        midpoint = (self.dof_lower_limits + self.dof_upper_limits) / 2
        half_range = (self.dof_upper_limits - self.dof_lower_limits) / 2
        return torch.clamp((dofs_pos - midpoint) / half_range, -1.0, 1.0)

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

        samples = torch.normal(mean=mid.expand(num_envs, -1), std=std.expand(num_envs, -1))
        return torch.clamp(samples, lower, upper)

    def _normal_sample_from_default_pose(
        self, num_envs: int, std_scale: float = 0.1
    ) -> torch.Tensor:
        """
        Sample joint positions from a truncated normal distribution centered at a known default pose.
        Two standard deviations (controlled by `std_scale`) cover a small range around that pose.

        Args:
            num_envs (int): Number of environments to sample for.
            std_scale (float): Scaling factor for standard deviation (in radians or meters).

        Returns:
            Tensor: Sampled joint positions, shape [num_envs, num_dofs]
        """
        # default_pose = torch.tensor([
        #     -1.5,  0.0,  0.0,
        #     -1.5,  0.0,  1.5,
        #     0.5,  0.04, 0.04,
        # ], device=self.device)

        default_pose = torch.tensor(
            [0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853, 0.04, 0.04], device=self.device
        )

        # Use 2σ to define a tight range around each joint angle
        std = (
            torch.tensor(
                [
                    1.0,
                    0.2,
                    0.2,  # joints 1–3
                    0.2,
                    0.2,
                    0.2,  # joints 4–6
                    0.2,
                    0.005,
                    0.005,  # joint7, fingers
                ],
                device=self.device,
            )
            * std_scale
        )  # scaled

        samples = torch.normal(mean=default_pose.expand(num_envs, -1), std=std.expand(num_envs, -1))

        # Clamp to joint limits
        lower, upper = self.robot.get_dofs_limit()
        return torch.clamp(samples, lower, upper)

    # cosine correction
    def _sample_hemisphere(self, num_points=1000, radius_range=(0.2, 0.7)):
        phi = torch.rand(num_points, device=self.device) * 2 * math.pi
        z = torch.rand(num_points, device=self.device)  # z ∈ [0, 1]
        radius = torch.empty(num_points, device=self.device).uniform_(
            radius_range[0], radius_range[1]
        )
        theta = torch.acos(z)  # corrected inclination

        x = torch.sin(theta) * torch.cos(phi)
        y = torch.sin(theta) * torch.sin(phi)
        z = torch.cos(theta)

        points = torch.stack([x, y, z], dim=1) * radius[:, None]
        return points

    def _sample_sphere(self, num_points: int = 1000, radius: float = 0.2) -> torch.Tensor:
        """
        Sample points uniformly distributed on the surface of a sphere.

        Args:
            num_points (int): Number of points to sample.
            radius (float): Radius of the sphere.

        Returns:
            torch.Tensor: Sampled points in shape [N, 3], where N is the number of points.
        """
        # Generate random spherical coordinates
        phi = torch.rand(num_points, device=self.device) * 2 * math.pi  # azimuth
        z = torch.rand(num_points, device=self.device) * 2 - 1  # z ∈ [-1, 1]
        theta = torch.acos(z)  # inclination

        # radius = torch.empty(num_points, device=self.device).uniform_(radius_range[0], radius_range[1])

        x = torch.sin(theta) * torch.cos(phi)
        y = torch.sin(theta) * torch.sin(phi)
        z = torch.cos(theta)

        points = torch.stack([x, y, z], dim=1) * radius  # scale by radius
        return points

    def _sample_around_ee_ball_restricted(
        self,
        envs_idx: torch.Tensor = None,
        radius_from_ee: float = 0.25,
        radius_from_root: float = 0.8,
    ) -> torch.Tensor:
        """
        Sample points in a sphere around the end effector, ensuring they are within a certain
        distance from the root. Points are clamped to be within a specified radius from the root
        and not below ground level.

        Args:
            envs_idx (torch.Tensor): Indices of environments to sample for.
            radius_from_ee (float): Radius around the end effector to sample points.
            radius_from_root (float): Maximum distance from the root (0, 0, 0) to clamp points.

        Returns:
            torch.Tensor: Sampled points in shape [N, 3], where N is the number of environments.
        """

        # get the end effector position
        ee_pos = self.robot.get_link("hand").get_pos(envs_idx)  # [N, 3]
        # print(f"[DEBUG] End effector positions: {ee_pos}")

        # generate a point in a sphere around the end effector
        pnts = self._sample_sphere(num_points=envs_idx.shape[0], radius=radius_from_ee)
        # print(f"[DEBUG] Sampled points around end effector: {pnts}")


        # now transport the points to the end effector position
        pnts = pnts + ee_pos  # [N, 3]
        # print(f"[DEBUG] Points after translation to end effector: {pnts}")

        # get debug distances between points and end effector positions
        new_distances = torch.norm(pnts - ee_pos, dim=-1)  # [N, 3] -> [N]
        # print(f"[DEBUG] Distances from sampled points to end effector after translation: {new_distances}")

        # # if any point is farther from the root than radius_from_root, clamp it to
        # (abs(radius_from_root), abs(radius_from_root), +radius_from_root) following (x,y,z)

        # Clamp points that are too far from root (0, 0, 0)
        distances = torch.norm(pnts, dim=-1)  # [N]
        # print(f"[DEBUG] Distances from points to root: {distances}")
        mask = distances > radius_from_root
        if mask.any():
            # print(f"[DEBUG] Clamping points that are too far from root: {pnts[mask]} where mask: {mask}")
            pnts[mask] = pnts[mask] / distances[mask].unsqueeze(-1) * radius_from_root

        # Ensure points are not below ground (z < 0)
        pnts[..., 2] = torch.clamp(pnts[..., 2], min=0.0)

        return pnts

    def _sample_goal_positions(self):

        # self.goal_pos = self._sample_circle(radius_range=(0.2, 0.7), z_range=(0.0, 1.0), num_points=self.num_envs)
        self.goal_pos = self._sample_around_ee_ball_restricted(
            envs_idx=self.envs_idx, radius_from_ee=self.curriculum_radius_from_ee
        )
        self.goal.set_pos(self.goal_pos, envs_idx=self.envs_idx)

    def _init_robot(self, envs_idx=None):
        """
        Initialize the robot's joint positions.
        If envs_idx is None, it initializes for all environments.
        """
        if envs_idx is None:
            envs_idx = self.envs_idx  # tensor([0, 1, 2, ..., num_envs])

        qpos = self._normal_sample_from_default_pose(
            envs_idx.shape[0], std_scale=1
        )  # [envs_idx.shape[0], 9]

        # change only newly sampled positions
        self.dof_targets[envs_idx] = qpos[:]  # shape [num_envs, num_actions] ~ [1024, 9]

        # just hard sets every time to the same position
        # qpos = torch.tensor([
        #     -1.5,  0.0,  0.0,
        #     -1.5,  0.0,  1.5,
        #     0.5,  0.04, 0.04,
        # ], device=self.device).expand(envs_idx.shape[0], -1).clone()

        self.robot.set_qpos(self.dof_targets[envs_idx], envs_idx=envs_idx)

        # dofs limits:
        # (tensor([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973,  0.0000, 0.0000], device='cuda:0'),
        #  tensor([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973,  0.0400, 0.0400], device='cuda:0'))

    # --------------0-----------0--------
    #               0           0
    #      |============================|
    #      |                            |
    #      |  Termination / Observation |
    #      |                            |
    #      |============================|

    def reached_goal(self):
        """
        Check if the gripper has reached the goal position.
        Returns a boolean tensor of shape (num_envs,).
        """
        gripper_pos = self.robot.get_link("hand").get_pos(envs_idx=self.envs_idx)
        distance_to_goal = torch.norm(gripper_pos - self.goal_pos, dim=1)
        reached = distance_to_goal <= self.goal_reach_threshold  # threshold for reaching the goal

        return reached

    def forbidden_contact(self):
        """Mask to reset envs that have collided with anything."""
        # get contact info from the robot
        contact_info = self.robot.get_contacts()
        valid_mask = contact_info.get("valid_mask", None)

        # valid_mask shape: (n_envs, n_contacts)
        # envs_with_contacts shape: (n_envs,), True if any valid contact
        envs_with_contacts = valid_mask.any(dim=1)
        # print(f"[DEBUG] envs_with_contacts: {envs_with_contacts}")

        return envs_with_contacts

    def get_dones(self):
        """
        Compute the done condition for the current step.
        Returns a tensor of shape (num_envs,).
        Also logs the number of dones in self.extras["log"].
        """
        reached_goal_envs = self.reached_goal()
        self.curriculum_goal_reached_once |= reached_goal_envs
        timeout_envs = self.episode_length_buf >= self.max_episode_length  # [N,]

        raw_dofs_vel = self.robot.get_dofs_velocity(envs_idx=self.envs_idx)  # [N, 9]
        # check if any robots went crazy with moving real fast or glitching
        crazy_envs = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if (raw_dofs_vel.abs() > 5.0).any():
            # get the envs of those robots
            crazy_mask = (raw_dofs_vel.abs() > 5.0).any(dim=1)
            crazy_envs = crazy_mask
            print(
                f"[DEBUG] Robots {self.envs_idx[crazy_mask]} are moving too fast! Resetting them."
            )

        contact_envs = self.forbidden_contact()

        final_dones = (
            timeout_envs | reached_goal_envs | crazy_envs | contact_envs
        )  # return True if either condition is met

        self.extras["log"]["episode/num_reached_goal_envs"] = reached_goal_envs.sum().item()
        self.extras["log"]["episode/num_timeout_envs"] = timeout_envs.sum().item()
        self.extras["log"]["episode/num_final_dones"] = final_dones.sum().item()
        self.extras["log"]["episode/num_contact_envs"] = contact_envs.sum().item()
        self.extras["log"]["episode/num_crazy_envs"] = crazy_envs.sum().item()
        self.extras["time_outs"] = timeout_envs

        return final_dones

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

    def check_observation_sanity(self):
        """
        Check for invalid or suspicious values in the observation buffer.
        Logs offending indices and raises an error if fatal issues are found.
        """
        obs = self.obs_buf  # [num_envs, num_obs]

        has_nan = torch.isnan(obs)
        has_inf = torch.isinf(obs)
        too_large = obs.abs() > 1e3
        all_zero = (obs == 0).all()
        any_none = obs is None

        if has_nan.any():
            print("[ERROR] Observation contains NaN values.")
            print("  -> NaN indices:", has_nan.nonzero(as_tuple=False))

        if has_inf.any():
            print("[ERROR] Observation contains Inf values.")
            print("  -> Inf indices:", has_inf.nonzero(as_tuple=False))

        if too_large.any():
            print("[WARNING] Observation contains unusually large values (> 1e3).")
            print("  -> Large value indices and values:")
            indices = too_large.nonzero(as_tuple=False)
            for idx in indices:
                env_idx, obs_idx = idx.tolist()
                print(f"    obs_buf[{env_idx}, {obs_idx}] = {obs[env_idx, obs_idx].item():.4e}")

        if all_zero:
            print(
                "[WARNING] Entire observation buffer is all zeros. This may indicate a faulty reset or uninitialized data."
            )

        if any_none:
            print("[ERROR] Observation buffer is None — check initialization.")
            raise ValueError("Observation buffer is None.")

        if has_nan.any() or has_inf.any() or any_none:
            raise ValueError("Invalid observation values detected. Halting for debugging.")

    # --------------0-----------0--------
    #               0           0
    #      |============================|
    #      |                            |
    #      |     Main Gym Functions     |
    #      |                            |
    #      |============================|

    def step(self, actions):

        envs_to_reset = self.reset_buf.nonzero(as_tuple=False).reshape((-1,))
        self.reset_idx(envs_to_reset=envs_to_reset)
        # delayed reset?


        ### Act (set the PD controller targets)
        self.act(actions)

        ### Step the scene
        self.scene.step()

        new_goals = self._sample_around_ee_ball_restricted(
            envs_idx=envs_to_reset,
            radius_from_ee=self.curriculum_radius_from_ee,
            radius_from_root=0.8,
        )
        self.goal_pos[envs_to_reset] = new_goals
        self.goal.set_pos(new_goals, envs_idx=envs_to_reset)

        ### Get observations and reward
        self.obs_buf, self.extras = self.get_observations()
        self.rew_buf = self.get_reward()
        self.episode_length_buf += 1
        # print(f"[DEBUG] self.episode_length_buf: {self.episode_length_buf}")

        ### Check if any envs are done
        self.reset_buf = self.get_dones()

        # reset the environments that are done
        self.num_steps += 1
        self.steps_from_last_curriculum += 1

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def act(self, actions):
        actions = actions.to(self.device)
        # log action max and mins before clamping
        self.extras["log"]["obs/action_max_mean_noclamp"] = self.actions.max(dim=1).values.mean()
        self.extras["log"]["obs/action_min_mean_noclamp"] = self.actions.min(dim=1).values.mean()
        self.actions = actions.clone().clamp(-1.0, 1.0)

        target_q = (
            self.dof_targets + self.dof_speed_scales * self.dt * self.actions * self.action_scale
        )
        self.dof_targets[:] = torch.clamp(target_q, self.dof_lower_limits, self.dof_upper_limits)
        # print(f"self.dof_targets: {self.dof_targets}")

        # sets the PD targets
        self.robot.control_dofs_position(self.dof_targets, envs_idx=self.envs_idx)

    def get_reward(self):
        """
        Compute the reward for the current step.
        Returns a tensor of shape (num_envs,).
        """

        r_proximity_reward = grasp_proximity_reward(
            self.robot.get_link("hand").get_pos(), self.goal_pos, scale=1.0
        )  # grasp proximity reward
        r_action_penalty = action_penalty(self.actions, scale=-0.01)  # action penalty
        r_joint_velocity_penalty = joint_velocity_penalty(
            self.raw_dofs_vel, scale=-0.001
        )  # joint velocity penalty
        r_joint_acc_penalty = joint_acc_penalty(
            self.raw_dofs_acc, scale=-0.001
        )  # joint acceleration penalty

        self.extras["log"] = {
            "rewards/grasp_proximity": r_proximity_reward,
            "rewards/action_penalty": r_action_penalty,
            "rewards/joint_velocity_penalty": r_joint_velocity_penalty,
            "rewards/joint_acc_penalty": r_joint_acc_penalty,
        }

        reward = torch.zeros_like(self.rew_buf, device=self.device, dtype=gs.tc_float)

        reward += r_proximity_reward
        reward += r_action_penalty
        reward += r_joint_velocity_penalty

        self.extras["log"]["rewards/total"] = reward

        return reward

    def get_observations(self):
        # actions = self.actions  # [N, 9]

        raw_dofs_vel = self.robot.get_dofs_velocity(envs_idx=self.envs_idx)  # [N, 9]

        # compute accelerations
        self.raw_dofs_acc = self.raw_dofs_vel - raw_dofs_vel

        self.raw_dofs_vel = raw_dofs_vel  # [N, 9]

        self.raw_dofs_pos = self.robot.get_dofs_position(envs_idx=self.envs_idx)  # [N, 9]

        # normalize dofs_pos and dofs_vel
        self.dof_pos = self.normalize_dof_pos(self.raw_dofs_pos)  # normalize dofs_pos to [-1, 1]
        self.dof_vel = self.normalize_dof_vel(self.raw_dofs_vel)  # normalize dofs_vel to [-1, 1]

        gripper_pos = self.robot.get_link("hand").get_pos()  # [N, 3]
        goal_pos = self.goal_pos  # [N, 3]
        distance_to_goal = torch.norm(gripper_pos - goal_pos, dim=1, keepdim=True)  # [N, 1]
        relative_goal_pos = goal_pos - gripper_pos  # [N, 3]

        # check for curriculum related things
        if distance_to_goal.mean() <= self.curriculum_radius_from_ee:
            # increase radius only if the time since last curriculum update is enough
            if self.steps_from_last_curriculum > 1000:
                self.steps_from_last_curriculum = 0

                for _ in range(3):
                    print(f"[X] [X] [X] [X] [X] [X]\n")
                print(
                    f"[INFO] Curriculum goal reached: mean distance to goal {distance_to_goal.mean():.4f} <= radius {self.curriculum_radius_from_ee:.4f}"
                )
                self.curriculum_radius_from_ee *= 1.1
                self.curriculum_radius_from_ee = min(self.curriculum_radius_from_ee, 0.8)
                print(
                    f"[INFO] Increasing curriculum radius to {self.curriculum_radius_from_ee:.4f}"
                )
                for _ in range(3):
                    print(f"[X] [X] [X] [X] [X] [X]\n")
        # CURRICULUM: compute percentage of successful envs
        # successes = self.curriculum_goal_reached_once[envs_to_reset].sum().item()

        # success_rate = successes / (self.num_envs + 1e-6)  # avoid division by zero
        # self.curriculum_goal_success_rate_ema = (
        #     self.curriculum_goal_success_smoothing * self.curriculum_goal_success_rate_ema
        #     + (1.0 - self.curriculum_goal_success_smoothing) * success_rate
        # )

        # # also track the curriculum success rate
        # self.extras["debug/curriculum_success_rate"] = self.curriculum_goal_success_rate_ema
        # if self.num_steps % 100 == 0:
        #     print(
        #         f"[DEBUG] Curriculum success rate EMA: {self.curriculum_goal_success_rate_ema:.2f} (successes: {successes}/{self.num_envs})"
        #     )
        #     print(
        #         f"[DEBUG] Curriculum radius from end-effector: {self.curriculum_radius_from_ee:.2f}"
        #     )

        # if self.curriculum_goal_success_rate_ema > 0.5:
        #     # increase the radius from end-effector for sampling
        #     self.curriculum_radius_from_ee *= 1.2
        #     self.curriculum_radius_from_ee.clamp_(min=0.2, max=0.7)
        #     print(
        #         f"[DEBUG] Curriculum success rate {self.curriculum_goal_success_rate_ema:.2f} > 0.5, increasing radius to {self.curriculum_radius_from_ee:.2f}"
        #     )


        self.obs_buf = torch.cat(
            [
                self.actions,  # 9
                self.dof_pos,  # 9
                self.dof_vel,  # 9
                relative_goal_pos,  # 3
            ],
            axis=-1,
        )  # [N, num_obs]

        self.check_observation_sanity()

        if self.obs_buf.shape[1] != self.num_obs:
            raise ValueError(
                f"Observation buffer shape mismatch: expected {self.num_obs}, got {self.obs_buf.shape[1]}"
            )

        self.extras["observations"]["critic"] = self.obs_buf
        self.extras["observations"]["actions"] = self.actions
        self.extras["observations"]["dofs_pos"] = self.dof_pos
        self.extras["observations"]["dofs_vel"] = self.dof_vel
        self.extras["observations"]["gripper_pos"] = gripper_pos
        self.extras["observations"]["goal_pos"] = goal_pos
        self.extras["observations"]["distance_to_goal"] = distance_to_goal.squeeze(1)

        self.extras["log"]["obs/distance_to_goal"] = distance_to_goal.squeeze(1)
        self.extras["log"]["obs/curriculum_radius_from_ee"] = self.curriculum_radius_from_ee

        return self.obs_buf, self.extras

    def reset_idx(self, envs_to_reset):
        """
        Reset the environments specified by envs_to_reset.
        This is a helper function to reset specific environments.
        """
        if envs_to_reset.numel() == 0:
            return

        # print(f"[DEBUG] Resetting environments: {envs_to_reset}")
        self.robot.zero_all_dofs_velocity(envs_to_reset)
        self._init_robot(envs_idx=envs_to_reset)

        # new_goals = self._sample_around_ee_ball_restricted(
        #     envs_idx=envs_to_reset,
        #     radius_from_ee=self.curriculum_radius_from_ee,
        #     radius_from_root=0.8,
        # )
        # self.goal_pos[envs_to_reset] = new_goals
        # self.goal.set_pos(new_goals, envs_idx=envs_to_reset)

        # Uncomment to sample a hemisphere instead of a ball around the end-effector
        # new_goals = self._sample_hemisphere(num_points=envs_to_reset.shape[0])
        # self.goal_pos[envs_to_reset] = new_goals
        # self.goal.set_pos(new_goals, envs_idx=envs_to_reset)

        self.episode_length_buf[envs_to_reset] = 0
        self.reset_buf[envs_to_reset] = False

    def reset(self):
        self.reset_buf[:] = False
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        return self.obs_buf, None


if __name__ == "__main__":
    gs.init(backend=gs.gpu, precision="32", logging_level="info")
    env = FrankaReachTask(
        num_envs=8,
        env_cfg={"action_scale": 1.0},
        obs_cfg={},
        reward_cfg={},
        command_cfg={},
        show_viewer=True,
    )
    env.reset()

    for res in range(5):
        print(f"Resetting environment {res + 1}")
        env.force_reset()
        for _ in range(500):
            actions = torch.randn((env.num_envs, 9), device=env.device)
            actions = torch.zeros_like(actions)
            obs, rew, done, extras = env.step(actions)
            # print everything in a formatted way
            # print(f"Obs: {obs}, Reward: {rew}, Done: {done}, Extras: {extras}")
            if done.any():
                print(f"\n\nResetting naturally.\n\n")
                env.reset()
