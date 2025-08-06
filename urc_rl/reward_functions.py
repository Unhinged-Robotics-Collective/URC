import torch

# --------------0-----------0--------
#               0           0
#      |============================|
#      |                            |
#      |     Rewards / Penalties    |
#      |                            |
#      |============================|

def joint_centering_penalty(dof_limits, dof_positions, scale=-0.1):
    lower, upper = dof_limits
    midpoint = (lower + upper) / 2
    range_ = (upper - lower) / 2
    penalty = torch.norm((dof_positions - midpoint) / range_, dim=1)  # [N,], positive
    return scale * penalty

def action_penalty(actions, scale=-0.01):
    action_penalty = torch.sum(actions ** 2, dim=1)  # [N,]

    return scale * action_penalty

def joint_acc_penalty(raw_dofs_acc, scale=-0.0001):
    """
    Penalize joint accelerations using L2 norm squared.

    Returns a tensor of shape (num_envs,).
    """
    penalty = torch.sum(raw_dofs_acc ** 2, dim=1)  # [N,]
    return scale * penalty

def joint_velocity_penalty(raw_dofs_vel, scale=-0.0001):
    """
    Penalize joint velocities using L2 norm squared.

    Returns a tensor of shape (num_envs,).
    """
    penalty = torch.sum(raw_dofs_vel ** 2, dim=1)  # [N,]
    return scale * penalty

def contact_penalty(contact_vector, num_envs, device, scale=-0.1):
    # Get contact info from robot
    contact_info = contact_vector
    valid_mask = contact_info.get("valid_mask", None)

    # Count per-env contacts
    if valid_mask is not None:
        contact_count = valid_mask.sum(dim=1).float()
    else:
        contact_count = torch.tensor(
            [len(contact_info["position"])] * num_envs,
            dtype=torch.float32,
            device=device
        )

    penalty = scale * contact_count  # [N,]

    return penalty, contact_count

def grasp_proximity_reward(grasp_pos, goal_pos, scale=1.0):
    """
    Reward for how close the gripper is to the target grasp position.
    Gives smoothly decaying reward with a precision bonus near the goal.

    Returns a tensor of shape (num_envs,), positive reward in range (0, 2 * scale].
    """

    d = torch.norm(grasp_pos - goal_pos, p=2, dim=-1)  # [N,]
    r = 1.0 / (1.0 + d**2)        # Smooth inverse squared
    r = r * r                     # Sharpened peak
    r = torch.where(d <= 0.1, r * 2.0, r)  # Proximity bonus

    return scale * r