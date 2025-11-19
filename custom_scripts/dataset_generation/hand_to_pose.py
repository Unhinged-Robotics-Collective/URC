"""
Reasoning:
index 8 => tip of index finger
7,6,5 => rest of index finger
When grasping, the finger usually moves about joint 5
index 4 => tip of thumb
"""
import numpy as np

def hand_to_points(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tip_of_index = xyz[8]
    tip_of_thumb = xyz[4]
    # these two are used to find the open/closed dist
    base_of_index = xyz[5]
    base_of_thumb = xyz[2]
    gripper_base = 0.5 * (base_of_index + base_of_thumb)
    gripper_tip = (0.25 * tip_of_index + 0.75 * tip_of_thumb) # roughly doesnt move
    return gripper_base, gripper_tip


def hand_to_pose(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns:
        tuple R, t"""
    # 3 points are necessary
    gripper_base, gripper_tip = hand_to_points(xyz)
    forward_dir = gripper_tip - gripper_base
    tip_of_index = xyz[8]
    tip_of_thumb = xyz[4]
    side_dir = tip_of_thumb - tip_of_index
    up_dir = np.cross(forward_dir, side_dir)
    return rotation_from_directions(forward_dir, side_dir, up_dir), gripper_base



def _norm(v, eps=1e-12):
    n = np.linalg.norm(v)
    if n < eps:
        n = 1
        # raise ValueError("Zero (or near-zero) vector encountered.")
    return v / n


def rotation_from_directions(forward_dir: np.ndarray, side_dir: np.ndarray, up_dir: np.ndarray,
                             axes_order=("side", "up", "forward"),
                             right_handed=True):
    """Return a proper rotation matrix (3x3) from 3 direction hints.

    Parameters
    ----------
    forward_dir, side_dir, up_dir : array_like shape (3,)
        Approximate axis directions expressed in the *destination/world* frame.
    axes_order : tuple[str]
        Which axes to put as columns, e.g. ("side","up","forward").
        Common choices:
          - graphics 'look' matrix (right-handed, +Z forward): ("side","up","forward")
          - robotics (x,y,z): ("side","up","forward") if side=x, up=y, forward=z
          - if your forward is -Z, swap/negate accordingly.
    right_handed : bool
        Enforce right-handed frame (determinant +1) by flipping one axis if needed.

    Returns
    -------
    R : ndarray shape (3,3)
        Rotation matrix mapping a vector from the *local* frame spanned by
        (side, up, forward) to the *world* frame: v_world = R @ v_local.
        Its inverse is R.T.
    """
    f = _norm(np.asarray(forward_dir, dtype=float))

    # Gram–Schmidt to ensure orthonormality and stability even if inputs aren't exact
    s_raw = np.asarray(side_dir, dtype=float)
    s = s_raw - f * np.dot(s_raw, f)
    s = _norm(s)

    u_raw = np.asarray(up_dir, dtype=float)
    u = u_raw - f * np.dot(u_raw, f) - s * np.dot(u_raw, s)
    u = _norm(u)

    # Assemble columns in requested order
    axes = {"forward": f, "side": s, "up": u}
    Rm = np.column_stack([axes[name] for name in axes_order])

    # Enforce proper rotation (determinant +1)
    if right_handed and np.linalg.det(Rm) < 0:
        # Flip the smallest-impact axis (heuristic: flip the last one)
        last = axes_order[-1]
        axes[last] = -axes[last]
        Rm = np.column_stack([axes[name] for name in axes_order])

    # Final sanity checks (optional):
    # assert np.allclose(Rm.T @ Rm, np.eye(3), atol=1e-3)
    assert np.linalg.det(Rm) >= 0

    return Rm


