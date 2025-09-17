from scipy.spatial.transform import Rotation as R
import numpy as np

from dataset_generation.hand_to_pose import *


# Example usage
forward = [0.0, 1.0, 1.0]
side    = [1.0, 1.0, 0.0]
up      = [1.0, 1.0, 1.0]

R_world_from_local = rotation_from_directions(forward, side, up,
                                              axes_order=("side","up","forward"),
                                              right_handed=True)

# Convert to SciPy Rotation for quaternions/Euler if needed
rot = R.from_matrix(R_world_from_local)
quat_xyzw = rot.as_quat()       # [x,y,z,w]
euler_xyz = rot.as_euler("xyz") # radians
print(rot.as_matrix())
print(rot.as_rotvec())
print(quat_xyzw)
print(euler_xyz)
