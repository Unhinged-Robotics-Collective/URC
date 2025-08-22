import numpy as np
import genesis as gs
import math
from scipy.spatial.transform import Rotation as R


# 1. Initialize Genesis
gs.init(backend=gs.gpu)

# 2. Create and configure the scene
scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.01),
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(3, -1, 1.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=30,
        max_FPS=60,
    ),
    show_viewer=True,
)

# 3. Add entities
scene.add_entity(gs.morphs.Plane())
cube1 = scene.add_entity(gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0.65, 0.0, 0.02)))
cube2 = scene.add_entity(gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0.65, 0.65, 0.02)))
#franka = scene.add_entity(gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'))

theta = math.radians(45)  # 45° around Z
rot_quat = R.from_euler('z', theta).as_quat()  # returns [x, y, z, w]

franka = scene.add_entity(
    gs.morphs.MJCF(
        file="xml/trs_so_arm100/so_arm100.xml",
        pos=(0, 0, 0),
        euler = (0, 0, 135)
    )
)
scene.build()

# 4. PD gains
#franka.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
franka.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000]))

# franka.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))
franka.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200]))

# franka.set_dofs_force_range(
#     np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
#     np.array([ 87,  87,  87,  87,  12,  12,  12,  100,  100])
# )

joints_name = (
        "Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"
    )
motors_dof_idx = [franka.get_joint(name).dofs_idx_local[0] for name in joints_name]

franka.set_dofs_force_range(
        lower=np.array([-87, -87, -87, -87, -12, -12]),
        upper=np.array([87, 87, 87, 87, 12, 12]),
        dofs_idx_local=motors_dof_idx,
    )  

# 5. Define target pose for IK
end_effector = franka.get_link('Moving_Jaw')
target_pos1 = np.array([0.25, 0.0, 0.02])
target_pos2 = np.array([0.0, 0.25, 0.02])
rest_pos = np.array([0.049, -3.32, 3.14, 1.21, -0.17, -0.17])

r = R.from_euler('xyz', [math.pi, 0, 0])  # flip 180° around X
target_quat = r.as_quat()  # returns [x, y, z, w]
#target_quat = np.array([0, 0, 1, 0])

# 6. Compute IK once
qpos_target = franka.inverse_kinematics(link=end_effector, pos=target_pos1, quat=target_quat)
qpos_target[-2:] = 0.04  # open gripper

# 7. Interpolate manually from current to target configuration
qpos_current = franka.get_qpos()
steps = 200
for i in range(steps):
    alpha = (i + 1) / steps
    qpos_interp = qpos_current * (1 - alpha) + qpos_target * alpha
    franka.control_dofs_position(qpos_interp)
    scene.step()

# Reset to home position
franka.set_dofs_position(rest_pos, motors_dof_idx)

# 8. Move to second target position
qpos_current = franka.get_qpos()
qpos_target = franka.inverse_kinematics(link=end_effector, pos=target_pos2, quat=target_quat)
qpos_target[-2:] = 0.04  # open gripper

for i in range(steps):
    alpha = (i + 1) / steps
    qpos_interp = qpos_current * (1 - alpha) + qpos_target * alpha
    franka.control_dofs_position(qpos_interp)
    scene.step()
    
# 8. Hold final pose
for i in range(steps):
    scene.step()