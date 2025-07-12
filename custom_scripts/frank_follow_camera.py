import numpy as np
import time
import torch
import argparse
from typing import Callable
import numpy as np
import math
import genesis as gs
from genesis.utils.geom import quat_to_R
import genesis as gs

########################## init ##########################
gs.init(backend=gs.gpu, precision="32")
########################## create a scene ##########################
scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(3, -1, 1.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=30,
        res=(960, 640),
        max_FPS=60,
    ),
    sim_options=gs.options.SimOptions(
        dt=0.01,
    ),
    rigid_options=gs.options.RigidOptions(
        box_box_detection=True,
    ),
    show_viewer=False,# turn on/off default camera
)

########################## entities ##########################
plane = scene.add_entity(
    gs.morphs.Plane(),
)
franka = scene.add_entity(
    gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
)

sphere = scene.add_entity(
    gs.morphs.Sphere(
        radius=0.05,
        pos=(0.65, 0.2, 0.2),
    )
)
cam_0 = scene.add_camera(
    res=(1920, 1080),
    pos=(3, -1, 1.5),
    lookat=(0.0, 0.0, 0.5),
    fov=60,
    GUI=True,
    spp=4,
)
cube = scene.add_entity(
    gs.morphs.Box(
        size=(0.04, 0.04, 0.04),
        pos=(0.65, 0.0, 0.02),
    )
)
########################## build ##########################
scene.build()

motors_dof = np.arange(7)
fingers_dof = np.arange(7, 9)
qpos = np.array([-1.0124, 1.5559, 1.3662, -1.6878, -1.5799, 1.7757, 1.4602, 0.04, 0.04])
franka.set_qpos(qpos)
scene.step()

end_effector = franka.get_link("hand")
qpos = franka.inverse_kinematics(
    link=end_effector,
    pos=np.array([0.65, 0.0, 0.135]),
    quat=np.array([0, 1, 0, 0]),
)

franka.control_dofs_position(qpos[:-2], motors_dof)

cam_0.start_recording()
# hold
for i in range(100):
    print("hold", i)
    scene.step()
    cam_0.render(
        rgb=True,
        # depth        = True,
        # segmentation = True,
    )

# grasp
finder_pos = -0.0
for i in range(100):
    hand_p: np.ndarray = franka.get_links_pos()[-3].cpu().numpy()
    hand_r: np.ndarray = franka.get_links_quat()[-3].cpu().numpy()
    R = quat_to_R(hand_r)
    # hand + R @ x + R @ z => look at
    # hand + R @ x => cam pos
    # R @ x => cam up
    up = R @ np.array([0.05, 0.0, 0.0])
    cam_pos = up + hand_p
    look_at = cam_pos + R @ np.array([0.0, 0.0, 1.0])
    cam_0.set_pose(pos=cam_pos, lookat=look_at, up=up)
    
    print("grasp", i)
    franka.control_dofs_position(qpos[:-2], motors_dof)
    franka.control_dofs_position(np.array([finder_pos, finder_pos]), fingers_dof)
    scene.step()
    cam_0.render(
        rgb=True,
        # depth        = True,
        # segmentation = True,
    )    

# lift
qpos = franka.inverse_kinematics(
    link=end_effector,
    pos=np.array([0.65, 0.0, 0.3]),
    quat=np.array([0, 1, 0, 0]),
)
for i in range(100):
    print("lift", i)
    hand_p: np.ndarray = franka.get_links_pos()[-3].cpu().numpy()
    hand_r: np.ndarray = franka.get_links_quat()[-3].cpu().numpy()
    R = quat_to_R(hand_r)
    # hand + R @ x + R @ z => look at
    # hand + R @ x => cam pos
    # R @ x => cam up
    up = R @ np.array([0.05, 0.0, 0.0])
    cam_pos = up + hand_p
    look_at = cam_pos + R @ np.array([0.0, 0.0, 1.0])
    cam_0.set_pose(pos=cam_pos, lookat=look_at, up=up)
    
    franka.control_dofs_position(qpos[:-2], motors_dof)
    franka.control_dofs_position(np.array([finder_pos, finder_pos]), fingers_dof)
    scene.step()

    cam_0.render(
        rgb=True,
        # depth        = True,
        # segmentation = True,
    )
cam_0.stop_recording("cube_pov.mp4")
