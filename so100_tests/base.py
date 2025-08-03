import numpy as np

import genesis as gs

########################## init ##########################
gs.init(backend=gs.gpu)

########################## create a scene ##########################
scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(0, -3.5, 2.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=30,
        max_FPS=60,
    ),
    sim_options=gs.options.SimOptions(
        dt=0.01,
    ),
    show_viewer=True,
)

########################## entities ##########################
plane = scene.add_entity(
    gs.morphs.Plane(),
)
franka = scene.add_entity(
    gs.morphs.MJCF(
        file="xml/trs_so_arm100/so_arm100.xml",
    ),
)

# franka = scene.add_entity(
#     gs.morphs.MJCF(
#         file="xml/franka_emika_panda/panda.xml",
#     ),
# )

########################## build ##########################
scene.build()

print(franka.__dict__)

# for i in range(1000):
#     scene.step()