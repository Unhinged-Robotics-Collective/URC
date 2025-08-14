import numpy as np
import genesis as gs

HOME_QPOS = [0, -np.pi/2, np.pi/2, np.pi/2, -np.pi/2, 0]

QPOS_MAP = {
        "middle": [0, -np.pi/2, np.pi/2, 0, 0, -0.15],
        "zero": [0, 0, 0, 0, 0, 0],
        "rotated": [-np.pi/2, -np.pi/2, np.pi/2, np.pi/2, -np.pi/2, np.pi/2],
        "rest": [0.049, -3.32, 3.14, 1.21, -0.17, -0.17]
    }


def setup_scene():
    gs.init(backend=gs.gpu)
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0, -3.5, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=30,
            max_FPS=60,
        ),
        sim_options=gs.options.SimOptions(dt=0.01),
        show_viewer=True,
    )
    plane = scene.add_entity(gs.morphs.Plane())
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/trs_so_arm100/so_arm100.xml")
    )
    scene.build()
    return scene, franka

def get_motors_dof_idx(franka):
    joints_name = (
        "Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"
    )
    return [franka.get_joint(name).dofs_idx_local[0] for name in joints_name]

def set_control_gains(franka, motors_dof_idx):
    # Set positional and velocity gains
    franka.set_dofs_kp(
        kp=np.array([4500, 4500, 3500, 3500, 2000, 2000]),
        dofs_idx_local=motors_dof_idx,
    )
    franka.set_dofs_kv(
        kv=np.array([450, 450, 350, 350, 200, 200]),
        dofs_idx_local=motors_dof_idx,
    )
    # Set force range for safety
    franka.set_dofs_force_range(
        lower=np.array([-87, -87, -87, -87, -12, -12]),
        upper=np.array([87, 87, 87, 87, 12, 12]),
        dofs_idx_local=motors_dof_idx,
    )   

def hard_reset(franka, motors_dof_idx, scene):
    for i in range(600):
        if i < 100:
            franka.set_dofs_position(np.array(HOME_QPOS), motors_dof_idx)
        elif i < 200:
            franka.set_dofs_position(np.array(QPOS_MAP["middle"]), motors_dof_idx)
        elif i < 300:
            franka.set_dofs_position(np.array(QPOS_MAP["rest"]), motors_dof_idx)
        elif i < 400:
            franka.set_dofs_position(np.array(QPOS_MAP["rotated"]), motors_dof_idx)
        elif i < 500:
            franka.set_dofs_position(np.array(QPOS_MAP["zero"]), motors_dof_idx)
        else:
            franka.set_dofs_position(np.array(HOME_QPOS), motors_dof_idx)
            
        scene.step()

def control_loop(franka, motors_dof_idx, scene):
    for i in range(1250):
        if i == 0:
            franka.control_dofs_position(np.array([1, 1, 0, 0, 0.04, 0.04]), motors_dof_idx)
        elif i == 250:
            franka.control_dofs_position(np.array([-1, 0.8, 0.5, -0.5, 0.04, 0.04]), motors_dof_idx)
        elif i == 500:
            franka.control_dofs_position(np.array([0, 0, 0, 0, 0, 0]), motors_dof_idx)
        elif i == 750:
            franka.control_dofs_position(np.array([0, 0, 0, 0, 0, 0])[1:], motors_dof_idx[1:])
            franka.control_dofs_velocity(np.array([1.0]), [motors_dof_idx[0]])
        elif i == 1000:
            franka.control_dofs_force(np.array([0, 0, 0, 0, 0, 0]), motors_dof_idx)
        # Print every 50 steps for performance
        if i % 50 == 0:
            print(f"Step {i}:")
            print("  control force:", franka.get_dofs_control_force(motors_dof_idx))
            print("  internal force:", franka.get_dofs_force(motors_dof_idx))
        scene.step()

def main():
    scene, franka = setup_scene()
    motors_dof_idx = get_motors_dof_idx(franka)
    set_control_gains(franka, motors_dof_idx)
    hard_reset(franka, motors_dof_idx, scene)
    #control_loop(franka, motors_dof_idx, scene)

if __name__ == "__main__":
    main()
