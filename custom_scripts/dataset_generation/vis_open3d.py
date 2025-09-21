
import numpy as np
import open3d as o3d
import threading
from logging import getLogger
from multiprocessing import Event
from scipy.spatial.transform import Rotation

from dataset_generation.socket import XYZMetadata, XYZHandler

logger = getLogger(__name__)
import numpy as np
import open3d as o3d

def set_camera(ctr: o3d.visualization.ViewControl,
               eye: np.ndarray,
               lookat: np.ndarray,
               up: np.ndarray) -> None:
    """Set Open3D camera from eye, lookat, up vectors."""
    eye = np.array(eye, dtype=float)
    lookat = np.array(lookat, dtype=float)
    up = np.array(up, dtype=float)

    front = (lookat - eye)
    front /= np.linalg.norm(front)

    right = np.cross(up, front)
    right /= np.linalg.norm(right)

    true_up = np.cross(front, right)

    # Rotation (camera to world → transpose for world to camera)
    R = np.vstack([right, true_up, -front])
    t = -R @ eye

    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = t

    params = ctr.convert_to_pinhole_camera_parameters()
    params.extrinsic = extrinsic
    ctr.convert_from_pinhole_camera_parameters(params)


def make_frame(scale: float = 0.2) -> o3d.geometry.LineSet:
    """Create a coordinate frame as a LineSet (X=red, Y=green, Z=blue)."""
    points = [
        [0, 0, 0],
        [scale, 0, 0],
        [0, scale, 0],
        [0, 0, scale],
    ]
    lines = [
        [0, 1],  # X-axis
        [0, 2],  # Y-axis
        [0, 3],  # Z-axis
    ]
    colors = [
        [1, 0, 0],  # red
        [0, 1, 0],  # green
        [0, 0, 1],  # blue
    ]
    frame = o3d.geometry.LineSet()
    frame.points = o3d.utility.Vector3dVector(points)
    frame.lines = o3d.utility.Vector2iVector(lines)
    frame.colors = o3d.utility.Vector3dVector(colors)
    return frame


class PositionVisualizer:
    def __init__(self, stop_event: threading.Event, sub_metadata: XYZMetadata):
        self.xyz_handler = XYZHandler(sub_metadata)
        self.stop_event = stop_event
        self.sub_metadata = sub_metadata
        self.transforms = np.zeros((self.sub_metadata.num_rows, 4, 4))

        # one LineSet per row
        self.frames: list[o3d.geometry.LineSet] = []
        for _ in range(self.sub_metadata.num_rows):
            self.transforms[_] = np.eye(4)
            self.frames.append(make_frame(0.2))

        # start update thread
        threading.Thread(target=self.update_positions, daemon=True).start()

    def update_positions(self):
        logger.debug("Updating positions")
        count = 0
        while not self.stop_event.is_set():
            if count == self.xyz_handler._counter[0]:
                continue
            # just update translations
            self.transforms[:, :3, 3] = self.xyz_handler.hands[:self.sub_metadata.num_rows]
            count = self.xyz_handler._counter[0]

    def run(self):
        vis = o3d.visualization.Visualizer()
        vis.create_window("Position Visualizer")

        for frame in self.frames:
            vis.add_geometry(frame)

        # SETUP CAMERA
        ctr = vis.get_view_control()
        params = ctr.convert_to_pinhole_camera_parameters()
        extrinsic = np.eye(4)
        extrinsic[2, 3] = 1
        params.extrinsic = extrinsic
        ctr.convert_from_pinhole_camera_parameters(params)
        # ctr.set_zoom(1.5)

        ctr.set_lookat([0,0,0])
        # ctr.set_front(front.tolist())
        # ctr.set_up([0, 0, 1])
        # render = o3d.visualization.rendering.OffscreenRenderer(1024, 768)
        # scene = render.scene
        # cam = scene.camera
        # cam.set_projection(fov_deg=60.0, aspect=1024/768, near=0.01, far=100.0)

        # set_camera(ctr, eye=[5,5,5], lookat=[0,0,0], up=[0,0,1])

        params = ctr.convert_to_pinhole_camera_parameters()
        print("Extrinsic:\n", params.extrinsic)
        print("Intrinsic:\n", params.intrinsic.intrinsic_matrix)

        # Main event/rendering loop
        while not self.stop_event.is_set():
            for i, frame in enumerate(self.frames):
                T = self.transforms[i]
                # print(T)
                base = np.array([[0, 0, 0],
                                [0.2, 0, 0],
                                [0, 0.2, 0],
                                [0, 0, 0.2]])
                base = (T[:3, :3] @ base.T).T + T[:3, 3]
                frame.points = o3d.utility.Vector3dVector(base)
                vis.update_geometry(frame)

            vis.poll_events()      # must be in main thread
            vis.update_renderer()  # must be in main thread

        vis.destroy_window()



def main():
    stop_event = Event()
    num_frames = 100
    pub_sub_path = "vis_pos"
    _, sub_metadata = XYZMetadata.create_pair(
        num_frames,
        np.dtype(np.float64),
        pub_sub_path
    )
    vis = PositionVisualizer(stop_event, sub_metadata)
    vis.run()


if __name__ == "__main__":
    main()
