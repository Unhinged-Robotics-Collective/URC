import numpy as np
import threading
from logging import getLogger
from multiprocessing import Event

import vispy
vispy.use("glfw")  # force GLFW backend, avoids Qt
from vispy import app, scene

from dataset_generation import config
from dataset_generation.socket import XYZMetadata, XYZHandler
from dataset_generation.deforms import umeyama_transform, calculate_cam_matrices
from dataset_generation.hand_to_pose import hand_to_points, hand_to_pose

from enum import Enum, auto

from multiprocessing.synchronize import Event as EventType

logger = getLogger(__name__)


class VisTransforms(Enum):
    NONE = auto()
    UMEYAMA = auto()
    TRIANGULATE = auto()


SELECTED_TRANSFORM = VisTransforms.NONE


class PositionVisualizer(scene.SceneCanvas):
    def __init__(self, stop_event: EventType, sub_metadata: XYZMetadata):
        super().__init__(keys="interactive", size=(800, 600), title="Position Visualizer", show=True)
        self.unfreeze()
        self.count = 0
        self.stop_event = stop_event
        self.sub_metadata = sub_metadata
        self.xyz_handler = XYZHandler(sub_metadata)

        # Add a central widget with a view
        self.unfreeze()
        self.view = self.central_widget.add_view()
        self.view.camera = scene.cameras.TurntableCamera(fov=45, azimuth=45, elevation=20, distance=2.5)

        # Add world axis
        axis = scene.visuals.XYZAxis(parent=self.view.scene)  # type: ignore[attr-defined]

        # Scatter plot for points
        self.num_points = self.sub_metadata.num_rows
        self.scatter = scene.visuals.Markers(parent=self.view.scene)  # type: ignore[attr-defined]
        self.scatter.set_data(np.zeros((self.num_points, 3)),
                              face_color=(1, 1, 1, 1),
                              size=5)

        # Background thread for updating positions
        self.latest_points = np.random.rand(self.num_points, 3)
        # threading.Thread(target=self.update_positions, daemon=True).start()

        # Timer for GUI refresh
        self.timer = app.Timer(interval=1/30, connect=self.update_scatter, start=True)
        self.freeze()

    def update_positions(self):
        """Background: pull latest positions from shared memory."""
        count = 0
        while not self.stop_event.is_set():
            if count == self.xyz_handler.get_counter():
                continue
            count = self.xyz_handler.get_counter()
            # debug points expansion would go here

    def update_scatter(self, event):
        """GUI: redraw scatter plot."""
        if self.latest_points is not None:
            if self.count == self.xyz_handler.get_counter():
                return
            # try:
            print("hands", self.xyz_handler.hands.shape)
            self.count = self.xyz_handler.get_counter()
            match SELECTED_TRANSFORM:
                case VisTransforms.NONE:
                    hand_data = self.xyz_handler.hands
                    print("hand data", hand_data.shape)
                    self.scatter.set_data(self.xyz_handler.hands.reshape(-1, 3), face_color=(1, 1, 1, 1), size=5)
                case VisTransforms.UMEYAMA:
                    hands1: np.ndarray = self.xyz_handler.hands[0][0] # 21, 3
                    hands2: np.ndarray = self.xyz_handler.hands[1][0] # 21, 3
                    new_hands1 = umeyama_transform(hands1.T, hands2.T)
                    self.latest_points = self.xyz_handler.hands.copy()
                    self.latest_points[0][0] = new_hands1.T
                    self.scatter.set_data(self.latest_points.reshape(-1, 3), face_color=(1, 1, 1, 1), size=5)
                case VisTransforms.TRIANGULATE:
                    hands1: np.ndarray = self.xyz_handler.hands[0][0] # 21, 3
                    hands2: np.ndarray = self.xyz_handler.hands[1][0] # 21, 3
                    print("TRIANGULATE")
                    unnorm_uv1 = config.CAMS[0].unnormalize(hands1[:, :2])
                    unnorm_uv2 = config.CAMS[1].unnormalize(hands2[:, :2])
                    # print("UNNORMED", unnorm_uv1, unnorm_uv2)
                    hand_xyz = calculate_cam_matrices(config.CAMS[0], config.CAMS[1], unnorm_uv1, unnorm_uv2)
                    # print("CALCULATED MATRICES")
                    # print(np.pad(hand_xyz, ((0, self.num_points - hand_xyz.shape[0]), (0, 0))))
                    self.scatter.set_data(hand_xyz, face_color=(1, 1, 1, 1), size=5)
                    print("CALCULATED MATRICES")
                case _:
                    raise ValueError(f"Invalid option selected: {_}")
                    # default do nothing
            # except Exception as e:
                # print(e)


def main():
    stop_event = Event()
    config.init_caps()
    vis = PositionVisualizer(stop_event, config.SUB_METADATA)
    app.run()


if __name__ == "__main__":
    main()
