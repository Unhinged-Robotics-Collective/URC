import numpy as np
import threading
from logging import getLogger
from multiprocessing import Event

import vispy
vispy.use("glfw")  # force GLFW backend, avoids Qt
from vispy import app, scene

from dataset_generation import config
from dataset_generation.socket import XYZMetadata, XYZHandler
from dataset_generation.deforms import umeyama_transform
from dataset_generation.hand_to_pose import hand_to_points, hand_to_pose

logger = getLogger(__name__)


class PositionVisualizer(scene.SceneCanvas):
    def __init__(self, stop_event: Event, sub_metadata: XYZMetadata):
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
        axis = scene.visuals.XYZAxis(parent=self.view.scene)

        # Scatter plot for points
        DEBUG_POINTS = 30
        num_points = self.sub_metadata.num_rows
        self.scatter = scene.visuals.Markers(parent=self.view.scene)
        self.scatter.set_data(np.zeros((num_points, 3)),
                              face_color=(1, 1, 1, 1),
                              size=5)

        # Background thread for updating positions
        self.latest_points = np.random.rand(num_points, 3)
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
            print("LATEST POINTS", self.xyz_handler.hands.min(), self.xyz_handler.hands.max())
            # self.latest_points = self.xyz_handler.hands.copy().reshape(-1, 3)
            count = self.xyz_handler.get_counter()
            # debug points expansion would go here

    def update_scatter(self, event):
        """GUI: redraw scatter plot."""
        if self.latest_points is not None:
            if self.count == self.xyz_handler.get_counter():
                return
            # print("LATEST POINTS", self.xyz_handler.hands.shape)
            # print("LATEST POINTS", self.xyz_handler.hands.shape, self.xyz_handler.hands.min(), self.xyz_handler.hands.max())
            try:
                hands1 = self.xyz_handler.hands[0][0]
                hands2 = self.xyz_handler.hands[1][0]
                new_hands1 = umeyama_transform(hands1.T, hands2.T)
                # print("new hands", new_hands1)
                self.latest_points = self.xyz_handler.hands.copy()
                print("copied")
                self.latest_points[0][0] = new_hands1.T
                print("asigned")
                self.count = self.xyz_handler.get_counter()
                print("updated counter")
                self.scatter.set_data(self.latest_points.reshape(-1, 3), face_color=(1, 1, 1, 1), size=5)
                print("setted data")

            except Exception as e:
                print(e)


def main():
    stop_event = Event()
    config.init_caps()
    vis = PositionVisualizer(stop_event, config.SUB_METADATA)
    app.run()


if __name__ == "__main__":
    main()
