# import os
# os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"  # sometimes helps
# os.environ["QT_QPA_PLATFORM"] = "offscreen"
import numpy as np
from PyQt5 import QtGui, QtWidgets
# fmt = QtGui.QSurfaceFormat()
# fmt.setRenderableType(QtGui.QSurfaceFormat.OpenGL)
# fmt.setVersion(3, 3)
# fmt.setProfile(QtGui.QSurfaceFormat.CoreProfile)
# fmt.setSwapBehavior(QtGui.QSurfaceFormat.DoubleBuffer)
# fmt.setRedBufferSize(8)
# fmt.setGreenBufferSize(8)
# fmt.setBlueBufferSize(8)
# fmt.setAlphaBufferSize(8)
# QtGui.QSurfaceFormat.setDefaultFormat(fmt)
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import threading
import sys
from logging import getLogger
from multiprocessing import Event

from dataset_generation import config
from dataset_generation.socket import XYZMetadata, XYZHandler
from dataset_generation.hand_to_pose import hand_to_points, hand_to_pose

logger = getLogger(__name__)


class PositionVisualizer:
    def __init__(self, stop_event: threading.Event, sub_metadata: XYZMetadata):
        self.xyz_handler = XYZHandler(sub_metadata)
        self.stop_event = stop_event
        self.sub_metadata = sub_metadata

        # Setup Qt window + OpenGL view
        self.app = QtWidgets.QApplication([])
        self.view = gl.GLViewWidget()
        self.view.setWindowTitle("Position Visualizer")
        self.view.setCameraPosition(distance=2.5, elevation=20, azimuth=45)
        self.view.show()

        # Add world axis
        world_axis = gl.GLAxisItem()
        world_axis.setSize(1, 1, 1)
        self.view.addItem(world_axis)

        DEBUG_POINTS = 30 # points for extra visualizations
        # One scatter plot for all points
        self.scatter = gl.GLScatterPlotItem(pos=np.zeros((self.sub_metadata.num_rows + DEBUG_POINTS, 3)),
                                            size=5,
                                            color=(1, 1, 1, 1),
                                            pxMode=True)
        self.view.addItem(self.scatter)

        # Thread to update data
        threading.Thread(target=self.update_positions, daemon=True).start()

        # Timer to refresh scatter plot
        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self.update_scatter)
        self.timer.start(30)  # ~33 FPS

    def update_positions(self):
        """Background: pull latest positions from shared memory."""
        count = 0
        while not self.stop_event.is_set():
            if count == self.xyz_handler.get_counter():
                continue
            # print("UPDATING POSITIONS")
            self.latest_points = self.xyz_handler.hands.copy()
            """(listeners, num hands per listener, 21, 3)"""
            count = self.xyz_handler.get_counter()
            # debug points
            for listener_idx in range(config.NUM_SRCS):
                for hand_idx in range(config.NUM_HANDS_PER_SRC):
                    hand_points = self.latest_points[listener_idx, hand_idx]
                    # TODO: visualize points
                    # extra_points1 = hand_to_points(hand_points)
                    # R, t = hand_to_pose(hand_points)
                    # xyz = np.eye(3) * 0.5
                    # final_point = R.T @ xyz + t
                    # print("BASE", t)
                    # extra_points = np.concatenate((extra_points1, final_point), axis=0)
                    # for k, extra_point in enumerate(extra_points):
                    #     self.latest_points[len(self.latest_points) - 1 - i * len(extra_points) - k] = extra_point

    def update_scatter(self):
        """GUI: redraw scatter plot."""
        if hasattr(self, "latest_points"):
            print(self.latest_points.reshape(-1, 3))
            self.scatter.setData(pos=self.latest_points.reshape(-1, 3))

    def run(self):
        sys.exit(self.app.exec_())


def main():
    stop_event = Event()
    # import os
    # os.environ["QT_QPA_PLATFORM"] = "offscreen"
    config.init_caps()
    vis = PositionVisualizer(stop_event, config.SUB_METADATA)
    vis.run()


if __name__ == "__main__":
    main()
