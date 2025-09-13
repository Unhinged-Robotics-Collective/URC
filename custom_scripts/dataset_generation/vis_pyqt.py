import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt5 import QtWidgets
import threading
import sys
from logging import getLogger
from multiprocessing import Event

from dataset_generation.config import SUB_METADATA
from dataset_generation.socket import XYZMetadata, XYZHandler

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

        # One scatter plot for all points
        self.scatter = gl.GLScatterPlotItem(pos=np.zeros((self.sub_metadata.num_rows, 3)),
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
            if count == self.xyz_handler.counter[0]:
                continue
            self.latest_points = self.xyz_handler.data[:self.sub_metadata.num_rows] #.copy()
            count = self.xyz_handler.counter[0]

    def update_scatter(self):
        """GUI: redraw scatter plot."""
        if hasattr(self, "latest_points"):
            self.scatter.setData(pos=self.latest_points)

    def run(self):
        sys.exit(self.app.exec_())


def main():
    stop_event = Event()
    vis = PositionVisualizer(stop_event, SUB_METADATA)
    vis.run()


if __name__ == "__main__":
    main()
