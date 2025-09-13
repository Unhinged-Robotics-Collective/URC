import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes # type:ignore
import numpy as np
from mpl_toolkits.mplot3d import axes3d  # noqa: F401

from pytransform3d.plot_utils import Frame
from pytransform3d.rotations import passive_matrix_from_angle, R_id
from pytransform3d.transformations import transform_from, concat

import sys
from logging import getLogger, DEBUG

import time
from multiprocessing import shared_memory, Event
import threading

from sympy import true
from dataset_generation.socket import XYZMetadata, XYZHandler
from dataset_generation.hand_tracking import HandManager, get_caps_ids, droid_src, Landmark

logger = getLogger(__name__)


class PositionVisualizer:
    def __init__(self, stop_event: threading.Event, sub_metadata: XYZMetadata):
        self.xyz_handler = XYZHandler(sub_metadata)
        self.stop_event = stop_event
        self.sub_metadata = sub_metadata
        self.transforms = np.zeros((self.sub_metadata.num_rows, 4, 4))

        fig = plt.figure(figsize=(9, 9))
        ax: Axes = fig.add_subplot(111, projection="3d")
        ax.set_xlim((-1, 1))
        ax.set_ylim((-1, 1))
        ax.set_zlim((-1, 1)) # type: ignore[attr-defined]
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z") # type: ignore[attr-defined]
        frames: list[Frame] = []
        for i in range(self.sub_metadata.num_rows):
            self.transforms[i] = np.eye(4)
            frame = Frame(self.transforms[i], s=0.2)
            frame.add_frame(ax) # type: ignore
            frames.append(frame)

        self.anim = animation.FuncAnimation(
            fig,
            self.update_frames, # type: ignore
            None,
            fargs=(frames, ),
            interval=50,
            blit=False,
            cache_frame_data=False,
        )
        threading.Thread(target=self.update_positions, daemon=True).start()

    def update_frames(self, step: int, frames: list[Frame]):
        """Update frame positions from self.transforms.

        Update self.transforms on a separate thread/listener"""
        logger.info("Updating frames")
        for i in range(self.sub_metadata.num_rows):
            frames[i].set_data(self.transforms[i])
        return frames

    def update_positions(self):
        logger.debug("Updating positions")
        count = 0
        while not self.stop_event.is_set():
            if count == self.xyz_handler.counter[0]: continue
            logger.debug("Updating positions")
            self.transforms[:, :3, 3] = self.xyz_handler.data[:self.sub_metadata.num_rows]
            count = self.xyz_handler.counter[0]


def main():
    stop_event = Event()
    num_frames = 100
    pub_sub_path = "vis_pos"
    _, sub_metadata = XYZMetadata.create_pair(
            num_frames,
            np.dtype(np.float64),
            pub_sub_path)
    position_visualizer = PositionVisualizer(stop_event, sub_metadata)
    plt.show()

if __name__ == "__main__":
    main()
