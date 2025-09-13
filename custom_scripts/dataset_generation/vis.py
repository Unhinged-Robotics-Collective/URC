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
    def __init__(self, stop_event: threading.Event, sub_metadata: XYZMetadata, show_window: bool = True):
        self.xyz_handler = XYZHandler(sub_metadata)
        self.stop_event = stop_event
        self.sub_metadata = sub_metadata
        self.transforms = np.zeros((self.sub_metadata.num_rows, 4, 4))

        if show_window:
            fig = plt.figure(figsize=(9, 9))
            ax: Axes = fig.add_subplot(111, projection="3d")
            ax.set_xlim((-10, 10))
            ax.set_ylim((-10, 10))
            ax.set_zlim((-10, 10)) # type: ignore[attr-defined]
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
            )

    def update_frames(self, step: int, frames: list[Frame]):
        """Update frame positions from self.transforms.

        Update self.transforms on a separate thread/listener"""
        # progress = float(step + 1) / float(n_frames)
        # H = np.eye(4)
        # H0 = transform_from(R_id, np.zeros(3))
        # H_mod = np.eye(4)
        # H0[:3, 3] = np.array([progress, 0, progress])
        # H_mod[:3, :3] = passive_matrix_from_angle(2, 8 * np.pi * progress)
        # H = concat(H0, H_mod)
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

            self.transforms[:, :3, 3] = self.xyz_handler.data[:self.sub_metadata.num_rows] * 10
            count = self.xyz_handler.counter[0]

    def start(self):
        self.t = threading.Thread(target=self.update_positions, daemon=True)
        self.t.start()


class PubSubVis:
    def __init__(self, stop_event: threading.Event, num_frames: int = 100, show_window: bool = True):
        self.stop_event = stop_event
        num_frames = 100
        pub_sub_path = "vis_pos"
        pub_metadata, sub_metadata = XYZMetadata.create_pair(
                num_frames,
                np.dtype(np.float64),
                pub_sub_path)
        self.xyz_handler = XYZHandler(pub_metadata)
        self.position_visualizer = PositionVisualizer(stop_event, sub_metadata, show_window)
        cap_ids = get_caps_ids()
        droid_ids = [droid_src(True)]
        droid_ids = []
        self.hand_manager = HandManager(cap_ids, droid_ids)
        # thread 1 get frames from hand manager
        # thread 2 set frames in shared memory

        # publish frames
        # start reading frames
        # start visualizing frames

    def read_frames_to_buffer(self):
        logger.info("Start getting frames")
        while not self.stop_event.is_set():
            latest_frames = self.hand_manager.get_latest_frames()
            if latest_frames is None: continue
            print("latest frames", self.xyz_handler.counter[0])
            # print(len(latest_frames))
            l = self.landmarks_to_arrays(latest_frames[0])
            # print(l)
            # print(list(map(lambda x: int(x[1]), latest_frames[0][0].ListFields())))
            # print(latest_frames[0][0].DESCRIPTOR.fields[0].value)
            # print(latest_frames[0][0].DESCRIPTOR.fields[0].number)
            self.xyz_handler.data[:l.shape[0], :] = l
            # print(self.xyz_handler.data[:l.shape[0], :])
            self.xyz_handler.counter[0] += 1

    @staticmethod
    def landmarks_to_arrays(landmarks: list[Landmark]):
        return np.array(list(map(lambda x: [x.x, x.y, x.z], landmarks)))

    def start(self):
        logger.info("Start visualization")
        self.hand_manager.start()
        self.t = threading.Thread(target=self.read_frames_to_buffer, daemon=True)
        self.t.start()
        self.position_visualizer.start()

    def stop(self):
        self.hand_manager.stop()

# if __name__ == "__main__":
#     stop_event = Event()
#     psv = PubSubVis(stop_event)
#     psv.start()
#     time.sleep(10)
#     stop_event.set()
#     psv.stop()
if __name__ == "__main__":
    stop_event = Event()
    psv = PubSubVis(stop_event, show_window=True)
    psv.start()                         # starts background worker threads
    time.sleep(1)

    plt.show(block=False)               # don't block the main thread
    try:
        while not stop_event.is_set():
            plt.pause(0.05)             # let GUI process events
    finally:
        psv.stop()
        plt.close("all")
