import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes # type:ignore
import numpy as np
from mpl_toolkits.mplot3d import axes3d  # noqa: F401

from pytransform3d.plot_utils import Frame
from pytransform3d.rotations import passive_matrix_from_angle, R_id
from pytransform3d.transformations import transform_from, concat

import sys
from logging import getLogger, DEBUG, lastResort

import time
from multiprocessing import shared_memory, Event
import threading

from sympy import true
from dataset_generation.socket import XYZMetadata, XYZHandler
from dataset_generation.hand_tracking import HandManager, get_caps_ids, droid_src, Landmark
from dataset_generation.config import PUB_METADATA

logger = getLogger(__name__)

class HandPublisher:
    def __init__(self, stop_event: threading.Event, debug: bool = False):
        self.stop_event = stop_event

        self.xyz_handler = XYZHandler(PUB_METADATA)
        cap_ids = get_caps_ids()
        droid_ids = [droid_src(True)]
        droid_ids = []
        self.hand_manager = HandManager(cap_ids, droid_ids, debug=debug)

    def read_frames_to_buffer(self):
        logger.info("Start getting frames")
        latest_frames = None
        while not self.stop_event.is_set():
            if latest_frames is not None:
                self.set_data(latest_frames)
            self.xyz_handler.counter[0] += 1
            latest_frames: list[list[tuple[int, int, list[Landmark]]]] | None = self.hand_manager.get_latest_frames()

    def set_data(self, latest_frames: list[list[tuple[int, int, list[Landmark]]]]):
        last_row = 0
        for i in range(len(latest_frames)):
            # if not None then it should contain something
            for j in range(len(latest_frames[0])):
                l = self.landmarks_to_arrays(latest_frames[i][j][2])
                last_row = (i+j+1)*l.shape[0]
                self.xyz_handler.data[(i+j)*l.shape[0]:last_row, :] = l
        self.xyz_handler.data[last_row:, :] = 0.0

    @staticmethod
    def landmarks_to_arrays(landmarks: list[Landmark]):
        return np.array(list(map(lambda x: [x.x, x.y, x.z], landmarks)))

    def start(self):
        logger.info("Start visualization")
        self.hand_manager.start()
        self.t = threading.Thread(target=self.read_frames_to_buffer, daemon=True)
        self.t.start()

    def stop(self):
        self.hand_manager.stop()
        try:
            self.xyz_handler.close()   # <-- ensure unlink happens only here
        except Exception:
            pass

def main():
    stop_event = Event()
    hand_publisher = HandPublisher(stop_event, True)
    try:
        hand_publisher.start()
        while True:  # keep running until Ctrl+C
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Stopping...")
        hand_publisher.stop()     # calls xyz_handler.close()
        stop_event.set()


if __name__ == "__main__":
    main()
