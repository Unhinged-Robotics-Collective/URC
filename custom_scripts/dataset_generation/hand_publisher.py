import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes # type:ignore
import numpy as np
from mpl_toolkits.mplot3d import axes3d  # noqa: F401

from pytransform3d.plot_utils import Frame
from pytransform3d.rotations import passive_matrix_from_angle, R_id
from pytransform3d.transformations import transform_from, concat

import sys
from logging import getLogger

import time
import multiprocessing
from multiprocessing import Event
import threading

from dataset_generation.socket import XYZMetadata, XYZHandler
from dataset_generation.hand_tracking import HandManager, Landmark, Hands
from dataset_generation import config
import logging

logging.basicConfig(
    level=logging.DEBUG,               # Show DEBUG and above
    format="%(asctime)s [%(levelname)s] %(message)s",
    filename="app.log",   # Write logs to a file
    filemode="w",          # Overwrite each run (use "a" to append)
)
logger = getLogger(__name__)


class HandPublisher:
    def __init__(self, stop_event: threading.Event, debug: bool = False):
        self.stop_event = stop_event

        self.xyz_handler = XYZHandler(config.PUB_METADATA)
        self.hand_manager = HandManager(config.CAP_IDS, config.DROID_IDS, debug=debug)
        self.point_state = np.zeros(self.xyz_handler.hands.shape, dtype=config.PUB_METADATA.dtype)
        logger.info("Initialized hand publisher")

    def read_frames_to_buffer(self):
        logger.info("Start getting frames")
        latest_frames = None
        while not self.stop_event.is_set():
            if latest_frames is not None:
                self.set_data(latest_frames)
            self.xyz_handler.update_counter()
            latest_frames = self.hand_manager.get_latest_frames()

    def set_data(self, latest_frames: list[Hands], lerp: float = 0.2):
        assert 0.0 <= lerp < 1.0, f"lerp: {lerp}"
        for hands in latest_frames:
            hand_counter = 0
            for hand in hands.landmarks:
                l = self.landmarks_to_arrays(hand)
                logger.debug("hands: %s listener: %d hand: %d", self.xyz_handler.hands.shape, hands.listener_id, hand_counter)
                self.xyz_handler.hands[hands.listener_id, hand_counter] = self.lerp(lerp, self.xyz_handler.hands[hands.listener_id, hand_counter], l)
                hand_counter += 1
            # self.xyz_handler.hands[hands.listener_id, hand_counter:, :] = 0.0
        # print(self.xyz_handler.hands.shape)

    @staticmethod
    def lerp(lerp: float, arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
        if lerp > 0.0:
            return lerp * arr1 + (1.0 - lerp) * arr2
        return arr2

    @staticmethod
    def landmarks_to_arrays(landmarks: list[Landmark]) -> np.ndarray:
        """Returns: np array (21, 3)"""
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
    config.init_caps()
    multiprocessing.set_start_method("spawn", force=True)
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
