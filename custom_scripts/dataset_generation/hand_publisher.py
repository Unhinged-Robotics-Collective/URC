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


from multiprocessing.synchronize import Event as EventType
from dataset_generation.train_interpolator import train_model
from sklearn.pipeline import Pipeline

logging.basicConfig(
    level=logging.DEBUG,               # Show DEBUG and above
    format="%(asctime)s [%(levelname)s] %(message)s",
    filename="app.log",   # Write logs to a file
    filemode="w",          # Overwrite each run (use "a" to append)
)
logger = getLogger(__name__)


class HandPublisher:
    def __init__(self, stop_event: EventType, model: Pipeline | None, debug: bool = False):
        self.stop_event = stop_event

        self.xyz_handler = XYZHandler(config.PUB_METADATA)
        self.hand_manager = HandManager(config.CAP_IDS, config.DROID_IDS, stop_event=stop_event, debug=debug)
        self.point_state = np.zeros(self.xyz_handler.hands.shape, dtype=config.PUB_METADATA.dtype)
        logger.info("Initialized hand publisher")
        self.model = model

    def read_frames_to_buffer(self):
        logger.info("Start getting frames")
        latest_frames = None
        while not self.stop_event.is_set():
            if latest_frames is not None:
                self.set_data(latest_frames)
            self.xyz_handler.update_counter()
            latest_frames = self.hand_manager.get_latest_frames()

    @staticmethod
    def mix_in_distance(xyz: np.ndarray, scale=10.0):
        assert xyz.shape[0] == 21, f"{xyz=}"
        def segment_dist(x: np.ndarray):
            return np.mean(np.sqrt(np.sum(np.diff(x, axis=0)**2, axis=1)))
        thumb_dists = segment_dist(xyz[1:5, 0:2])
        index_dists = segment_dist(xyz[5:9, 0:2])
        middle_dists = segment_dist(xyz[9:13, 0:2])
        ring_dists = segment_dist(xyz[13:17, 0:2])
        pinky_dists = segment_dist(xyz[17:21, 0:2])
        return config.TOTAL_SCALE * (config.MAX_HAND_SIZE / (scale * max(thumb_dists, index_dists, middle_dists, ring_dists, pinky_dists))) ** config.DIST_EXPONENT

    def set_data(self, latest_frames: list[Hands], lerp: float = 0.5):
        assert 0.0 <= lerp < 1.0, f"lerp: {lerp}"
        for hands in latest_frames:
            hand_counter = 0
            for hand in hands.landmarks:
                l = self.landmarks_to_arrays(hand)
                pred = 0.0
                if self.model:
                    pred = self.model.predict(l.reshape(1, -1))[0]
                    print("Predicted distance", float(pred))
                    l[:, 2] += float(pred)
                hand_vals = self.lerp(lerp, self.xyz_handler.hands[hands.listener_id, hand_counter], l)
                self.xyz_handler.hands[hands.listener_id, hand_counter] = hand_vals
                if config.MIXIN_DISTANCE:
                    dist = self.mix_in_distance(hand_vals)
                    self.xyz_handler.hands[hands.listener_id, hand_counter, :, 2] += dist
                    xy_mean = np.mean(hand_vals[:, :2], axis=0) * dist
                    self.xyz_handler.hands[hands.listener_id, hand_counter, :, :2] = (hand_vals[:, :2] - xy_mean) * 2*dist + xy_mean
                    print("MIXIN DIST", dist)
                hand_counter += 1

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
        print("before hand manager stop")
        self.hand_manager.stop()
        print("after hand manager stop")
        try:
            self.xyz_handler.close()   # <-- ensure unlink happens only here
            print("after handler stop")
        except Exception:
            pass
        print("after exception stop")


def _parse_args():
    import argparse
    parser = argparse.ArgumentParser(
                    prog='Hand position publisher',
                    description='Visualize hand positions')
    parser.add_argument('--train-file', required=False, help="File based on which to train distance estimator")
    return parser.parse_args()

def main():
    args = _parse_args()
    print(args)
    model = None
    if args.train_file:
        model = train_model(args.train_file)
    config.init_caps()
    multiprocessing.set_start_method("spawn", force=True)
    stop_event = Event()
    hand_publisher = HandPublisher(stop_event, model=model, debug=True)
    try:
        hand_publisher.start()
        while True:  # keep running until Ctrl+C
            time.sleep(0.5)
            if stop_event.is_set():
                break
    except KeyboardInterrupt:
        print("Stopping...")
        hand_publisher.stop()     # calls xyz_handler.close()
        stop_event.set()


if __name__ == "__main__":
    main()
