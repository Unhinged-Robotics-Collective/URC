import numpy as np
from dataclasses import dataclass, field

from dataset_generation.socket import XYZMetadata
import logging

logger = logging.getLogger(__name__)

# TODO: split shared memory to have a region for each hand so each region can be written to and read from independently, so we don't have to use locks
# hand z axis is estimated using the distance between joints on each finger.
# when I move the hand to cover the whole screen, the largest distance is ~1.8 or so
# so to have distance at camera position be == 0, we have to inverse the distance
MAX_HAND_SIZE = 1.85
NORMALIZATION_CONSTANT = 8
DIST_EXPONENT = 1.15

MIXIN_DISTANCE = True
USE_DROID = False
DEBUG = True
NUM_HANDS_PER_SRC = 1

IMAGE_SIZE = (640, 480)

HAND_POINTS = 21
NUM_FRAMES = 100
HAND_DEBUG_POINTS = 100
PUB_SUB_PATH = "vis_pos"
CAMS = ()

@dataclass
class CamInfo:
    w: float = IMAGE_SIZE[0]
    h: float = IMAGE_SIZE[1]
    x0: float = -1
    y0: float = -1
    f: float = -1
    R: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=float))
    t: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))

    def __post_init__(self):
        if self.x0 < 0:
            self.x0 = self.w / 2.0
        if self.y0 < 0:
            self.y0 = self.h / 2.0

    def unnormalize(self, arr: np.ndarray) -> np.ndarray:
        """Accepts: (N, 2)"""
        assert arr.shape[1] == 2
        assert np.min(arr) >= -0.1, f"min: {np.min(arr)}"
        assert np.max(arr) <= 1.1, f"max: {np.max(arr)}"
        print("arr", np.min(arr), np.max(arr))
        return np.concatenate([arr[:, :1] * self.w, arr[:, 1:2] * self.h], axis=1)


def init_caps():
    global CAP_IDS, DROID_IDS, USE_DROID, PUB_METADATA, SUB_METADATA, NUM_SRCS, SRCS, CAMS
    import cv2
    def get_caps_ids() -> list[int]:
        ids = []
        idx = 0
        while True:
            cap = cv2.VideoCapture(idx)
            ok, _ = cap.read()
            cap.release()
            if not ok:
                break
            ids.append(idx)
            idx += 1
        return ids


    def droid_src(use_usb: bool) -> str:
        return f'http://127.0.0.1:4747/video?{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}' if use_usb else f'http://192.168.0.95:4747/video?{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}'

    CAP_IDS = get_caps_ids()
    DROID_IDS = [droid_src(True)] if USE_DROID else []
    SRCS = CAP_IDS + DROID_IDS
    NUM_SRCS = max(len(SRCS), len(DROID_IDS) + 1) # at least one camera must be present
    CAMS = tuple(CamInfo() for _ in range(NUM_SRCS))

    PUB_METADATA, SUB_METADATA = XYZMetadata.create_pair(
        NUM_SRCS,
        NUM_HANDS_PER_SRC,
        HAND_POINTS,
        HAND_DEBUG_POINTS,
        np.dtype(np.float64),
        PUB_SUB_PATH)
    print("NUM SRCS", NUM_SRCS)
    logger.info("PUB %s SUB %s", PUB_METADATA, SUB_METADATA)
