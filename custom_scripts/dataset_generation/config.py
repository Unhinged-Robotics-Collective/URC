import numpy as np

from dataset_generation.socket import XYZMetadata
import logging

logger = logging.getLogger(__name__)

# TODO: split shared memory to have a region for each hand so each region can be written to and read from independently, so we don't have to use locks

USE_DROID = True
DEBUG = True
NUM_HANDS_PER_SRC = 1


HAND_POINTS = 21
NUM_FRAMES = 100
HAND_DEBUG_POINTS = 100
PUB_SUB_PATH = "vis_pos"


def init_caps():
    global CAP_IDS, DROID_IDS, USE_DROID, PUB_METADATA, SUB_METADATA, NUM_SRCS, SRCS
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
        return 'http://127.0.0.1:4747/video?640x480' if use_usb else 'http://192.168.0.95:4747/video?640x480'

    CAP_IDS = get_caps_ids()
    DROID_IDS = [droid_src(True)] if USE_DROID else []
    SRCS = CAP_IDS + DROID_IDS
    NUM_SRCS = max(len(SRCS), len(DROID_IDS) + 1) # at least one camera must be present

    PUB_METADATA, SUB_METADATA = XYZMetadata.create_pair(
        NUM_SRCS,
        NUM_HANDS_PER_SRC,
        HAND_POINTS,
        HAND_DEBUG_POINTS,
        np.dtype(np.float64),
        PUB_SUB_PATH)
    print("NUM SRCS", NUM_SRCS)
    logger.info("PUB %s SUB %s", PUB_METADATA, SUB_METADATA)