import numpy as np

from dataset_generation.socket import XYZMetadata


MAX_NUM_HANDS = 1
HAND_POINTS = 21
NUM_FRAMES = 100
PUB_SUB_PATH = "vis_pos"
PUB_METADATA, SUB_METADATA = XYZMetadata.create_pair(
    NUM_FRAMES,
    np.dtype(np.float64),
    PUB_SUB_PATH)
