# consumer.py
from sysconfig import is_python_build
import numpy as np
import threading
from multiprocessing import shared_memory, Process, Event
import time
from dataclasses import dataclass
from dataset_generation.handle_processes import get_handler
import signal
import sys

from logging import getLogger

logger = getLogger(__name__)

@dataclass
class XYZMetadata:
    """XYZ metadata. There might be some assumptions you need to make to parse it"""
    num_rows: int
    """Number of rows in total"""
    num_listeners: int
    """Number of listeners/data sources"""
    num_hands_per_listener: int
    """Number of hands to publish per listener"""
    hand_points: int
    """Number of points per hand"""
    debug_points: int
    """Number of debug points"""
    dtype: np.dtype
    """Numpy dtype"""
    data_path: str
    """Path to the actual data"""
    is_publisher: bool
    """Publisher creates shared memory, subscriber reads"""

    @classmethod
    def create_pair(cls, num_listeners: int, num_hands_per_listener: int, hand_points: int, debug_points: int, dtype: np.dtype, data_path: str) -> tuple["XYZMetadata", "XYZMetadata"]:
        num_rows: int = num_listeners * num_hands_per_listener * hand_points + debug_points
        return (
            cls(num_rows, num_listeners, num_hands_per_listener, hand_points, debug_points, dtype, data_path, True),
            cls(num_rows, num_listeners, num_hands_per_listener, hand_points, debug_points, dtype, data_path, False)
        )


# socket.py
import numpy as np
from multiprocessing import shared_memory
from multiprocessing import resource_tracker  # <-- add this

class XYZHandler:
    def __init__(self, metadata: XYZMetadata):
        if metadata.is_publisher:
            # Best-effort cleanup of stale segment from previous run
            try:
                old = shared_memory.SharedMemory(name=metadata.data_path)
                old.unlink()
                old.close()
            except FileNotFoundError:
                pass

        self.metadata = metadata
        self.shm = shared_memory.SharedMemory(
            name=metadata.data_path,
            create=metadata.is_publisher,
            # 1 64bit counter, then number of xyz points
            size=8 + metadata.num_rows * 3 * metadata.dtype.itemsize
        )
        logger.info("shared memory %s", self.shm)

        # IMPORTANT: do NOT let subscribers unlink on exit
        if not metadata.is_publisher:
            try:
                # resource_tracker registers objects by their raw name
                resource_tracker.unregister(self.shm._name, 'shared_memory')  # type: ignore[attr-defined]
            except Exception:
                pass
        pointer = 0
        self._counter = np.ndarray((1,), dtype=np.int64, buffer=self.shm.buf[:8])
        pointer += 8

        hand_data_size = 3 * metadata.num_listeners * metadata.hand_points * metadata.num_hands_per_listener * metadata.dtype.itemsize
        # [hand_id, point_id, x/y/z]
        self.hands = np.ndarray((metadata.num_listeners, metadata.num_hands_per_listener, metadata.hand_points, 3),
                               dtype=metadata.dtype,
                               buffer=self.shm.buf[pointer:pointer + hand_data_size])
        """(listeners, num hands per listener, 21, 3)"""
        pointer += hand_data_size

        debug_data_size = 3 * metadata.debug_points * metadata.dtype.itemsize
        self.debug_points = np.ndarray((metadata.debug_points, 3),
                               dtype=metadata.dtype,
                               buffer=self.shm.buf[pointer:pointer + debug_data_size])
        pointer += debug_data_size
        assert not (
            np.shares_memory(self._counter, self.hands)
            or np.shares_memory(self._counter, self.debug_points)
            or np.shares_memory(self.hands, self.debug_points))

    def update_counter(self) -> None:
        self._counter[0] += 1

    def get_counter(self) -> np.int64:
        return self._counter[0]

    def close(self):
        """Close the local handle. Publisher additionally unlinks."""
        try:
            self.shm.close()
        finally:
            if self.metadata.is_publisher:
                # Tear down the named segment so new subscribers can't attach
                try:
                    self.shm.unlink()
                except FileNotFoundError:
                    pass
