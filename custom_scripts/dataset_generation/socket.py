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


@dataclass
class XYZMetadata:
    """XYZ metadata. There might be some assumptions you need to make to parse it"""
    num_rows: int
    """Number of rows in total"""
    dtype: np.dtype
    """Numpy dtype"""
    data_path: str
    """Path to the actual data"""
    is_publisher: bool
    """Publisher creates shared memory, subscriber reads"""

    @classmethod
    def create_pair(cls, num_rows: int, dtype: np.dtype, data_path: str) -> tuple["XYZMetadata", "XYZMetadata"]:
        return cls(num_rows, dtype, data_path, True), cls(num_rows, dtype, data_path, False)


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
            size=8 + metadata.num_rows * 3 * metadata.dtype.itemsize
        )

        # IMPORTANT: do NOT let subscribers unlink on exit
        if not metadata.is_publisher:
            try:
                # resource_tracker registers objects by their raw name
                resource_tracker.unregister(self.shm._name, 'shared_memory')
            except Exception:
                pass

        self.counter = np.ndarray((1,), dtype=np.int64, buffer=self.shm.buf[:8])
        self.data = np.ndarray((metadata.num_rows, 3),
                               dtype=metadata.dtype,
                               buffer=self.shm.buf[8:])

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
