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


class XYZHandler:
    def __init__(self, metadata: XYZMetadata):
        if metadata.is_publisher:
                try:
                    old = shared_memory.SharedMemory(name=metadata.data_path)
                    old.unlink()   # delete the segment
                    old.close()
                except: ...
        self.metadata = metadata
        self.shm = shared_memory.SharedMemory(name=metadata.data_path, create=metadata.is_publisher, size=8 + metadata.num_rows * 3 * metadata.dtype.itemsize)
        self.counter = np.ndarray((1,), dtype=np.int64, buffer=self.shm.buf[:8])
        self.data = np.ndarray((metadata.num_rows, 3), dtype=metadata.dtype, buffer=self.shm.buf[8:])
