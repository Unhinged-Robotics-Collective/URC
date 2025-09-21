from sysconfig import is_python_build
import numpy as np
import threading
from multiprocessing import shared_memory, Process, Event
import time
from dataclasses import dataclass
from dataset_generation.handle_processes import get_handler
import signal
import sys


from dataset_generation.socket import XYZMetadata, XYZHandler


def generate_data(metadata: XYZMetadata, stop_event: threading.Event):
    try:
        old = shared_memory.SharedMemory(name=metadata.data_path)
        old.unlink()   # delete the segment
        old.close()
    except: ...
    xyz_handler = XYZHandler(metadata)
    while not stop_event.is_set():
        r = np.random.randn(metadata.num_rows, 3)
        xyz_handler.hands[:] = r
        xyz_handler._counter[0] += 1
        time.sleep(0.1)


def read_data(metadata: XYZMetadata, stop_event: threading.Event):
    xyz_handler = XYZHandler(metadata)
    counter = 0
    while not stop_event.is_set():
        if counter == xyz_handler._counter[0]: continue
        print(xyz_handler.hands)
        counter = xyz_handler._counter[0]


if __name__ == "__main__":
    end = Event()
    pm, sm = XYZMetadata.create_pair(2, np.dtype(np.float64), "pubsub_test")
    pub = Process(target=generate_data, args=(pm, end), daemon=True)
    sub = Process(target=read_data, args=(sm, end), daemon=True)
    pub.start()
    time.sleep(1)
    sub.start()

    pub.join()
    sub.join()

"""
Plan:

process 1: queue that gets hand data
process 2: read latest position data => set frame for vis
process 3: get positional data, transform it
"""
