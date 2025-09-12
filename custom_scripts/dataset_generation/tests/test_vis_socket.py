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
