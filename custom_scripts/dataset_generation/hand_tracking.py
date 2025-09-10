import multiprocessing
from multiprocessing.context import SpawnContext, SpawnProcess
import time
import math
from dataclasses import dataclass
import numpy as np
import multiprocessing as mp
from multiprocessing import Queue
from typing import TYPE_CHECKING, Callable, Iterable, Any
from dataset_generation.handle_processes import get_handler

import cv2
import mediapipe
import queue as _queue  # for Empty
from queue import LifoQueue
import threading


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

class LandmarkListener:

    def __init__(self, input_queue: mp.Queue):
        self.q = LifoQueue()
        self.t = threading.Thread(target=self.transfer_landmark, args=(input_queue, ), daemon=True)

    def start(self):
        self.t.start()

    def transfer_landmark(self, q_o: mp.Queue):
        while True:
            try:
                self.q.put_nowait(q_o.get(timeout=1))
            except _queue.Empty:
                pass


class KeypointArgs(dict):
    ctx: SpawnContext
    stop_event: threading.Event  # actually mp.Event
    q_o: mp.Queue
    src: int | str
    debug: bool


class HandManager:
    def __init__(self, webcam_ids: list[int], webcam_addresses: list[str], debug: bool = False):
        self.webcam_ids: list[int] = webcam_ids
        self.webcam_addresses: list[str] = webcam_addresses
        self.all_cams: list[str | int] = self.webcam_ids + self.webcam_addresses

        # spawn processes
        ctx: SpawnContext = mp.get_context("spawn")  # single context everywhere

        # get_cap_ids, droid_src
        self.keypoint_kwargs: list[KeypointArgs] = []
        self.keypoint_processes: list[mp.Process | SpawnProcess] = []
        self.listeners: list[LandmarkListener] = []
        self.stop_event = mp.Event()
        for cam_src in self.all_cams:
            kwarg = KeypointArgs(ctx=ctx, stop_event=self.stop_event, q_o=ctx.Queue(), src=cam_src, debug=debug)
            self.keypoint_kwargs.append(kwarg)
            self.keypoint_processes.append(get_handler(process_keypoints, **kwarg))
            # create thread listeners
            self.listeners.append(LandmarkListener(kwarg["q_o"]))

        self.monitor_thread = threading.Thread(
            target=self._monitor_processes, daemon=True)

    def _monitor_processes(self):
        try:
            while not self.stop_event.is_set():
                for p in self.keypoint_processes:
                    if not p.is_alive():
                        print(f"[WATCHDOG] Process {p.name} died, shutting down...")
                        self.stop_event.set()
                        return
                time.sleep(0.5)
        except Exception as e:
            print("[WATCHDOG] Exception in monitor:", e)
            self.stop_event.set()


    def start(self):
        for proc in self.listeners: proc.start()
        for proc in self.keypoint_processes: proc.start()
        self.monitor_thread.start()

    def stop(self):
        self.stop_event.set()
        for p in self.keypoint_processes: p.join()

def process_keypoints(stop_event: threading.Event, q_o: mp.Queue, src: str | int, debug=False, **_):
    """stop_event is actually a mp.Event but .pyi annotations are wrong"""
    mp_hands = mediapipe.solutions.hands # type: ignore
    hands = mp_hands.Hands()
    mpDraw = mediapipe.solutions.drawing_utils # type: ignore

    cap = cv2.VideoCapture(src)
    win_name = f"Debug source {src}"

    try:
        while not stop_event.is_set():
            ok, img = cap.read()
            if not ok:
                break

            img = cv2.flip(img, 1)
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(imgRGB)

            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    lms = list(handLms.landmark)
                    q_o.put_nowait(lms)
                    if debug:
                        mpDraw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

            if debug:
                cv2.imshow(win_name, img)
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break
    finally:
        print("EXITING", src)
        try:
            q_o.cancel_join_thread()
        except: ...
        try:
            q_o.close()
        except: ...
        cap.release()
        if debug:
            cv2.destroyWindow(win_name)
