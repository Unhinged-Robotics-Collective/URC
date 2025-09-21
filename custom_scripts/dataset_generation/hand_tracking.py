import multiprocessing
from multiprocessing.context import SpawnContext, SpawnProcess
import time
import math
from dataclasses import dataclass, field
from turtle import listen
import numpy as np
import multiprocessing as mp
# from multiprocessing import Queue
from typing import TYPE_CHECKING, Callable, Iterable, Any
from dataset_generation.handle_processes import get_handler
from dataset_generation import config

import cv2
import mediapipe
import queue as _queue  # for Empty
from queue import LifoQueue, Queue, Full, Empty
import threading


@dataclass
class Hands:
    listener_id: int = -1
    landmarks: list[list["Landmark"]] = field(default_factory=list)


class Landmark:
    x: float
    y: float
    z: float


class LandmarkListener:

    def __init__(self, input_queue: "mp.Queue[list[list[Landmark]]]"):
        self.q: Queue[list[list[Landmark]]] = Queue(maxsize=config.NUM_HANDS_PER_SRC)
        self.t = threading.Thread(target=self.transfer_landmark, args=(input_queue, ), daemon=True)

    def start(self):
        self.t.start()

    def transfer_landmark(self, q_o: "mp.Queue[list[list[Landmark]]]"):
        while True:
            try:
                res: list[list[Landmark]] = q_o.get(timeout=1)
                try:
                    self.q.put_nowait(res)
                except Full as e:
                    pass
            except _queue.Empty:
                pass


class KeypointArgs(dict):
    ctx: SpawnContext
    stop_event: threading.Event  # actually mp.Event
    q_o: "mp.Queue[list[list[Landmark]]]"
    src: int | str
    debug: bool

    def __getattr__(self, item: str):
        return self[item]


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
            kwarg = KeypointArgs(ctx=ctx, stop_event=self.stop_event, q_o=ctx.Queue(maxsize=config.NUM_HANDS_PER_SRC), src=cam_src, debug=debug)
            self.keypoint_kwargs.append(kwarg)
            self.keypoint_processes.append(get_handler(process_keypoints, **kwarg))
            # create thread listeners
            self.listeners.append(LandmarkListener(kwarg.q_o))

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

    def get_latest_frames(self) -> list[Hands] | None:
        ret = []
        for k, listener in enumerate(self.listeners):
            try:
                h = Hands(listener_id=k, landmarks=listener.q.get_nowait())
                # Delete because older positions shouldn't survive
                listener.q.queue.clear()
                ret.append(h)
            except Empty as e: pass
        return ret

    def stop(self):
        self.stop_event.set()
        for p in self.keypoint_processes: p.join()


def process_keypoints(stop_event: threading.Event, q_o: "mp.Queue[list[list[Landmark]]]", src: str | int, debug=False, **_):
    """stop_event is actually a mp.Event but .pyi annotations are wrong"""
    mp_hands = mediapipe.solutions.hands # type: ignore
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=config.NUM_HANDS_PER_SRC, # TODO: this should be divided by number of cameras
        model_complexity=0,       # lighter model
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )
    mpDraw = mediapipe.solutions.drawing_utils # type: ignore

    cap = cv2.VideoCapture(src)
    # cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # might be ignored by some backends
    win_name = f"Debug source {src}"
    frame_id = 0
    try:
        while not stop_event.is_set():
            ok, img = cap.read()
            if not ok:
                break
            # frame_id += 1
            # if frame_id % 2 == 0:
            #     continue
            img = cv2.flip(img, 1)
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(imgRGB)

            if results.multi_hand_landmarks:
                hand_marks = list(results.multi_hand_landmarks)[:config.NUM_HANDS_PER_SRC]
                try:
                    q_o.put_nowait([list(handLms.landmark) for handLms in hand_marks])
                except Full as e:
                    pass
                if debug:
                    for handLms in hand_marks:
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
