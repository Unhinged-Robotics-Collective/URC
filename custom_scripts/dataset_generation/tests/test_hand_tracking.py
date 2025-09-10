import multiprocessing
from multiprocessing.context import SpawnContext, SpawnProcess
import time
import math
from dataclasses import dataclass
import numpy as np
import multiprocessing as mp
from multiprocessing import Queue
from typing import TYPE_CHECKING, Callable, Iterable, Any

import cv2
import mediapipe
import queue as _queue  # for Empty
from queue import Queue, LifoQueue

import threading

from dataset_generation.handle_processes import get_handler
from dataset_generation.hand_tracking import HandManager, process_keypoints

USE_DROID = False
USE_WEBCAM = True
USE_USB = True

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

def droid_src() -> str:
    return 'http://127.0.0.1:4747/video?640x480' if USE_USB else 'http://192.168.0.95:4747/video?640x480'

# ---------- worker must be top-level and open its own capture ----------
def process_cap(*, src: int | str, win_name: str, **kwargs):
    import cv2
    import mediapipe as mp
    mp_hands = mediapipe.solutions.hands # type: ignore
    hands = mp_hands.Hands()
    mpDraw = mediapipe.solutions.drawing_utils # type: ignore

    cap = cv2.VideoCapture(src)

    def distance_adjusted_grip_dist(f1, f2) -> float:
        return (( (f1.x - f2.x)**2 + (f1.y - f2.y)**2 ) ** 0.5) / ((abs(f1.z) + abs(f2.z))**0.6)

    def fingers_are_closed(f1, f2, random_ass_constant: float = 0.3) -> bool:
        return distance_adjusted_grip_dist(f1, f2) < random_ass_constant

    while True:
        ok, img = cap.read()
        if not ok:
            break
        img = cv2.flip(img, 1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                lms = list(handLms.landmark)
                thumb, index = lms[4], lms[8]
                if fingers_are_closed(thumb, index):
                    print(f"{win_name}: CLOSED")
                h, w, _ = img.shape
                for i in (4, 8):
                    cx, cy = int(lms[i].x * w), int(lms[i].y * h)
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                mpDraw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

        cv2.imshow(win_name, img)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    cap.release()
    cv2.destroyWindow(win_name)



def test_multi_proc(**kwargs):
    print(kwargs)

def test_main():
    sources: list[tuple[str | int, str]] = []

    if USE_WEBCAM:
        for cam_id in get_caps_ids():
            sources.append((cam_id, f"Webcam {cam_id}"))   # cam_id is an int

    if USE_DROID:
        sources.append((droid_src(), "DroidCam"))

    print("SOURCES:", sources)
    ps = [get_handler(process_cap, ctx=None, src=src, win_name=name) for src, name in sources]

    for p in ps: p.start()
    for p in ps: p.join()


def read_features(q_i: Queue, q_o: mp.Queue):
    while True:
        try:
            _ = q_i.get_nowait()
            return
        except _queue.Empty:
            pass
        try:
            print(q_o.get(timeout=1))
        except _queue.Empty:
            pass


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


# ---------- launcher ----------
def test_process_keypoints():
    sources: list[tuple[mp.Queue, mp.Queue, str | int, bool]] = []

    ctx = mp.get_context("spawn")  # single context everywhere
    if USE_WEBCAM:
        for cam_id in get_caps_ids():
            sources.append((ctx.Queue(), ctx.Queue(), cam_id, True))   # cam_id is an int

    if USE_DROID:
        sources.append((ctx.Queue(), ctx.Queue(), droid_src(), True))

    print("SOURCES:", sources)
    ps = [get_handler(process_keypoints, ctx=ctx, q_i=qi, q_o=qo, src=src, debug=name) for qi, qo, src, name in sources]
    q_ts = [Queue() for _ in sources]
    ts = [threading.Thread(target=read_features, args=(q, q_o)) for q, (_, q_o, _, _) in zip(q_ts, sources)]
    for t in ts: t.start()
    for p in ps: p.start()
    time.sleep(10)
    print("PUTTING THREAD")
    for q_t in q_ts: q_t.put_nowait(None)
    for t in ts: t.join()
    for qi, qo, src, debug in sources:
        print("PUTTING", src, flush=True)
        qi.put_nowait(None)
        qi.close(); qi.join_thread()
    for p in ps: p.join()

if __name__ == "__main__":
    # prefer spawn for OpenCV/Mediapipe stability
    mp.set_start_method("spawn", force=True)
    # test_main()
    # test_process_keypoints()
    cap_ids = get_caps_ids()
    droid = [droid_src()]
    droid = []
    hm = HandManager(cap_ids, droid, True)
    hm.start()
    time.sleep(10)
    hm.stop()
