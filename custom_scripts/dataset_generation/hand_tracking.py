import cv2
import mediapipe as mp
import time
import math
import numpy as np
import multiprocessing as mp_proc

from handle_processes import get_handler

USE_DROID = True
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
    mp_hands = mp.solutions.hands # type: ignore
    hands = mp_hands.Hands()
    mpDraw = mp.solutions.drawing_utils # type: ignore

    cap = cv2.VideoCapture(src)

    def distance_adjusted_grip_dist(f1, f2) -> float:
        return (( (f1.x - f2.x)**2 + (f1.y - f2.y)**2 ) ** 0.5) / ((abs(f1.z) + abs(f2.z))**0.6)

    def fingers_are_closed(f1, f2, thr: float = 0.3) -> bool:
        return distance_adjusted_grip_dist(f1, f2) < thr

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

# ---------- launcher ----------
def test_main():
    sources: list[tuple[str | int, str]] = []

    if USE_WEBCAM:
        for cam_id in get_caps_ids():
            sources.append((cam_id, f"Webcam {cam_id}"))   # cam_id is an int

    if USE_DROID:
        sources.append((droid_src(), "DroidCam"))

    print("SOURCES:", sources)
    ps = [get_handler(process_cap, src=src, win_name=name) for src, name in sources]

    for p in ps: p.start()
    for p in ps: p.join()

if __name__ == "__main__":
    # prefer spawn for OpenCV/Mediapipe stability
    try:
        mp_proc.set_start_method("spawn")
    except RuntimeError:
        pass
    test_main()
