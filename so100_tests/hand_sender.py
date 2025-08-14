import cv2
import json
import time
import zmq
import argparse

try:
    import mediapipe as mp
except ImportError as e:
    raise SystemExit("MediaPipe not installed. Run: pip install mediapipe")

# ---- CLI ----
parser = argparse.ArgumentParser()
parser.add_argument("--bind", default="tcp://*:5555", help="ZMQ bind address for PUB socket")
parser.add_argument("--topic", default="hands", help="ZMQ topic string")
parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
parser.add_argument("--fps", type=float, default=60.0, help="Max publish rate (Hz)")
args = parser.parse_args()

# ---- ZMQ Publisher ----
ctx = zmq.Context.instance()
sock = ctx.socket(zmq.PUB)
sock.setsockopt(zmq.SNDHWM, 10)  # small HWM to keep latency low
sock.bind(args.bind)
print(f"[PUB] Bound at {args.bind} on topic '{args.topic}'")

# ---- MediaPipe Hands ----
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(args.camera)
if not cap.isOpened():
    raise SystemExit(f"Could not open camera index {args.camera}")

publish_period = 1.0 / max(args.fps, 1.0)
last_pub = 0.0

with mp_hands.Hands(
    model_complexity=0,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
) as hands:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[PUB] Frame grab failed; exiting.")
            break

        h, w = frame.shape[:2]
        # Convert BGR->RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = hands.process(rgb)

        # Draw landmarks for visual feedback
        if res.multi_hand_landmarks:
            for hand_landmarks in res.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                )

        # Publish at the requested rate
        now = time.time()
        if now - last_pub >= publish_period:
            last_pub = now
            payload = {
                "topic": args.topic,
                "t_send": now,
                "width": int(w),
                "height": int(h),
                "landmarks": []
            }
            if res.multi_hand_landmarks:
                # Use the first detected hand
                hand = res.multi_hand_landmarks[0]
                for idx, lm in enumerate(hand.landmark):
                    payload["landmarks"].append({
                        "id": int(idx),
                        "x": float(lm.x),
                        "y": float(lm.y),
                        "z": float(lm.z),
                    })
            # Topic framing: send as multipart [topic, json]
            sock.send_multipart([
                args.topic.encode("utf-8"),
                json.dumps(payload).encode("utf-8")
            ])

        # Show window and allow exit
        cv2.imshow("Hand Sender (MediaPipe)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()