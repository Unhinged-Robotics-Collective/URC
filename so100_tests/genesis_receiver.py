import json
import time
import queue
import threading
import argparse
from dataclasses import dataclass

import numpy as np
import zmq

# ---------- Import Genesis ----------
import genesis as gs  # noqa: F401

class GenesisAdapter:
    def __init__(self):
        self.scene, self.sphere = self._init_real_genesis()
        self._last_print = 0.0

    def _init_real_genesis(self):
        gs.init(backend=gs.gpu)
        scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(0, -3.5, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=30,
                max_FPS=60,
            ),
            sim_options=gs.options.SimOptions(dt=0.01),
            show_viewer=True,
        )
        plane = scene.add_entity(gs.morphs.Plane())
        sphere = scene.add_entity(
            gs.morphs.Sphere(radius=0.02)
        )
        scene.build()
        return scene, sphere

    def set_index_tip_position(self, pos3):
        self.sphere.set_pos(pos3)

    def step(self):
        self.scene.step()

    def debug_print(self, pos3):
        now = time.time()
        if now - self._last_print > 0.5:
            self._last_print = now
            print(f"IndexTip @ {pos3}")

# ---------- ZMQ Listener Thread ----------
@dataclass
class HandPacket:
    t_send: float
    width: int
    height: int
    landmarks: list  # list of {id, x, y, z}

class LandmarkReceiver:
    def __init__(self, address: str, topic: str, hwm: int = 10):
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.SUB)
        self.sock.setsockopt(zmq.RCVHWM, hwm)
        self.sock.setsockopt_string(zmq.SUBSCRIBE, topic)
        self.sock.connect(address)
        self.queue = queue.Queue(maxsize=4)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        print(f"[SUB] Connected to {address} subscribing to topic '{topic}'")

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=1.0)

    def _run(self):
        poller = zmq.Poller()
        poller.register(self.sock, zmq.POLLIN)
        while not self._stop.is_set():
            events = dict(poller.poll(10))  # 10 ms
            if self.sock in events:
                try:
                    topic, payload = self.sock.recv_multipart(flags=zmq.NOBLOCK)
                    data = json.loads(payload.decode("utf-8"))
                    pkt = HandPacket(
                        t_send=float(data.get("t_send", 0.0)),
                        width=int(data.get("width", 0)),
                        height=int(data.get("height", 0)),
                        landmarks=data.get("landmarks", []),
                    )
                    # keep only the most recent
                    while not self.queue.empty():
                        try:
                            self.queue.get_nowait()
                        except queue.Empty:
                            break
                    self.queue.put_nowait(pkt)
                except Exception as e:
                    print("[SUB] Error parsing packet:", e)

# ---------- Mapping to Genesis space ----------
SCALE = np.array([1.0, 1.0, 10.0], dtype=float)  # 
SCALE = np.array([0, 0, 10.0], dtype=float)  # 
OFFSET = np.array([0.0, 0.0, 0.5], dtype=float) #

def image_to_world(x, y, z):
    xw = x * SCALE[0] + OFFSET[0]
    yw = y * SCALE[1] + OFFSET[1]
    zw = z * SCALE[2] + OFFSET[2]
    return np.array([xw, yw, zw], dtype=float)

# ---------- Main App ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--connect", default="tcp://localhost:5555", help="ZMQ connect address for SUB socket")
    parser.add_argument("--topic", default="hands", help="ZMQ topic to subscribe to")
    parser.add_argument("--hz", type=float, default=120.0, help="Simulation step rate (Hz)")
    args = parser.parse_args()

    recv = LandmarkReceiver(args.connect, args.topic)
    recv.start()

    sim = GenesisAdapter()

    step_period = 1.0 / max(args.hz, 1.0)
    last = time.time()

    try:
        while True:
            now = time.time()
            dt = now - last
            if dt < step_period:
                time.sleep(step_period - dt)
                continue
            last = time.time()

            # Get latest packet if available
            pkt = None
            try:
                pkt = recv.queue.get_nowait()
            except queue.Empty:
                pass

            if pkt and pkt.landmarks:
                # Index fingertip id in MediaPipe is 8
                tip = next((lm for lm in pkt.landmarks if lm.get("id") == 8), None)
                if tip is not None:
                    pos = image_to_world(tip["x"], tip["y"], tip["z"])
                    sim.set_index_tip_position(pos)
                    sim.debug_print(pos)

            sim.step()
    except KeyboardInterrupt:
        print("[Receiver] Interrupted. Exiting.")
    finally:
        recv.stop()