import json
import time
import queue
import threading
import argparse
from dataclasses import dataclass

import numpy as np
import zmq

# ---------- Optional: Try to import Genesis, else fall back to a tiny mock ----------
USING_MOCK = False
try:
    # Replace this import with the correct one for your Genesis installation
    # For example, if the package exposes a 'genesis' namespace:
    import genesis as gs  # noqa: F401
except Exception as e:
    USING_MOCK = True
    print("[WARN] Genesis import failed; using MockGenesis. Reason:", e)

# ----- Simple mock to visualize the index fingertip as a moving point -----
class MockGenesis:
    def __init__(self):
        self._pos = np.zeros(3, dtype=float)
        print("[MockGenesis] Initialized. (No real physics, just printing positions)")

    def set_index_tip_position(self, pos3):
        self._pos[:] = pos3

    def step(self):
        # In a real sim you'd advance physics and render; we just print occasionally
        pass

    def render_text(self, text):
        print(text)

# ----- Replace these with your real Genesis scene wiring -----
class GenesisAdapter:
    def __init__(self):
        if USING_MOCK:
            self.engine = MockGenesis()
        else:
            # TODO: initialize the actual Genesis engine, scene, and objects
            # e.g., gs.Engine(), load assets, create actors, etc.
            self.engine = self._init_real_genesis()
        self._last_print = 0.0

    def _init_real_genesis(self):
        # Pseudocode placeholders; change to match the real Genesis API
        # engine = gs.Engine(headless=False)
        # scene = engine.create_scene()
        # self.index_marker = scene.add_sphere(radius=0.02, color=(0.9, 0.2, 0.2))
        # return engine
        return MockGenesis()  # remove when you wire the real engine

    def set_index_tip_position(self, pos3):
        # If you created an object (e.g., a small sphere) to represent the fingertip,
        # set its transform here. Replace with actual Genesis calls.
        self.engine.set_index_tip_position(pos3)

    def step(self):
        self.engine.step()

    def maybe_debug_print(self, pos3):
        now = time.time()
        if now - self._last_print > 0.25:
            self._last_print = now
            self.engine.render_text(f"IndexTip @ {pos3}")

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

# ---------- Mapping from MediaPipe to Genesis space ----------
# This maps normalized image coords to a small 3D box centered at the origin.
# Tweak SCALE and OFFSET to fit your robot/scene.
SCALE = np.array([0.4, 0.3, 0.4], dtype=float)  # meters
OFFSET = np.array([0.0, 0.0, 0.6], dtype=float) # meters (e.g., 60 cm in front)

# MediaPipe uses image coords: origin top-left, y down. We flip Y to conventional up.
# Z is relative depth (negative is closer); we negate it so closer -> larger +Z.
def image_to_world(x, y, z):
    xw = (x - 0.5) * SCALE[0] + OFFSET[0]
    yw = (0.5 - y) * SCALE[1] + OFFSET[1]
    zw = (-z) * SCALE[2] + OFFSET[2]
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
                    sim.maybe_debug_print(pos)

            sim.step()
    except KeyboardInterrupt:
        print("[Receiver] Interrupted. Exiting.")
    finally:
        recv.stop()