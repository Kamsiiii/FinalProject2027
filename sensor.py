# sensor.py — connects directly to Leap Motion v4 WebSocket service
import json
import time
import threading
import websocket

WS_URL = "ws://localhost:6437/v6.json"

class SensorConnection:

    def __init__(self):
        self._latest    = None
        self._lock      = threading.Lock()
        self._ws_open   = False
        self._connect()

    def _connect(self):
        def on_message(ws, message):
            try:
                data = json.loads(message)
                # v4 sends different event types — we only want tracking frames
                if "hands" in data and len(data["hands"]) > 0:
                    with self._lock:
                        self._latest = self._parse_frame(data)
            except Exception as e:
                pass

        def on_error(ws, error):
            print("WebSocket error: " + str(error))

        def on_close(ws, *args):
            self._ws_open = False
            print("WebSocket closed.")

        def on_open(ws):
            self._ws_open = True
            print("Connected to Leap Motion v4 service.")
            # Enable tracking data
            ws.send(json.dumps({"focused": True}))
            ws.send(json.dumps({"optimizeHMD": False}))

        ws = websocket.WebSocketApp(
            WS_URL,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open,
        )
        t = threading.Thread(target=ws.run_forever, daemon=True)
        t.start()

        # Wait for connection
        timeout = time.time() + 10
        while not self._ws_open:
            if time.time() > timeout:
                raise RuntimeError(
                    "Could not connect to Leap Motion v4 service.\n"
                    "Make sure the Leap Motion Control Panel is running."
                )
            time.sleep(0.1)

        print("Sensor ready. Show your hand to the Leap Motion.")

    def _parse_frame(self, data):
        """Convert v4 WebSocket frame into the dict format our code expects."""
        hand = data["hands"][0]
        pointables = [p for p in data.get("pointables", [])
                      if p["handId"] == hand["id"]]

        # Sort fingers by type (0=thumb to 4=pinky)
        pointables.sort(key=lambda p: p["type"])

        # Pad to 5 fingers if any are missing
        while len(pointables) < 5:
            pointables.append(self._empty_finger())

        fingers = []
        for p in pointables[:5]:
            bones = []
            for b in ["metacarpal", "proximal", "intermediate", "distal"]:
                bone_dir = p.get(b + "Direction", [0, 0, 1])
                bones.append({"direction": bone_dir})
            fingers.append({
                "is_extended"  : p.get("extended", False),
                "tip_position" : p.get("tipPosition", [0, 0, 0]),
                "bones"        : bones,
            })

        return {
            "direction"    : hand.get("direction", [0, 0, 1]),
            "palm_normal"  : hand.get("palmNormal", [0, -1, 0]),
            "palm_position": hand.get("palmPosition", [0, 0, 0]),
            "fingers"      : fingers,
        }

    def _empty_finger(self):
        return {
            "is_extended"  : False,
            "tip_position" : [0, 0, 0],
            "bones"        : [{"direction": [0, 0, 1]}] * 4,
        }

    def get_hand(self):
        with self._lock:
            return self._latest

    def is_connected(self):
        return self._ws_open