# sensor_listener.py  —  Python 2.7, runs in LMC-p27 conda env
import sys
import os
import json
import time
import threading
import SimpleWebSocketServer  # install: pip install SimpleWebSocketServer

sys.path.insert(0, os.path.abspath("../lib"))
import Leap

# WebSocket server that broadcasts hand data to Python 3 clients
from SimpleWebSocketServer import SimpleWebSocketServer, WebSocket

clients = []

class LeapHandler(WebSocket):
    def handleConnected(self):
        clients.append(self)
        print("Client connected:", self.address)

    def handleClose(self):
        clients.remove(self)
        print("Client disconnected:", self.address)

    def handleMessage(self):
        pass

class LeapListener(Leap.Listener):
    def on_frame(self, controller):
        frame = controller.frame()
        if not frame.hands.is_empty:
            hand = frame.hands[0]
            data = _extract(hand, frame)
            msg  = json.dumps(data)
            for client in clients:
                try:
                    client.sendMessage(unicode(msg))
                except Exception:
                    pass

def _extract(hand, frame):
    """Pull raw bone/finger data out of the Leap hand object."""
    fingers = []
    for finger in hand.fingers:
        bones = []
        for b in range(4):   # 0=metacarpal, 1=proximal, 2=intermediate, 3=distal
            bone = finger.bone(b)
            bones.append({
                "direction": [bone.direction.x,
                               bone.direction.y,
                               bone.direction.z],
            })
        fingers.append({
            "is_extended": finger.is_extended,
            "tip_position": [finger.tip_position.x,
                              finger.tip_position.y,
                              finger.tip_position.z],
            "bones": bones,
        })

    return {
        "timestamp"    : frame.timestamp,
        "direction"    : [hand.direction.x, hand.direction.y, hand.direction.z],
        "palm_normal"  : [hand.palm_normal.x, hand.palm_normal.y, hand.palm_normal.z],
        "palm_position": [hand.palm_position.x, hand.palm_position.y, hand.palm_position.z],
        "fingers"      : fingers,
    }

if __name__ == "__main__":
    listener   = LeapListener()
    controller = Leap.Controller()
    controller.add_listener(listener)

    server = SimpleWebSocketServer("localhost", 6437, LeapHandler)
    print("WebSocket server running on ws://localhost:6437")
    print("Waiting for Leap Motion frames...")
    server.serveforever()