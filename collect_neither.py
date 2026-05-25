import numpy as np
import os
import time
from sensor import SensorConnection
from utils import extract_features

OUTPUT_DIR  = "data/dynamic/Neither"
SEQUENCES   = 400
WINDOW_SIZE = 30

def collect_sequence(sensor):
    frames = []
    while len(frames) < WINDOW_SIZE:
        hand = sensor.get_hand()
        if hand:
            try:
                frames.append(extract_features(hand))
            except Exception:
                pass
        time.sleep(1 / 115)
    return np.array(frames)

def main():
    sensor = SensorConnection()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    existing = len(os.listdir(OUTPUT_DIR))
    needed   = SEQUENCES - existing

    if needed <= 0:
        print(f"Already have {existing} Neither sequences.")
        return

    print(f"Collecting {needed} Neither sequences.")
    print("Hold any static letter pose during each recording.\n")
    print("Vary the letters — sign A, B, C, D etc. across sequences.\n")

    for i in range(needed):
        input(f"  [{i+1}/{needed}] Press Enter, hold a static letter pose...")
        time.sleep(0.2)
        seq = collect_sequence(sensor)
        filepath = os.path.join(OUTPUT_DIR, f"Neither_{existing+i:04d}.npy")
        np.save(filepath, seq)
        print(f"  Saved ({seq.shape})")

    print("\nDone collecting Neither sequences.")

if __name__ == "__main__":
    main()