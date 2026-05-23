import numpy as np
import os
import time
from sensor import SensorConnection
from utils import extract_features

DYNAMIC_LETTERS  = ["J", "Z"]
SEQUENCES        = 200      # sequences per letter
WINDOW_SIZE      = 30       # frames per sequence (~260ms at 115fps)
OUTPUT_DIR       = "data/dynamic"

def collect_sequence(sensor, label):
    """Record one sequence of WINDOW_SIZE frames."""
    frames = []
    while len(frames) < WINDOW_SIZE:
        hand = sensor.get_hand()
        if hand:
            try:
                frames.append(extract_features(hand))
            except Exception:
                pass
        time.sleep(1 / 115)
    return np.array(frames)   # shape: (30, 71)

def main():
    sensor = SensorConnection()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for letter in DYNAMIC_LETTERS:
        letter_dir = os.path.join(OUTPUT_DIR, letter)
        os.makedirs(letter_dir, exist_ok=True)

        existing = len(os.listdir(letter_dir))
        needed   = SEQUENCES - existing
        if needed <= 0:
            print(f"'{letter}' already has {existing} sequences, skipping.")
            continue

        print(f"\nCollecting {needed} sequences for '{letter}'")
        print("Sign the letter naturally when prompted.\n")

        for i in range(needed):
            input(f"  [{i+1}/{needed}] Press Enter, then sign '{letter}'...")
            time.sleep(0.2)   # brief pause before capture starts
            seq = collect_sequence(sensor, letter)
            filepath = os.path.join(letter_dir, f"{letter}_{existing + i:04d}.npy")
            np.save(filepath, seq)
            print(f"  Saved sequence ({seq.shape})")

        print(f"Done with '{letter}'.")

    print("\nAll dynamic sequences collected.")

if __name__ == "__main__":
    main()