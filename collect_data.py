import csv
import time
import os
import sys
from sensor import SensorConnection
from utils import extract_features   # we'll build this next

# ── Config ────────────────────────────────────────────────────────────────────
LETTERS         = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
DYNAMIC_LETTERS     = {"J", "Z"}
SAMPLES_PER_LETTER  = 150
EXTRA_LETTERS   = {"A", "E", "S", "M", "N", "D", "G"}  # high-confusion groups
EXTRA_SAMPLES   = 50     # additional samples for those letters
OUTPUT_FILE     = "data/static/gesture_data.csv"
COUNTDOWN_SECS  = 3      # time given to form gesture before recording starts
# ─────────────────────────────────────────────────────────────────────────────

def countdown(label, seconds=COUNTDOWN_SECS):
    print(f"\n>>> Get ready for '{label}'...")
    for i in range(seconds, 0, -1):
        print(f"    {i}...", end="\r")
        time.sleep(1)
    print("  GO!          ")

def collect_letter(sensor, label, n_samples):
    """Collect n_samples frames for the given letter label."""
    countdown(label)
    samples  = []
    attempts = 0
    max_attempts = n_samples * 20  # increased from 5 to 20

    print("  Hold your hand steady over the sensor...")

    while len(samples) < n_samples and attempts < max_attempts:
        hand = sensor.get_hand()
        if hand is not None:
            try:
                features = extract_features(hand)
                samples.append([label] + features.tolist())
                if len(samples) % 30 == 0:
                    print("  Collected " + str(len(samples)) + "/" + str(n_samples))
            except Exception as e:
                print("  Frame error: " + str(e))
        else:
            time.sleep(0.05)  # wait a bit if no hand data yet
        attempts += 1
        time.sleep(1 / 60)

    if len(samples) < n_samples:
        print("  Warning: only got " + str(len(samples)) + "/" + str(n_samples) + " samples for '" + label + "'")

    return samples

def save_samples(samples, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    mode = "a" if os.path.exists(filepath) else "w"
    with open(filepath, mode, newline="") as f:
        writer = csv.writer(f)
        writer.writerows(samples)

def main():
    sensor = SensorConnection()

    # Allow resuming — skip letters already collected
    collected = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE) as f:
            reader = csv.reader(f)
            collected = {row[0] for row in reader if row}
        print(f"Resuming — already collected: {sorted(collected)}")

    for letter in LETTERS:
        if letter in DYNAMIC_LETTERS:
            print(f"Skipping '{letter}' (dynamic - will collect separately)")
            continue
        if letter in collected:
            print(f"Skipping '{letter}' (already done)")
            continue

        n = SAMPLES_PER_LETTER + (EXTRA_SAMPLES if letter in EXTRA_LETTERS else 0)

        input(f"\nPress Enter when ready for '{letter}' ({n} samples)...")
        samples = collect_letter(sensor, letter, n)
        save_samples(samples, OUTPUT_FILE)
        print(f"  Saved {len(samples)} samples for '{letter}'.")

    print(f"\nDone! Dataset saved to '{OUTPUT_FILE}'.")

if __name__ == "__main__":
    main()