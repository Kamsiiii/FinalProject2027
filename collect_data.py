import csv
import time
import os
import sys
from sensor import SensorConnection
from utils import extract_features   # we'll build this next

# ── Config ────────────────────────────────────────────────────────────────────
LETTERS         = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
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
    samples = []
    attempts = 0
    max_attempts = n_samples * 5  # allow some missed frames

    while len(samples) < n_samples and attempts < max_attempts:
        hand = sensor.get_hand()
        if hand:
            try:
                features = extract_features(hand)
                samples.append([label] + features.tolist())
            except Exception as e:
                pass  # skip malformed frames silently
        attempts += 1
        time.sleep(1 / 60)  # ~60 fps polling rate

    if len(samples) < n_samples:
        print(f"  Warning: only got {len(samples)}/{n_samples} samples for '{label}'")

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