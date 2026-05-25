"""
inference.py
------------
Real-time inference engine for ASL fingerspelling recognition.

Combines two models:
  1. Random Forest  - classifies the 24 static letters (A-Z, excluding J/Z)
  2. LSTM           - detects the 2 dynamic letters (J and Z)

The engine runs in a dedicated background thread, polling the Leap Motion
sensor at ~115fps, maintaining a sliding window for the LSTM, and placing
confirmed predictions into a thread-safe queue consumed by the GUI.

Public API
----------
    engine = InferenceEngine()
    engine.start()
    engine.stop()
    result = engine.get_prediction()
    engine.set_confidence_threshold(0.80)
"""

import time
import threading
import queue
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import joblib
from tensorflow import keras

from sensor import SensorConnection
from utils  import extract_features, extract_sequence_features, is_valid_frame


# ── Config ────────────────────────────────────────────────────────────────────

MODEL_DIR   = "models"

RF_MODEL_PATH    = f"{MODEL_DIR}/rf_model.pkl"
LE_PATH          = f"{MODEL_DIR}/label_encoder.pkl"
LSTM_MODEL_PATH  = f"{MODEL_DIR}/lstm_jz.keras"
LSTM_SCALER_PATH = f"{MODEL_DIR}/lstm_scaler.pkl"

WINDOW_SIZE          = 30
RF_CONFIDENCE        = 0.85
LSTM_CONFIDENCE      = 0.98
DEBOUNCE_FRAMES      = 3
LOCKOUT_FRAMES       = 150
GUI_QUEUE_MAXSIZE    = 5
POLL_SLEEP_SECS      = 1 / 115


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class PredictionResult:
    """A single confirmed gesture prediction."""
    letter      : str
    confidence  : float
    is_dynamic  : bool
    timestamp   : float = field(default_factory=time.time)

    def __str__(self):
        tag = " [dynamic]" if self.is_dynamic else ""
        return f"{self.letter}  ({self.confidence:.0%}){tag}"


# ── Inference engine ──────────────────────────────────────────────────────────

class InferenceEngine:
    """
    Runs the Leap Motion polling loop and classification pipeline
    in a background daemon thread.
    """

    def __init__(self):
        self._lock       = threading.RLock()
        self._thread     = None
        self._stop_event = threading.Event()

        self._queue = queue.Queue(maxsize=GUI_QUEUE_MAXSIZE)

        self._sensor      = None
        self._rf_model    = None
        self._le          = None
        self._lstm_model  = None
        self._lstm_scaler = None

        self._window          = deque(maxlen=WINDOW_SIZE)
        self._debounce_buffer = []
        self._last_emitted    = None
        self._lockout_counter = 0

        self._rf_threshold   = RF_CONFIDENCE
        self._lstm_threshold = LSTM_CONFIDENCE

        self._status = "stopped"

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self):
        """Load models and start the background inference thread."""
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._status = "loading"

        self._load_models()
        self._sensor = SensorConnection()

        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="InferenceThread",
        )
        self._thread.start()
        self._status = "running"
        print("Inference engine started.")

    def stop(self):
        """Signal the inference thread to stop."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        self._status = "stopped"
        print("Inference engine stopped.")

    def get_prediction(self):
        """Non-blocking retrieval of the most recent prediction."""
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def set_confidence_threshold(self, threshold: float):
        """Adjust the RF confidence gate at runtime."""
        with self._lock:
            self._rf_threshold = float(np.clip(threshold, 0.0, 1.0))

    def get_status(self) -> str:
        return self._status

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_models(self):
        print("Loading models...")

        try:
            self._rf_model = joblib.load(RF_MODEL_PATH)
            self._le       = joblib.load(LE_PATH)
            print(f"  Random Forest loaded  ({len(self._le.classes_)} classes)")
        except FileNotFoundError as e:
            raise RuntimeError(
                f"Static model not found: {e}\n"
                "Run train_model.py first."
            )

        try:
            self._lstm_model  = keras.models.load_model(LSTM_MODEL_PATH)
            self._lstm_scaler = joblib.load(LSTM_SCALER_PATH)
            print("  LSTM loaded")
        except FileNotFoundError as e:
            raise RuntimeError(
                f"Dynamic model not found: {e}\n"
                "Run train_dynamic.py first."
            )

        print("All models loaded.\n")

    # ── Main loop ─────────────────────────────────────────────────────────────

    def _run_loop(self):
        """Core polling loop - runs on the background thread."""
        while not self._stop_event.is_set():
            try:
                hand = self._sensor.get_hand()

                if hand is None:
                    self._status = "no_hand"
                    self._window.clear()
                    self._lockout_counter = 0
                    time.sleep(POLL_SLEEP_SECS)
                    continue

                if not is_valid_frame(hand):
                    time.sleep(POLL_SLEEP_SECS)
                    continue

                self._status = "running"

                # Extract features
                features = extract_features(hand)
                self._window.append(features)

                # Classify
                prediction, confidence, is_dynamic = self._classify(features)

                # Debounce and emit
                if prediction is not None:
                    self._debounce(prediction, confidence, is_dynamic)

            except Exception as e:
                self._status = "error"
                print(f"[InferenceEngine] Frame error: {e}")

            time.sleep(POLL_SLEEP_SECS)

    # ── Classification ────────────────────────────────────────────────────────

    def _classify(self, features: np.ndarray):
        """
        Run both classifiers and return the best prediction.

        Priority order:
          1. Check lockout first
          2. LSTM (dynamic) if window full and not in lockout
          3. Random Forest (static) fallback
        """
        with self._lock:
            rf_threshold   = self._rf_threshold
            lstm_threshold = self._lstm_threshold

        # Step 1: Decrement lockout counter if active
        if self._lockout_counter > 0:
            self._lockout_counter -= 1

        # Step 2: LSTM
        if len(self._window) == WINDOW_SIZE:
            letter, conf = self._classify_dynamic()
            if letter is not None and conf >= lstm_threshold:
                self._lockout_counter = LOCKOUT_FRAMES
                return letter, conf, True

        # Step 3: Static Random Forest only
        letter, conf = self._classify_static(features)
        if letter is not None and conf >= rf_threshold:
            return letter, conf, False

        return None, 0.0, False

    def _classify_static(self, features: np.ndarray):
        """Run the Random Forest on a single feature vector."""
        vec   = features.reshape(1, -1)
        proba = self._rf_model.predict_proba(vec)[0]
        idx   = int(np.argmax(proba))
        conf  = float(proba[idx])
        letter = self._le.inverse_transform([idx])[0]
        return letter, conf

    def _classify_dynamic(self):
        """Run the LSTM on the current sliding window."""
        window_array = np.array(list(self._window), dtype=np.float32)
        augmented    = extract_sequence_features(window_array)

        T, F = augmented.shape
        scaled = self._lstm_scaler.transform(
        augmented.reshape(-1, F)
        ).reshape(1, T, F).astype(np.float32)

        proba  = self._lstm_model.predict(scaled, verbose=0)[0]
        idx    = int(np.argmax(proba))
        conf   = float(proba[idx])

        # Classes: 0=J, 1=Z, 2=Neither
        if idx == 2:
            return None, conf

        # Extra guard — only emit if Neither probability is low
        neither_prob = float(proba[2])
        if neither_prob > 0.08:
            return None, conf

        letter = ["J", "Z"][idx]
        return letter, conf

    # ── Debouncing ────────────────────────────────────────────────────────────

    def _debounce(self, letter: str, confidence: float, is_dynamic: bool):
        """
        Only emit a prediction after the same letter appears
        DEBOUNCE_FRAMES times consecutively.
        Dynamic letters bypass debounce - lockout handles deduplication.
        """
        if is_dynamic:
            self._emit(PredictionResult(
                letter=letter, confidence=confidence, is_dynamic=True
            ))
            return

        self._debounce_buffer.append(letter)
        if len(self._debounce_buffer) > DEBOUNCE_FRAMES:
            self._debounce_buffer.pop(0)

        if (len(self._debounce_buffer) == DEBOUNCE_FRAMES and
                len(set(self._debounce_buffer)) == 1):
            if letter != self._last_emitted:
                self._last_emitted = letter
                self._emit(PredictionResult(
                    letter=letter, confidence=confidence, is_dynamic=False
                ))
        elif len(set(self._debounce_buffer)) > 1:
            self._last_emitted = None

    def _emit(self, result: PredictionResult):
        """Place a prediction on the GUI queue."""
        try:
            self._queue.put_nowait(result)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(result)


# ── Standalone test (no GUI) ──────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running inference engine in standalone mode.")
    print("Sign ASL letters in front of the Leap Motion Controller.")
    print("Press Ctrl+C to stop.\n")

    engine = InferenceEngine()
    engine.start()

    try:
        while True:
            result = engine.get_prediction()
            if result:
                print(f"  -> {result}")
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        engine.stop()