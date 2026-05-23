"""
inference.py
────────────
Real-time inference engine for ASL fingerspelling recognition.

Combines two models:
  1. Random Forest  – classifies the 24 static letters (A–Z, excluding J/Z)
  2. LSTM           – detects the 2 dynamic letters (J and Z)

The engine runs in a dedicated background thread, polling the Leap Motion
sensor at ~115fps, maintaining a sliding window for the LSTM, and placing
confirmed predictions into a thread-safe queue consumed by the GUI.

Public API
----------
    engine = InferenceEngine()
    engine.start()                    # begin inference loop in background thread
    engine.stop()                     # gracefully shut down

    # From the GUI thread:
    result = engine.get_prediction()  # returns PredictionResult | None
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

WINDOW_SIZE          = 30     # frames in the LSTM sliding window
RF_CONFIDENCE        = 0.85   # default minimum confidence for static letters
LSTM_CONFIDENCE      = 0.80   # minimum confidence for J / Z detection
DEBOUNCE_FRAMES      = 3      # same prediction must appear N times before emit
LOCKOUT_FRAMES       = 45     # frames to suppress dynamic classifier after J/Z hit
GUI_QUEUE_MAXSIZE    = 5      # max buffered predictions (oldest dropped if full)
POLL_SLEEP_SECS      = 1 / 115  # ~115fps sensor polling


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class PredictionResult:
    """A single confirmed gesture prediction."""
    letter      : str               # predicted letter, e.g. "A"
    confidence  : float             # posterior probability 0–1
    is_dynamic  : bool              # True if predicted by LSTM (J or Z)
    timestamp   : float = field(default_factory=time.time)

    def __str__(self):
        tag = " [dynamic]" if self.is_dynamic else ""
        return f"{self.letter}  ({self.confidence:.0%}){tag}"


# ── Inference engine ──────────────────────────────────────────────────────────

class InferenceEngine:
    """
    Runs the Leap Motion polling loop and classification pipeline in a
    background daemon thread.

    Thread safety
    -------------
    All public methods are safe to call from any thread.
    Internal state is protected by a reentrant lock (_lock).
    Predictions are delivered via a thread-safe Queue.
    """

    def __init__(self):
        self._lock   = threading.RLock()
        self._thread : Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Prediction queue consumed by the GUI
        self._queue : queue.Queue = queue.Queue(maxsize=GUI_QUEUE_MAXSIZE)

        # Sensor
        self._sensor : Optional[SensorConnection] = None

        # Models (loaded on start to keep __init__ fast)
        self._rf_model  = None
        self._le        = None
        self._lstm_model = None
        self._lstm_scaler = None

        # Sliding window: deque of raw feature vectors (np.ndarray shape 71)
        self._window : deque = deque(maxlen=WINDOW_SIZE)

        # Debounce state
        self._debounce_buffer : list = []   # recent raw predictions
        self._last_emitted    : Optional[str] = None

        # Dynamic classifier lockout
        self._lockout_counter : int = 0

        # Runtime settings (adjustable while running)
        self._rf_threshold   : float = RF_CONFIDENCE
        self._lstm_threshold : float = LSTM_CONFIDENCE

        # Status
        self._status : str = "stopped"   # "stopped" | "running" | "no_hand" | "error"

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self):
        """Load models and start the background inference thread."""
        if self._thread and self._thread.is_alive():
            return  # already running

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
        """Signal the inference thread to stop and wait for it to finish."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        self._status = "stopped"
        print("Inference engine stopped.")

    def get_prediction(self) -> Optional[PredictionResult]:
        """
        Non-blocking retrieval of the most recent prediction.
        Returns None if the queue is empty.
        """
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def set_confidence_threshold(self, threshold: float):
        """Adjust the RF confidence gate at runtime (called from GUI slider)."""
        with self._lock:
            self._rf_threshold = float(np.clip(threshold, 0.0, 1.0))

    def get_status(self) -> str:
        """Returns current engine status string."""
        return self._status

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_models(self):
        """Load all serialised models from disk."""
        print("Loading models...")

        try:
            self._rf_model    = joblib.load(RF_MODEL_PATH)
            self._le          = joblib.load(LE_PATH)
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
        """
        Core polling loop — runs on the background thread.

        Each iteration:
          1. Poll Leap Motion for a frame
          2. Validate and extract features
          3. Update sliding window
          4. Try dynamic classification (if window full and not in lockout)
          5. Fall through to static classification
          6. Debounce and emit confirmed predictions
        """
        while not self._stop_event.is_set():
            try:
                hand = self._sensor.get_hand()

                if hand is None or not is_valid_frame(hand):
                    self._status = "no_hand"
                    time.sleep(POLL_SLEEP_SECS)
                    continue

                self._status = "running"

                # ── Feature extraction ────────────────────────────────────
                features = extract_features(hand)   # shape (71,)
                self._window.append(features)

                # ── Dynamic classification (J / Z) ────────────────────────
                prediction, confidence, is_dynamic = self._classify(features)

                # ── Debounce ──────────────────────────────────────────────
                if prediction is not None:
                    self._debounce(prediction, confidence, is_dynamic)

            except Exception as e:
                # Don't let a single bad frame crash the engine
                self._status = "error"
                print(f"[InferenceEngine] Frame error: {e}")

            time.sleep(POLL_SLEEP_SECS)

    # ── Classification ────────────────────────────────────────────────────────

    def _classify(self, features: np.ndarray):
        """
        Run both classifiers and return the best prediction.

        Priority order:
          1. LSTM (dynamic) — only when window is full and not in lockout
          2. Random Forest  (static) — fallback

        Returns
        -------
        letter     : str | None
        confidence : float
        is_dynamic : bool
        """
        with self._lock:
            rf_threshold   = self._rf_threshold
            lstm_threshold = self._lstm_threshold

        # ── Step 1: Try LSTM if window is full ────────────────────────────
        if len(self._window) == WINDOW_SIZE and self._lockout_counter == 0:
            letter, conf = self._classify_dynamic()
            if letter is not None and conf >= lstm_threshold:
                # Trigger lockout to prevent re-detecting the same gesture
                self._lockout_counter = LOCKOUT_FRAMES
                return letter, conf, True
        elif self._lockout_counter > 0:
            self._lockout_counter -= 1

        # ── Step 2: Static Random Forest ──────────────────────────────────
        letter, conf = self._classify_static(features)
        if letter is not None and conf >= rf_threshold:
            return letter, conf, False

        return None, 0.0, False

    def _classify_static(self, features: np.ndarray):
        """
        Run the Random Forest on a single feature vector.

        Returns
        -------
        letter     : str | None    (None if below threshold)
        confidence : float
        """
        vec   = features.reshape(1, -1)
        proba = self._rf_model.predict_proba(vec)[0]
        idx   = int(np.argmax(proba))
        conf  = float(proba[idx])
        letter = self._le.inverse_transform([idx])[0]
        return letter, conf

    def _classify_dynamic(self):
        """
        Run the LSTM on the current sliding window.

        Returns
        -------
        letter     : str | None    ("J", "Z", or None if "Neither" / low confidence)
        confidence : float
        """
        # Build augmented sequence from the window
        window_array = np.array(list(self._window), dtype=np.float32)   # (30, 71)
        augmented    = extract_sequence_features(window_array)           # (30, 213)

        # Scale using the fitted scaler
        T, F = augmented.shape
        scaled = self._lstm_scaler.transform(
            augmented.reshape(-1, F)
        ).reshape(1, T, F).astype(np.float32)

        # Predict
        proba    = self._lstm_model.predict(scaled, verbose=0)[0]  # (3,)
        idx      = int(np.argmax(proba))
        conf     = float(proba[idx])
        # Classes: 0=J, 1=Z, 2=Neither
        if idx == 2:
            return None, conf   # "Neither" — don't emit
        letter = ["J", "Z"][idx]
        return letter, conf

    # ── Debouncing ────────────────────────────────────────────────────────────

    def _debounce(self, letter: str, confidence: float, is_dynamic: bool):
        """
        Only emit a prediction after the same letter appears DEBOUNCE_FRAMES
        times consecutively, preventing single-frame noise from inserting
        incorrect letters into the word buffer.

        Dynamic letters (J, Z) bypass debounce — they are inherently brief
        and the LSTM's lockout mechanism prevents duplicates.
        """
        if is_dynamic:
            # Dynamic letters emitted immediately — lockout handles deduplication
            self._emit(PredictionResult(
                letter=letter, confidence=confidence, is_dynamic=True
            ))
            return

        self._debounce_buffer.append(letter)
        if len(self._debounce_buffer) > DEBOUNCE_FRAMES:
            self._debounce_buffer.pop(0)

        # Emit only when all DEBOUNCE_FRAMES slots agree on the same letter
        if (len(self._debounce_buffer) == DEBOUNCE_FRAMES and
                len(set(self._debounce_buffer)) == 1):

            # Don't re-emit the same letter repeatedly while the hand is held
            if letter != self._last_emitted:
                self._last_emitted = letter
                self._emit(PredictionResult(
                    letter=letter, confidence=confidence, is_dynamic=False
                ))
        elif len(set(self._debounce_buffer)) > 1:
            # Buffer disagrees — reset last emitted so a new stable prediction
            # will trigger an emit even if it's the same letter as before
            self._last_emitted = None

    def _emit(self, result: PredictionResult):
        """
        Place a prediction on the GUI queue.
        If the queue is full, drop the oldest item to make room.
        """
        try:
            self._queue.put_nowait(result)
        except queue.Full:
            try:
                self._queue.get_nowait()   # discard oldest
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
                print(f"  → {result}")
            time.sleep(0.05)   # poll GUI queue at ~20fps
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        engine.stop()