"""
train_dynamic.py
────────────────
Trains an LSTM sequence classifier for the two dynamic ASL letters
J and Z, which require temporal motion tracking rather than single-frame
pose classification.

Each training example is a sequence of 30 consecutive frames (~260ms at
115fps), represented as a (30, 213) array: the 71-d base feature vector
augmented with velocity and acceleration (see utils.extract_sequence_features).

The model outputs probabilities over 3 classes:
    0 → J
    1 → Z
    2 → Neither  (static / transitional hand pose)

Usage
-----
    python train_dynamic.py

Outputs
-------
    models/lstm_jz.keras            – Saved LSTM model
    models/lstm_scaler.pkl          – Fitted StandardScaler (per-feature z-score)
    evaluation/lstm_report.txt      – Classification report
    evaluation/lstm_confusion.png   – Confusion matrix
    evaluation/lstm_training_curve.png
"""

import os
import numpy as np
import joblib
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.preprocessing   import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics         import classification_report, confusion_matrix, ConfusionMatrixDisplay

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks


# ── Config ────────────────────────────────────────────────────────────────────

DYNAMIC_DIR     = "data/dynamic"       # contains subfolders J/ and Z/
STATIC_DIR      = "data/static"        # CSV used to sample 'Neither' frames
MODEL_DIR       = "models"
EVAL_DIR        = "evaluation"

WINDOW_SIZE     = 30                   # frames per sequence
AUGMENTED_DIMS  = 213                  # 71 base + 71 velocity + 71 acceleration

NEITHER_SEQUENCES = 400                # how many 'Neither' sequences to generate
                                       # (sampled from static letter data)

TEST_SIZE       = 0.20
RANDOM_SEED     = 42
BATCH_SIZE      = 32
MAX_EPOCHS      = 100
PATIENCE        = 15                   # early stopping patience

CLASS_NAMES     = ["J", "Z", "Neither"]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_dynamic_sequences(dynamic_dir: str):
    """
    Load all .npy sequence files for J and Z.

    Each file is expected to be shape (WINDOW_SIZE, 71) — the raw base
    feature vectors before velocity/acceleration augmentation.

    Returns
    -------
    sequences : list of np.ndarray, each shape (30, 71)
    labels    : list of str  ("J" or "Z")
    """
    sequences, labels = [], []

    for letter in ["J", "Z"]:
        letter_dir = os.path.join(dynamic_dir, letter)
        if not os.path.exists(letter_dir):
            raise FileNotFoundError(
                f"Dynamic data folder not found: '{letter_dir}'.\n"
                "Run collect_dynamic.py first."
            )
        files = sorted(f for f in os.listdir(letter_dir) if f.endswith(".npy"))
        if len(files) == 0:
            raise ValueError(f"No .npy files found in '{letter_dir}'.")

        for fname in files:
            seq = np.load(os.path.join(letter_dir, fname))
            if seq.shape != (WINDOW_SIZE, 71):
                print(f"  Skipping {fname}: unexpected shape {seq.shape}")
                continue
            sequences.append(seq)
            labels.append(letter)

        print(f"Loaded {len(files)} sequences for '{letter}'.")

    return sequences, labels


def generate_neither_sequences(static_csv: str, n_sequences: int):
    """
    Synthesise 'Neither' class sequences from the static letter training data.

    Each Neither sequence is built by sampling WINDOW_SIZE consecutive rows
    from a single letter's static frames, simulating a held static pose over
    time. A small amount of Gaussian noise is added to prevent the model
    from learning to trivially distinguish Neither from dynamic signs by
    the absence of variation.

    Returns
    -------
    sequences : list of np.ndarray, each shape (30, 71)
    """
    import csv
    from utils import BASE_FEATURES

    # Load all static frames grouped by letter
    frames_by_letter = {}
    with open(static_csv, newline="") as f:
        for row in csv.reader(f):
            if not row or len(row) - 1 != BASE_FEATURES:
                continue
            label = row[0].strip().upper()
            vec   = np.array([float(v) for v in row[1:]], dtype=np.float32)
            frames_by_letter.setdefault(label, []).append(vec)

    rng = np.random.default_rng(RANDOM_SEED)
    sequences = []
    letters   = sorted(frames_by_letter.keys())

    for _ in range(n_sequences):
        letter = rng.choice(letters)
        pool   = np.array(frames_by_letter[letter])

        # Pick WINDOW_SIZE frames (with replacement if pool is small)
        indices = rng.choice(len(pool), size=WINDOW_SIZE, replace=True)
        seq     = pool[indices]

        # Add small Gaussian noise to simulate natural micro-movement
        noise = rng.normal(0, 0.01, size=seq.shape).astype(np.float32)
        sequences.append(seq + noise)

    print(f"Generated {n_sequences} 'Neither' sequences from static data.")
    return sequences


def build_dataset(dynamic_dir: str, static_csv: str):
    """
    Combine J, Z, and Neither sequences into a single labelled dataset.

    Applies velocity/acceleration augmentation and stacks into arrays.

    Returns
    -------
    X : np.ndarray, shape (N, WINDOW_SIZE, AUGMENTED_DIMS)
    y : np.ndarray, shape (N,)  integer class indices 0/1/2
    le : LabelEncoder
    """
    from utils import extract_sequence_features

    sequences, labels = load_dynamic_sequences(dynamic_dir)

    # Neither class
    if not os.path.exists(static_csv):
        raise FileNotFoundError(
            f"Static CSV not found at '{static_csv}'.\n"
            "Run collect_data.py to gather static letter data first."
        )
    neither_seqs   = generate_neither_sequences(static_csv, NEITHER_SEQUENCES)
    neither_labels = ["Neither"] * len(neither_seqs)

    sequences += neither_seqs
    labels    += neither_labels

    print(f"\nTotal sequences: {Counter(labels)}\n")

    # Augment: add velocity and acceleration to each sequence
    print("Augmenting sequences with velocity and acceleration...")
    X_augmented = []
    for seq in sequences:
        augmented = extract_sequence_features(seq)   # (30, 213)
        X_augmented.append(augmented)

    X = np.array(X_augmented, dtype=np.float32)   # (N, 30, 213)

    # Encode labels
    le = LabelEncoder()
    le.classes_ = np.array(CLASS_NAMES)            # fix order: J=0, Z=1, Neither=2
    y = le.transform(labels)

    return X, y, le


# ── Preprocessing ─────────────────────────────────────────────────────────────

def scale_dataset(X_train, X_test):
    """
    Apply per-feature z-score normalisation across the training set.

    The scaler is fit only on training data and applied to both train and test
    to prevent data leakage.

    Input shape: (N, T, F) — flattened to (N*T, F) for fitting, then reshaped back.

    Returns
    -------
    X_train_scaled, X_test_scaled, scaler
    """
    N_train, T, F = X_train.shape
    N_test         = X_test.shape[0]

    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, F)
    X_test_flat  = X_test.reshape(-1, F)

    X_train_scaled = scaler.fit_transform(X_train_flat).reshape(N_train, T, F)
    X_test_scaled  = scaler.transform(X_test_flat).reshape(N_test,  T, F)

    return X_train_scaled.astype(np.float32), \
           X_test_scaled.astype(np.float32), \
           scaler


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(input_shape, n_classes: int) -> keras.Model:
    """
    Build the LSTM classifier.

    Architecture
    ────────────
    Input   → (batch, 30, 213)
    LSTM-1  → 128 units, return_sequences=True
    Dropout → 0.3
    LSTM-2  → 64 units
    Dropout → 0.3
    Dense   → 32 units, ReLU
    Output  → n_classes units, Softmax

    Parameters
    ----------
    input_shape : tuple  e.g. (30, 213)
    n_classes   : int    number of output classes (3)
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),

        layers.LSTM(128, return_sequences=True,
                    kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.Dropout(0.3),

        layers.LSTM(64,
                    kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.Dropout(0.3),

        layers.Dense(32, activation="relu"),

        layers.Dense(n_classes, activation="softmax"),
    ], name="lstm_jz_classifier")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── Training ──────────────────────────────────────────────────────────────────

def train(model: keras.Model, X_train, y_train, X_val, y_val):
    """
    Train the LSTM with early stopping and learning rate reduction.

    Returns
    -------
    history : keras History object
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    cb_list = [
        # Stop training if val_loss doesn't improve for PATIENCE epochs
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        # Reduce LR when training plateaus
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1,
        ),
        # Save the best checkpoint during training
        callbacks.ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, "lstm_jz_best.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
    ]

    print(f"\nTraining LSTM (max {MAX_EPOCHS} epochs, early stopping patience={PATIENCE})...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=cb_list,
        verbose=1,
    )
    return history


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model: keras.Model, X_test, y_test, le: LabelEncoder):
    os.makedirs(EVAL_DIR, exist_ok=True)

    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred       = np.argmax(y_pred_proba, axis=1)
    class_names  = le.classes_

    # ── Classification report ─────────────────────────────────────────────
    report = classification_report(y_test, y_pred, target_names=class_names)
    print("\nLSTM Classification Report:")
    print(report)

    report_path = os.path.join(EVAL_DIR, "lstm_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved report → {report_path}")

    # ── Confusion matrix ──────────────────────────────────────────────────
    cm   = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("LSTM — J / Z / Neither Confusion Matrix")
    plt.tight_layout()
    cm_path = os.path.join(EVAL_DIR, "lstm_confusion.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix → {cm_path}")

    # ── Per-class confidence analysis ─────────────────────────────────────
    print("\nMean prediction confidence per true class:")
    for cls_idx, cls_name in enumerate(class_names):
        mask = y_test == cls_idx
        if mask.sum() == 0:
            continue
        mean_conf = y_pred_proba[mask, cls_idx].mean()
        print(f"  {cls_name:8s}: {mean_conf:.1%} mean confidence on correct samples")


def plot_training_curve(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history["loss"],     label="Train loss")
    ax1.plot(history.history["val_loss"], label="Val loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training / Validation Loss")
    ax1.legend()

    ax2.plot(history.history["accuracy"],     label="Train acc")
    ax2.plot(history.history["val_accuracy"], label="Val acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training / Validation Accuracy")
    ax2.legend()

    plt.tight_layout()
    curve_path = os.path.join(EVAL_DIR, "lstm_training_curve.png")
    plt.savefig(curve_path, dpi=150)
    plt.close()
    print(f"Saved training curve → {curve_path}")


# ── Saving ────────────────────────────────────────────────────────────────────

def save_artifacts(model: keras.Model, scaler: StandardScaler):
    os.makedirs(MODEL_DIR, exist_ok=True)

    model_path  = os.path.join(MODEL_DIR, "lstm_jz.keras")
    scaler_path = os.path.join(MODEL_DIR, "lstm_scaler.pkl")

    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    print(f"\nLSTM model saved  → {model_path}")
    print(f"Scaler saved       → {scaler_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  ASL Fingerspelling — LSTM Trainer (J & Z)")
    print("=" * 55, "\n")

    static_csv = os.path.join(STATIC_DIR, "gesture_data.csv")

    # 1. Build dataset
    X, y, le = build_dataset(DYNAMIC_DIR, static_csv)
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"Classes: {list(zip(le.classes_, range(len(le.classes_))))}\n")

    # 2. Train / test split  (stratified to keep J/Z/Neither balanced)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = TEST_SIZE,
        stratify     = y,
        random_state = RANDOM_SEED,
    )

    # 3. Scale
    X_train, X_test, scaler = scale_dataset(X_train, X_test)

    # 4. Build model
    model = build_model(input_shape=(WINDOW_SIZE, AUGMENTED_DIMS),
                        n_classes=len(CLASS_NAMES))
    model.summary()

    # 5. Train  (20% of training set used as validation)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size    = 0.2,
        stratify     = y_train,
        random_state = RANDOM_SEED,
    )
    history = train(model, X_tr, y_tr, X_val, y_val)

    # 6. Plot training curve
    plot_training_curve(history)

    # 7. Evaluate on held-out test set
    print("\nEvaluating on test set...")
    evaluate(model, X_test, y_test, le)

    # 8. Save
    save_artifacts(model, scaler)

    print("\nLSTM training complete.")


if __name__ == "__main__":
    main()