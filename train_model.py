"""
train_model.py
──────────────
Trains a Random Forest classifier on the collected ASL fingerspelling
dataset (24 static letters) and saves the model for real-time inference.

Usage
-----
    python train_model.py

Outputs
-------
    models/rf_model.pkl          – Trained Random Forest classifier
    models/label_encoder.pkl     – Fitted LabelEncoder for class mapping
    evaluation/classification_report.txt
    evaluation/confusion_matrix.png
"""

import os
import csv
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from collections import Counter

from sklearn.ensemble            import RandomForestClassifier
from sklearn.model_selection     import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing       import LabelEncoder
from sklearn.metrics             import (classification_report,
                                         confusion_matrix,
                                         ConfusionMatrixDisplay)
from utils import filter_outliers, feature_names, BASE_FEATURES


# ── Config ────────────────────────────────────────────────────────────────────

DATA_FILE       = "data/static/gesture_data.csv"
MODEL_DIR       = "models"
EVAL_DIR        = "evaluation"

# Dynamic letters handled separately by LSTM — exclude from this classifier
DYNAMIC_LETTERS = {"J", "Z"}

# Random Forest hyperparameters
RF_PARAMS = dict(
    n_estimators  = 300,
    max_features  = "sqrt",
    max_depth     = None,
    min_samples_leaf = 2,
    class_weight  = "balanced",
    random_state  = 42,
    n_jobs        = -1,
)

TEST_SIZE   = 0.20
RANDOM_SEED = 42

# High-confusion letter groups (used for targeted analysis in the report)
CONFUSION_GROUPS = [
    {"A", "E", "S"},
    {"M", "N"},
    {"D", "G"},
    {"R", "U", "V"},
]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_dataset(filepath: str):
    """
    Load the CSV dataset into feature matrix X and label array y.

    CSV format: label, f0, f1, ..., f70
    Rows for dynamic letters J and Z are excluded.

    Returns
    -------
    X : np.ndarray, shape (N, 71)
    y : np.ndarray, shape (N,), dtype str
    """
    X_rows, y_rows = [], []

    with open(filepath, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            label = row[0].strip().upper()
            if label in DYNAMIC_LETTERS:
                continue
            if len(row) - 1 != BASE_FEATURES:
                print(f"  Skipping malformed row for '{label}' "
                      f"(got {len(row)-1} features, expected {BASE_FEATURES})")
                continue
            y_rows.append(label)
            X_rows.append([float(v) for v in row[1:]])

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=str)

    print(f"Loaded {len(y)} samples across {len(set(y))} classes.")
    _print_class_distribution(y)

    return X, y


def _print_class_distribution(y):
    counts = Counter(y)
    print("\nClass distribution:")
    for letter in sorted(counts):
        bar = "█" * (counts[letter] // 5)
        print(f"  {letter}: {counts[letter]:4d}  {bar}")
    print()


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(X: np.ndarray, y: np.ndarray):
    """
    Prepares the raw dataset for training.

    Parameters
    ----------
    X : np.ndarray, shape (N, 71)
        The feature matrix — N samples, each with 71 feature values.
    y : np.ndarray, shape (N,)
        The label array — N strings like "A", "B", "C" etc.

    Returns
    -------
    X : np.ndarray
        The feature matrix (unchanged, outlier filter disabled).
    y_encoded : np.ndarray
        Labels converted from strings to integers.
        e.g. "A" -> 0, "B" -> 1, "C" -> 2 etc.
        The model needs numbers, not strings.
    le : LabelEncoder
        The fitted encoder — saved alongside the model so we can
        convert integer predictions BACK to letters at inference time.
        e.g. prediction of 0 -> "A", 1 -> "B" etc.
    """

    # LabelEncoder converts string labels to integers
    # fit_transform() learns the mapping AND applies it in one step
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # le.classes_ shows the mapping e.g. ['A', 'B', 'C' ...]
    # index 0 = A, index 1 = B etc.
    print(f"Classes after encoding: {list(le.classes_)}\n")

    # Return X unchanged, the encoded labels, and the encoder itself
    return X, y_encoded, le


# ── Training ──────────────────────────────────────────────────────────────────

def train(X_train, y_train, tune_hyperparams=False):
    """
    Train a Random Forest classifier.

    Parameters
    ----------
    tune_hyperparams : bool
        If True, run a grid search over n_estimators and min_samples_leaf.
        Takes longer but may improve accuracy.

    Returns
    -------
    RandomForestClassifier
    """
    if tune_hyperparams:
        print("Running hyperparameter grid search (this may take a few minutes)...")
        param_grid = {
            "n_estimators":      [100, 200, 300],
            "min_samples_leaf":  [1, 2, 5],
        }
        base_rf = RandomForestClassifier(
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        search = GridSearchCV(base_rf, param_grid, cv=cv,
                              scoring="f1_macro", n_jobs=-1, verbose=1)
        search.fit(X_train, y_train)
        print(f"Best params: {search.best_params_}")
        print(f"Best CV F1 (macro): {search.best_score_:.4f}\n")
        return search.best_estimator_

    else:
        print("Training Random Forest...")
        rf = RandomForestClassifier(**RF_PARAMS)
        rf.fit(X_train, y_train)
        return rf


def cross_validate(X, y):
    """
    Run 5-fold stratified cross-validation and report mean / std F1.
    """
    print("Running 5-fold cross-validation...")
    cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    rf  = RandomForestClassifier(**RF_PARAMS)
    f1_scores = []

    from sklearn.metrics import f1_score

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
        rf.fit(X[train_idx], y[train_idx])
        preds = rf.predict(X[val_idx])
        f1 = f1_score(y[val_idx], preds, average="macro")
        f1_scores.append(f1)
        print(f"  Fold {fold}: F1 = {f1:.4f}")

    mean_f1 = np.mean(f1_scores)
    std_f1  = np.std(f1_scores)
    print(f"\nCV F1 (macro): {mean_f1:.4f} ± {std_f1:.4f}\n")
    return mean_f1, std_f1


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, X_test, y_test, le: LabelEncoder):
    """
    Print and save the classification report and confusion matrix.
    """
    os.makedirs(EVAL_DIR, exist_ok=True)
    y_pred = model.predict(X_test)

    class_names = le.classes_

    # ── Classification report ─────────────────────────────────────────────
    report = classification_report(y_test, y_pred,
                                   target_names=class_names)
    print("Classification Report:")
    print(report)

    report_path = os.path.join(EVAL_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved report → {report_path}")

    # ── Confusion matrix ──────────────────────────────────────────────────
    _save_confusion_matrix(y_test, y_pred, class_names)

    # ── Per-confusion-group breakdown ─────────────────────────────────────
    _confusion_group_analysis(y_test, y_pred, le)

    # ── Feature importances ───────────────────────────────────────────────
    _save_feature_importances(model)


def _save_confusion_matrix(y_test, y_pred, class_names):
    cm   = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=class_names)

    fig, ax = plt.subplots(figsize=(14, 12))
    disp.plot(ax=ax, colorbar=True, cmap="Blues", xticks_rotation=45)
    ax.set_title("ASL Fingerspelling — Confusion Matrix", fontsize=14, pad=16)
    plt.tight_layout()

    path = os.path.join(EVAL_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix → {path}")


def _confusion_group_analysis(y_test, y_pred, le: LabelEncoder):
    """
    For each high-confusion letter group, print within-group accuracy
    to identify which letters are hardest to distinguish.
    """
    print("\nHigh-Confusion Group Analysis:")
    print("─" * 40)
    for group in CONFUSION_GROUPS:
        # Map letter names to encoded integers
        group_encoded = [le.transform([l])[0] for l in group
                         if l in le.classes_]
        if len(group_encoded) < 2:
            continue

        mask = np.isin(y_test, group_encoded)
        if mask.sum() == 0:
            continue

        group_true = y_test[mask]
        group_pred = y_pred[mask]
        acc = np.mean(group_true == group_pred)

        print(f"  {'/'.join(sorted(group))}: within-group accuracy = {acc:.1%}  "
              f"({mask.sum()} samples)")
    print()


def _save_feature_importances(model):
    """Save a bar chart of the top-20 most important features."""
    importances = model.feature_importances_
    names       = feature_names()
    indices     = np.argsort(importances)[::-1][:20]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh([names[i] for i in reversed(indices)],
            [importances[i] for i in reversed(indices)],
            color="steelblue")
    ax.set_xlabel("Mean Decrease in Impurity")
    ax.set_title("Top 20 Feature Importances")
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    plt.tight_layout()

    path = os.path.join(EVAL_DIR, "feature_importances.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved feature importances → {path}")


# ── Saving ────────────────────────────────────────────────────────────────────

def save_model(model, le: LabelEncoder):
    os.makedirs(MODEL_DIR, exist_ok=True)

    model_path = os.path.join(MODEL_DIR, "rf_model.pkl")
    le_path    = os.path.join(MODEL_DIR, "label_encoder.pkl")

    joblib.dump(model, model_path)
    joblib.dump(le,    le_path)

    print(f"\nModel saved   → {model_path}")
    print(f"Encoder saved → {le_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  ASL Fingerspelling — Random Forest Trainer")
    print("=" * 55, "\n")

    # 1. Load
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"Dataset not found at '{DATA_FILE}'.\n"
            "Run collect_data.py first to gather training samples."
        )
    X, y = load_dataset(DATA_FILE)

    # 2. Preprocess
    X, y_enc, le = preprocess(X, y)

    # 3. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc,
        test_size    = TEST_SIZE,
        stratify     = y_enc,
        random_state = RANDOM_SEED,
    )
    print(f"Train: {len(y_train)} samples  |  Test: {len(y_test)} samples\n")

    # 4. Cross-validate to check stability
    cross_validate(X_train, y_train)

    # 5. Train final model on full training set
    #    Set tune_hyperparams=True to run a grid search instead
    model = train(X_train, y_train, tune_hyperparams=False)

    # 6. Evaluate on held-out test set
    print("\nEvaluating on test set...")
    evaluate(model, X_test, y_test, le)

    # 7. Save
    save_model(model, le)

    print("\nTraining complete.")


if __name__ == "__main__":
    main()