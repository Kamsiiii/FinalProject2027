"""
utils.py
────────
Feature extraction and normalisation for ASL fingerspelling recognition
using the Ultraleap Leap Motion Controller.

Each detected Hand object is converted into a fixed-length 71-dimensional
feature vector that is invariant to the absolute position, distance, and
gross orientation of the hand within the sensor field.

Feature vector layout (71 dimensions total):
  [0:5]   – Finger extension flags (binary, one per finger)
  [5:25]  – Bone pitch angles  (4 bones × 5 fingers, hand-local frame)
  [25:45] – Bone yaw angles    (4 bones × 5 fingers, hand-local frame)
  [45:65] – Bone roll angles   (4 bones × 5 fingers, hand-local frame)
  [65]    – Palm pitch
  [66]    – Palm yaw
  [67:71] – Inter-fingertip distances to thumb (normalised by hand span)
"""

import numpy as np


# ── Constants ─────────────────────────────────────────────────────────────────

NUM_FINGERS     = 5
NUM_BONES       = 4
BASE_FEATURES   = 61   # dimension of a single-frame feature vector

# Finger indices (Leap Motion convention)
THUMB   = 0
INDEX   = 1
MIDDLE  = 2
RING    = 3
LITTLE  = 4

# Bone indices (Leap Motion convention)
METACARPAL   = 0
PROXIMAL     = 1
INTERMEDIATE = 2
DISTAL       = 3


# ── Core feature extraction ───────────────────────────────────────────────────

def extract_features(hand) -> np.ndarray:
    """
    Convert a hand dict into a feature vector using tip positions
    and joint positions (v4 WebSocket compatible).
    """
    fingers = hand["fingers"]

    if len(fingers) != NUM_FINGERS:
        raise ValueError(
            "Expected " + str(NUM_FINGERS) + " fingers, got " + str(len(fingers))
        )

    palm_pos    = np.array(hand["palm_position"], dtype=np.float32)
    palm_normal = np.array(hand["palm_normal"],   dtype=np.float32)
    direction   = np.array(hand["direction"],     dtype=np.float32)

    # Build local frame for normalisation
    local_frame = _build_local_frame_from_vecs(direction, palm_normal)

    # Hand span for scale normalisation
    hand_span = _compute_hand_span_from_dict(hand)

    features = []

    # 1. Finger extension flags (5 dims)
    for f in fingers:
        features.append(float(f["is_extended"]))

    # 2. Normalised tip positions relative to palm (15 dims: 3 per finger)
    for f in fingers:
        tip = np.array(f["tip_position"], dtype=np.float32)
        rel = (tip - palm_pos) / (hand_span + 1e-6)
        local_rel = local_frame @ rel
        features.extend(local_rel.tolist())

    # 3. Normalised MCP joint positions relative to palm (15 dims)
    for f in fingers:
        mcp = np.array(f.get("mcp_position", f["tip_position"]), dtype=np.float32)
        rel = (mcp - palm_pos) / (hand_span + 1e-6)
        local_rel = local_frame @ rel
        features.extend(local_rel.tolist())

    # 4. Normalised PIP joint positions relative to palm (15 dims)
    for f in fingers:
        pip = np.array(f.get("pip_position", f["tip_position"]), dtype=np.float32)
        rel = (pip - palm_pos) / (hand_span + 1e-6)
        local_rel = local_frame @ rel
        features.extend(local_rel.tolist())

    # 5. Palm orientation (2 dims)
    local_normal = local_frame @ palm_normal
    palm_p, palm_y, _ = _direction_to_pyr(local_normal)
    features.append(palm_p)
    features.append(palm_y)

    # 6. Inter-fingertip distances to thumb (4 dims)
    thumb_tip = np.array(fingers[THUMB]["tip_position"], dtype=np.float32)
    for i in range(INDEX, NUM_FINGERS):
        tip  = np.array(fingers[i]["tip_position"], dtype=np.float32)
        dist = np.linalg.norm(tip - thumb_tip)
        features.append(dist / (hand_span + 1e-6))

    # 7. Finger curl angles from tip-to-mcp vectors (5 dims)
    for f in fingers:
        tip = np.array(f["tip_position"],                              dtype=np.float32)
        mcp = np.array(f.get("mcp_position", f["tip_position"]),       dtype=np.float32)
        vec = tip - mcp
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec = vec / norm
        curl = float(np.dot(vec, np.array([0, 1, 0], dtype=np.float32)))
        features.append(curl)

    vec = np.array(features, dtype=np.float32)
    return vec


def _build_local_frame_from_vecs(direction, palm_normal):
    x = _safe_normalise(np.array(direction,   dtype=np.float32))
    y = _safe_normalise(np.array(palm_normal, dtype=np.float32))
    z = _safe_normalise(np.cross(x, y))
    y = _safe_normalise(np.cross(z, x))
    return np.stack([x, y, z], axis=0)


def _compute_hand_span_from_dict(hand) -> float:
    index_base  = np.array(hand["fingers"][INDEX]["bones"][METACARPAL]["direction"], dtype=np.float32)
    little_base = np.array(hand["fingers"][LITTLE]["bones"][METACARPAL]["direction"], dtype=np.float32)
    return max(float(np.linalg.norm(index_base - little_base)), 1.0)


# ── Velocity and acceleration augmentation (for LSTM / dynamic signs) ────────

def extract_sequence_features(frame_sequence: list) -> np.ndarray:
    """
    Augment a sequence of base feature vectors with velocity and acceleration.

    Converts a list of raw Hand objects (or pre-extracted feature vectors)
    into a (T, 213) array where T is the sequence length, and 213 = 71 * 3
    (base features + first-order diff + second-order diff).

    Parameters
    ----------
    frame_sequence : list of leap.Hand or list of np.ndarray (shape 71,)
        A temporal window of hand observations.

    Returns
    -------
    np.ndarray, shape (T, 213), dtype float32
    """
    # Accept either raw Hand objects or pre-extracted vectors
    if hasattr(frame_sequence[0], 'fingers'):
        vectors = np.array([extract_features(h) for h in frame_sequence],
                           dtype=np.float32)
    else:
        vectors = np.array(frame_sequence, dtype=np.float32)

    T = vectors.shape[0]

    # First-order finite difference (velocity approximation)
    velocity = np.zeros_like(vectors)
    velocity[1:] = vectors[1:] - vectors[:-1]

    # Second-order finite difference (acceleration approximation)
    acceleration = np.zeros_like(vectors)
    acceleration[2:] = velocity[2:] - velocity[1:-1]

    augmented = np.concatenate([vectors, velocity, acceleration], axis=1)
    return augmented.astype(np.float32)   # shape: (T, 213)


# ── Outlier / quality check ───────────────────────────────────────────────────

def is_valid_frame(hand) -> bool:
    """
    Quick sanity check on a hand dict before feature extraction.
    """
    if hand is None:
        return False

    if "fingers" not in hand:
        return False

    if len(hand["fingers"]) != NUM_FINGERS:
        return False

    for finger in hand["fingers"]:
        for bone in finger["bones"]:
            direction = np.array(bone["direction"])
            if np.linalg.norm(direction) < 1e-4:
                return False

    return True


def filter_outliers(X: np.ndarray, threshold: float = 3.5) -> np.ndarray:
    """
    Remove rows from a feature matrix that are statistical outliers.

    Uses the interquartile range (IQR) method: a sample is flagged as an
    outlier if any of its feature values lies more than `threshold` IQRs
    from the median of that feature across the dataset.

    Parameters
    ----------
    X : np.ndarray, shape (N, 71)
    threshold : float
        Number of IQRs beyond which a sample is considered an outlier.

    Returns
    -------
    np.ndarray
        Filtered feature matrix with outlier rows removed.
    """
    q1  = np.percentile(X, 25, axis=0)
    q3  = np.percentile(X, 75, axis=0)
    iqr = q3 - q1

    # Avoid division by zero for near-constant features
    iqr = np.where(iqr < 1e-6, 1e-6, iqr)

    lower = q1 - threshold * iqr
    upper = q3 + threshold * iqr

    mask = np.all((X >= lower) & (X <= upper), axis=1)
    n_removed = np.sum(~mask)

    if n_removed > 0:
        print(f"  Outlier filter: removed {n_removed} / {len(X)} samples.")

    return X[mask]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _build_local_frame(hand) -> np.ndarray:
    """
    Build a 3×3 orthonormal rotation matrix representing the hand-local frame.

    The local frame is defined by:
      x-axis : palm direction (pointing from wrist toward fingers)
      y-axis : palm normal    (pointing out of the palm)
      z-axis : x cross y      (pointing to the side)

    All bone directions are projected into this frame so that rotating the
    whole hand does not change the extracted features.

    Returns
    -------
    np.ndarray, shape (3, 3)
        Each row is a basis vector [x_axis, y_axis, z_axis].
    """
    x_axis = _safe_normalise(np.array(hand.direction))
    y_axis = _safe_normalise(np.array(hand.palm_normal))
    z_axis = _safe_normalise(np.cross(x_axis, y_axis))

    # Re-orthogonalise y in case of floating-point drift
    y_axis = _safe_normalise(np.cross(z_axis, x_axis))

    return np.stack([x_axis, y_axis, z_axis], axis=0)   # shape (3, 3)


def _to_local(world_vector: np.ndarray, frame: np.ndarray) -> np.ndarray:
    """Project a world-space direction vector into the hand-local frame."""
    return frame @ world_vector   # matrix-vector multiply


def _direction_to_pyr(v: np.ndarray):
    """
    Convert a 3D direction vector to (pitch, yaw, roll) in radians.

    pitch : rotation around the x-axis (up/down tilt)
    yaw   : rotation around the y-axis (left/right sweep)
    roll  : rotation around the z-axis (axial rotation)

    All values are in [-pi, pi].
    """
    v = _safe_normalise(v)
    pitch = float(np.arcsin(np.clip(-v[1], -1.0, 1.0)))
    yaw   = float(np.arctan2(v[0], v[2]))
    roll  = float(np.arctan2(v[1], v[0]))
    return pitch, yaw, roll


def _compute_hand_span(hand) -> float:
    """
    Compute a scale-normalisation factor for the hand.

    Defined as the distance between the metacarpal base of the index finger
    and the metacarpal base of the little finger. This is stable across
    different hand poses and correlates well with overall hand size.
    """
    index_base  = np.array(hand.fingers[INDEX].bone(METACARPAL).prev_joint)
    little_base = np.array(hand.fingers[LITTLE].bone(METACARPAL).prev_joint)
    span = np.linalg.norm(index_base - little_base)
    return max(span, 1.0)   # clamp to avoid division by near-zero


def _safe_normalise(v: np.ndarray) -> np.ndarray:
    """Normalise a vector, returning a zero vector if the norm is too small."""
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-6 else np.zeros_like(v)


# ── Feature name lookup (for interpretability / debugging) ───────────────────

FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Little"]
BONE_NAMES   = ["Metacarpal", "Proximal", "Intermediate", "Distal"]

def feature_names() -> list:
    """
    Return human-readable names for all 61 feature dimensions.
    """
    names = []

    # Extension flags (5)
    for f in FINGER_NAMES:
        names.append(f"{f}_extended")

    # Tip positions (15)
    for f in FINGER_NAMES:
        for axis in ["x", "y", "z"]:
            names.append(f"{f}_tip_{axis}")

    # MCP positions (15)
    for f in FINGER_NAMES:
        for axis in ["x", "y", "z"]:
            names.append(f"{f}_mcp_{axis}")

    # PIP positions (15)
    for f in FINGER_NAMES:
        for axis in ["x", "y", "z"]:
            names.append(f"{f}_pip_{axis}")

    # Palm orientation (2)
    names.append("palm_pitch")
    names.append("palm_yaw")

    # Inter-fingertip distances (4)
    for f in FINGER_NAMES[1:]:
        names.append(f"{f}_to_thumb_dist")

    # Curl angles (5)
    for f in FINGER_NAMES:
        names.append(f"{f}_curl")

    return names