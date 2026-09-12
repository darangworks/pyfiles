# test_causal_knn_corrected.py

import numpy as np
import pandas as pd
import pytest
from collections import deque


# ═══════════════════════════════════════════════════════════════
# Synthetic data helpers
# ═══════════════════════════════════════════════════════════════

def synthetic_close(n: int = 1200, seed: int = 42) -> pd.Series:
    """
    Synthetic close prices with drift and volatility.
    Not used as evidence of edge; only for logic testing.
    """
    rng = np.random.default_rng(seed)
    ret = rng.normal(loc=0.0002, scale=0.01, size=n)
    close = 100.0 * np.exp(np.cumsum(ret))
    return pd.Series(close)


def make_atr(close: pd.Series, length: int = 14) -> pd.Series:
    """
    Close-based ATR proxy.

    This is not the exact Pine Script ATR because we do not have
    high/low data here. It is sufficient for causal logic tests.
    """
    true_range_proxy = close.diff().abs()
    atr = true_range_proxy.rolling(length).mean()
    atr = atr.ffill()
    atr = atr.fillna(close.abs() * 0.01)
    return atr


# ═══════════════════════════════════════════════════════════════
# Causal feature builder
# ═══════════════════════════════════════════════════════════════

def make_features(
    close: pd.Series,
    atr: pd.Series,
    t: int,
    pattern_len: int
) -> np.ndarray:
    """
    Build a causal feature vector at bar t.

    Important:
    - Only data at bar t or earlier may be used.
    - Pine Script close[n] corresponds to close.iloc[t - n].
    """

    feats = []

    for j in range(pattern_len):
        bars_ago = pattern_len - 1 - j
        idx = t - bars_ago

        # Require enough history for the slow moving average proxy.
        if idx < 30:
            feats.extend([0.0, 0.0, 0.0, 0.0])
            continue

        c = close.iloc[idx]
        c1 = close.iloc[idx - 1]
        c5 = close.iloc[idx - 5]
        a = atr.iloc[idx]

        if (
            not np.isfinite(c)
            or not np.isfinite(a)
            or c <= 0.0
            or a <= 0.0
        ):
            feats.extend([0.0, 0.0, 0.0, 0.0])
            continue

        vol = max(a / c, 1e-12)

        # Feature 1: volatility-normalized log return
        f1 = np.log(c / c1) / vol if c1 > 0.0 else 0.0

        # Feature 2: 5-bar momentum normalized by ATR
        f2 = (c - c5) / a

        # Feature 3: bounded momentum proxy
        # This replaces RSI for the Python logic test.
        f3 = np.tanh((c / c5 - 1.0) * 20.0) if c5 > 0.0 else 0.0

        # Feature 4: fast/slow mean spread normalized by ATR
        fast = close.iloc[idx - 9:idx + 1].mean()
        slow = close.iloc[idx - 29:idx + 1].mean()
        f4 = (fast - slow) / a

        clipped = np.clip([f1, f2, f3, f4], -5.0, 5.0)
        feats.extend(clipped.tolist())

    return np.asarray(feats, dtype=float)


# ═══════════════════════════════════════════════════════════════
# Causal kNN reference implementation
# ═══════════════════════════════════════════════════════════════

class CausalKNN:
    """
    Causal delayed-label kNN model.

    At bar t:
      1. Mature pending samples whose label is now known.
         A sample from bar s matures when t >= s + ahead.
      2. Add current query pattern to pending, but not to training.
      3. Predict only from matured training samples.
    """

    def __init__(
        self,
        pattern_len: int = 10,
        memory_size: int = 80,
        k: int = 5,
        ahead: int = 2
    ):
        self.pattern_len = pattern_len
        self.memory_size = memory_size
        self.k = k
        self.ahead = ahead
        self.n_features = pattern_len * 4

        self.X = np.zeros((memory_size, self.n_features), dtype=float)
        self.y = np.zeros(memory_size, dtype=float)

        self.head = 0
        self.count = 0

        # Pending samples: (bar_index, feature_vector)
        self.pending = deque()

        # Diagnostics for testing
        self.matured_bars = []
        self.matured_at = []
        self.matured_labels = []

    def on_bar(
        self,
        t: int,
        query: np.ndarray,
        close: pd.Series
    ):
        query = np.asarray(query, dtype=float)

        if query.size != self.n_features:
            raise ValueError(
                f"Query size {query.size} != expected {self.n_features}"
            )

        # -------------------------------------------------------
        # 1. Mature pending samples whose forward return is known.
        # -------------------------------------------------------
        while self.pending and t >= self.pending[0][0] + self.ahead:
            stored_t, stored_x = self.pending.popleft()
            future_t = stored_t + self.ahead

            if future_t >= len(close):
                break

            base_price = close.iloc[stored_t]
            future_price = close.iloc[future_t]

            if base_price > 0.0 and future_price > 0.0:
                label = float(np.log(future_price / base_price))
            else:
                label = 0.0

            write_idx = self.head % self.memory_size

            self.X[write_idx] = stored_x
            self.y[write_idx] = label

            self.head += 1
            self.count = min(self.count + 1, self.memory_size)

            self.matured_bars.append(stored_t)
            self.matured_at.append(t)
            self.matured_labels.append(label)

        # -------------------------------------------------------
        # 2. Store current pattern as pending only.
        #    It must not be usable for prediction until matured.
        # -------------------------------------------------------
        self.pending.append((t, query.copy()))

        # -------------------------------------------------------
        # 3. Predict using matured samples only.
        # -------------------------------------------------------
        return self.predict(query)

    def predict(self, query: np.ndarray):
        """
        Pure prediction function. Does not mutate state.
        """

        if self.count < self.k:
            return None

        if self.count < self.memory_size:
            X_valid = self.X[:self.count]
            y_valid = self.y[:self.count]
        else:
            X_valid = self.X
            y_valid = self.y

        distances = np.linalg.norm(X_valid - query, axis=1)

        k = min(self.k, len(distances))

        if k < len(distances):
            top = np.argpartition(distances, k - 1)[:k]
        else:
            top = np.argsort(distances)

        d_top = distances[top]
        y_top = y_valid[top]

        weights = 1.0 / (d_top + 1e-9)
        weight_sum = weights.sum()

        if weight_sum <= 0.0:
            return None

        return float((weights * y_top).sum() / weight_sum)


# ═══════════════════════════════════════════════════════════════
# TEST 1
# Current bar must not enter the training set on the same bar.
# ═══════════════════════════════════════════════════════════════

def test_current_bar_never_enters_training_same_bar():
    close = synthetic_close(800, seed=11)
    atr = make_atr(close)
    model = CausalKNN(pattern_len=10, memory_size=80, k=5, ahead=2)

    start = 50

    for t in range(start, 300):
        q = make_features(close, atr, t, pattern_len=10)
        model.on_bar(t, q, close)

        # Current bar must be pending, not matured.
        assert model.pending[-1][0] == t

        # No sample from the current bar may already be matured.
        assert t not in model.matured_bars

        # The newest matured sample must be at least 'ahead' bars old.
        if model.matured_bars:
            assert max(model.matured_bars) <= t - model.ahead


# ═══════════════════════════════════════════════════════════════
# TEST 2
# Labels must be true forward returns, known only after horizon.
# ═══════════════════════════════════════════════════════════════

def test_labels_are_forward_returns_known_only_after_horizon():
    close = synthetic_close(1000, seed=22)
    atr = make_atr(close)
    model = CausalKNN(pattern_len=10, memory_size=80, k=5, ahead=2)

    start = 60

    for t in range(start, 500):
        q = make_features(close, atr, t, pattern_len=10)
        model.on_bar(t, q, close)

    assert len(model.matured_bars) > 0

    for stored_t, matured_at, label in zip(
        model.matured_bars,
        model.matured_at,
        model.matured_labels
    ):
        future_t = stored_t + model.ahead

        # The label may only be written at or after the future bar.
        assert matured_at >= future_t

        expected = np.log(close.iloc[future_t] / close.iloc[stored_t])
        assert np.isclose(label, expected, rtol=1e-12)


# ═══════════════════════════════════════════════════════════════
# TEST 3
# Ring buffer must never exceed memory_size.
# ═══════════════════════════════════════════════════════════════

def test_ring_buffer_never_exceeds_memory():
    close = synthetic_close(1500, seed=33)
    atr = make_atr(close)

    memory_size = 50
    model = CausalKNN(
        pattern_len=10,
        memory_size=memory_size,
        k=5,
        ahead=2
    )

    start = 60

    for t in range(start, 1300):
        q = make_features(close, atr, t, pattern_len=10)
        model.on_bar(t, q, close)
        assert model.count <= memory_size

    assert model.count == memory_size
    assert model.head > memory_size


# ═══════════════════════════════════════════════════════════════
# TEST 4
# Prediction must be a normalized inverse-distance weighted mean.
# ═══════════════════════════════════════════════════════════════

def test_prediction_is_normalized_inverse_distance_weighted_mean():
    model = CausalKNN(pattern_len=10, memory_size=10, k=3, ahead=2)

    n_features = model.n_features

    # Manually create three matured samples.
    model.X[:3] = 0.0
    model.X[1, 0] = 1.0
    model.X[2, 0] = 2.0

    model.y[:3] = np.array([0.02, 0.04, 0.06])
    model.count = 3

    query = np.zeros(n_features, dtype=float)

    pred = model.predict(query)

    # Distances are exactly 0, 1, 2.
    d = np.array([0.0, 1.0, 2.0])
    w = 1.0 / (d + 1e-9)
    expected = float((w * model.y[:3]).sum() / w.sum())

    assert pred is not None
    assert np.isclose(pred, expected, rtol=1e-9)

    # Because distance zero dominates, prediction should be near 0.02.
    assert pred < 0.0201


# ═══════════════════════════════════════════════════════════════
# TEST 5
# No prediction before k mature samples exist.
# ═══════════════════════════════════════════════════════════════

def test_no_prediction_before_k_mature_samples():
    close = synthetic_close(600, seed=44)
    atr = make_atr(close)

    k = 5
    ahead = 2
    model = CausalKNN(pattern_len=10, memory_size=80, k=k, ahead=ahead)

    start = 60

    # With ahead = 2:
    # t = start + 2 -> first mature sample
    # t = start + 2 + k - 2 -> kth mature sample appears
    # Predictions should be None before the kth mature sample.
    no_prediction_end_exclusive = start + ahead + k - 1

    for t in range(start, no_prediction_end_exclusive):
        q = make_features(close, atr, t, pattern_len=10)
        pred = model.on_bar(t, q, close)
        assert pred is None

    # On the next bar, enough samples should have matured.
    t = no_prediction_end_exclusive
    q = make_features(close, atr, t, pattern_len=10)
    pred = model.on_bar(t, q, close)

    assert model.count >= k
    assert pred is not None


# ═══════════════════════════════════════════════════════════════
# Random-walk helper
# ═══════════════════════════════════════════════════════════════

def random_walk_accuracy(seed: int = 7, n: int = 2500) -> float:
    """
    Run the causal kNN model on a pure random walk.

    If the pipeline is causal and leakage-free, directional accuracy
    should be close to 50%.
    """

    rng = np.random.default_rng(seed)
    ret = rng.normal(loc=0.0, scale=0.01, size=n)
    close = pd.Series(100.0 * np.exp(np.cumsum(ret)))

    atr = make_atr(close)

    model = CausalKNN(
        pattern_len=10,
        memory_size=120,
        k=5,
        ahead=2
    )

    hits = 0
    total = 0

    start = 80

    for t in range(start, n - model.ahead):
        q = make_features(close, atr, t, pattern_len=10)
        pred = model.on_bar(t, q, close)

        if pred is None:
            continue

        actual_ret = close.iloc[t + model.ahead] / close.iloc[t] - 1.0

        if abs(actual_ret) < 1e-12:
            continue

        if abs(pred) < 1e-12:
            continue

        if np.sign(pred) == np.sign(actual_ret):
            hits += 1

        total += 1

    if total == 0:
        return 0.5

    return hits / total


# ═══════════════════════════════════════════════════════════════
# TEST 6
# Random walk must not produce systematic predictive edge.
# ═══════════════════════════════════════════════════════════════

def test_random_walk_has_no_predictive_edge():
    """
    This is the primary leakage oracle.

    On a pure random walk, a causal model should not achieve
    persistent directional accuracy materially different from 50%.
    """

    seeds = (101, 202, 303)
    accs = [random_walk_accuracy(seed=seed, n=2500) for seed in seeds]

    mean_acc = float(np.mean(accs))

    # Individual runs may fluctuate.
    for acc in accs:
        assert 0.42 < acc < 0.58, f"Individual random-walk accuracy suspicious: {acc:.4f}"

    # The average should be close to chance.
    assert 0.46 < mean_acc < 0.54, (
        f"Mean random-walk accuracy {mean_acc:.4f} is too far from 0.50. "
        "This strongly suggests leakage in the feature builder, labeler, "
        "or prediction pipeline."
    )


# ═══════════════════════════════════════════════════════════════
# TEST 7
# Determinism.
# ═══════════════════════════════════════════════════════════════

def _run_prediction_sequence(seed: int = 99, n: int = 900):
    close = synthetic_close(n, seed=seed)
    atr = make_atr(close)

    model = CausalKNN(
        pattern_len=10,
        memory_size=80,
        k=5,
        ahead=2
    )

    preds = []

    start = 70

    for t in range(start, n - model.ahead):
        q = make_features(close, atr, t, pattern_len=10)
        pred = model.on_bar(t, q, close)
        if pred is not None:
            preds.append(pred)

    return preds


def test_determinism():
    preds_1 = _run_prediction_sequence(seed=123)
    preds_2 = _run_prediction_sequence(seed=123)

    assert len(preds_1) > 0
    assert preds_1 == preds_2