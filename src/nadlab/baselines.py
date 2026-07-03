from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .data import FEATURE_COLUMNS


def window_features(windows: np.ndarray) -> np.ndarray:
    """Compact temporal windows into mean, deviation, last-point, and peak features."""
    if windows.ndim != 3:
        raise ValueError("Expected windows with shape [samples, time, features].")
    mean = windows.mean(axis=1)
    std = windows.std(axis=1)
    last = windows[:, -1, :]
    peak = np.abs(windows).max(axis=1)
    return np.concatenate([mean, std, last, peak], axis=1)


class ZScoreDetector:
    def fit(self, normal_windows: np.ndarray) -> "ZScoreDetector":
        features = window_features(normal_windows)
        self.mean_ = features.mean(axis=0)
        self.std_ = np.maximum(features.std(axis=0), 1e-6)
        return self

    def score_samples(self, windows: np.ndarray) -> np.ndarray:
        features = window_features(windows)
        return np.abs((features - self.mean_) / self.std_).max(axis=1)


class IsolationForestDetector:
    def __init__(self, seed: int = 42, n_estimators: int = 200) -> None:
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination="auto",
            random_state=seed,
            n_jobs=-1,
        )

    def fit(self, normal_windows: np.ndarray) -> "IsolationForestDetector":
        self.model.fit(window_features(normal_windows))
        return self

    def score_samples(self, windows: np.ndarray) -> np.ndarray:
        # sklearn returns higher values for normality; invert for anomaly ranking.
        return -self.model.score_samples(window_features(windows))


def rolling_mad_scores(frame: pd.DataFrame, lookback: int = 24) -> pd.Series:
    """Host-aware rolling median absolute deviation score without future values."""
    if lookback < 3:
        raise ValueError("lookback must be at least 3.")
    scores = pd.Series(index=frame.index, dtype=float)
    for _, group in frame.sort_values(["host_id", "timestamp"]).groupby("host_id", sort=False):
        values = group[FEATURE_COLUMNS].astype(float)
        history = values.shift(1)
        median = history.rolling(lookback, min_periods=max(3, lookback // 3)).median()
        abs_deviation = (history - median).abs()
        mad = abs_deviation.rolling(lookback, min_periods=max(3, lookback // 3)).median()
        robust_z = (values - median).abs() / (1.4826 * mad.replace(0, np.nan))
        host_scores = robust_z.replace([np.inf, -np.inf], np.nan).max(axis=1).fillna(0.0)
        scores.loc[group.index] = host_scores.to_numpy()
    return scores.sort_index()


def row_scores_to_window_scores(
    frame: pd.DataFrame,
    row_scores: pd.Series,
    window_size: int,
) -> np.ndarray:
    """Aggregate row anomaly evidence into the same host windows used by deep models."""
    output: list[float] = []
    ordered = frame.sort_values(["host_id", "timestamp"]).copy()
    ordered["_score"] = row_scores.loc[ordered.index].to_numpy()
    for _, group in ordered.groupby("host_id", sort=True):
        values = group["_score"].to_numpy(dtype=float)
        for end in range(window_size - 1, len(values)):
            output.append(float(np.max(values[end - window_size + 1 : end + 1])))
    return np.asarray(output, dtype=float)


def fit_threshold(normal_scores: np.ndarray, quantile: float = 0.995) -> float:
    if len(normal_scores) == 0:
        raise ValueError("Need normal scores to fit a threshold.")
    return float(np.quantile(normal_scores, quantile))
