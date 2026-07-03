from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def _safe_metric(function, y_true: np.ndarray, values: np.ndarray, default: float = float("nan")) -> float:
    try:
        return float(function(y_true, values))
    except ValueError:
        return default


def detection_delays_minutes(
    labels: np.ndarray,
    predictions: np.ndarray,
    timestamps: np.ndarray,
    hosts: np.ndarray,
) -> tuple[float, int]:
    """Measure first alert delay for each contiguous anomalous run per host."""
    table = pd.DataFrame({"label": labels, "prediction": predictions, "timestamp": pd.to_datetime(timestamps), "host": hosts})
    delays: list[float] = []
    missed = 0
    for _, group in table.sort_values(["host", "timestamp"]).groupby("host", sort=False):
        anomaly = group["label"].to_numpy(dtype=int)
        predicted = group["prediction"].to_numpy(dtype=int)
        times = group["timestamp"].to_numpy()
        index = 0
        while index < len(group):
            if anomaly[index] != 1:
                index += 1
                continue
            start = index
            while index < len(group) and anomaly[index] == 1:
                index += 1
            stop = index
            hits = np.where(predicted[start:stop] == 1)[0]
            if len(hits) == 0:
                missed += 1
            else:
                first = start + int(hits[0])
                delay = (pd.Timestamp(times[first]) - pd.Timestamp(times[start])).total_seconds() / 60
                delays.append(float(delay))
    return (float(np.mean(delays)) if delays else float("nan"), missed)


def evaluate_scores(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    timestamps: np.ndarray | None = None,
    hosts: np.ndarray | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    y_true = np.asarray(labels, dtype=int)
    anomaly_scores = np.asarray(scores, dtype=float)
    predictions = (anomaly_scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    metrics = {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": _safe_metric(roc_auc_score, y_true, anomaly_scores),
        "pr_auc": _safe_metric(average_precision_score, y_true, anomaly_scores),
        "false_positive_rate": float(fp / max(1, fp + tn)),
        "true_positive": float(tp),
        "false_positive": float(fp),
        "true_negative": float(tn),
        "false_negative": float(fn),
    }
    if timestamps is not None and hosts is not None:
        delay, missed = detection_delays_minutes(y_true, predictions, timestamps, hosts)
        metrics["detection_delay_minutes"] = delay
        metrics["missed_anomaly_runs"] = float(missed)
    else:
        metrics["detection_delay_minutes"] = float("nan")
        metrics["missed_anomaly_runs"] = float("nan")
    predictions_frame = pd.DataFrame({"label": y_true, "score": anomaly_scores, "prediction": predictions})
    return metrics, predictions_frame


def curve_frames(labels: np.ndarray, scores: np.ndarray, model: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    y_true = np.asarray(labels, dtype=int)
    anomaly_scores = np.asarray(scores, dtype=float)
    fpr, tpr, roc_thresholds = roc_curve(y_true, anomaly_scores)
    precision, recall, pr_thresholds = precision_recall_curve(y_true, anomaly_scores)
    roc = pd.DataFrame({"model": model, "fpr": fpr, "tpr": tpr, "threshold": roc_thresholds})
    pr = pd.DataFrame({"model": model, "precision": precision, "recall": recall, "threshold": np.append(pr_thresholds, np.nan)})
    return roc, pr


def save_result_tables(
    output_dir: str | Path,
    model: str,
    metrics: dict[str, float],
    predictions: pd.DataFrame,
    roc: pd.DataFrame,
    pr: pd.DataFrame,
) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(destination / f"predictions_{model}.csv", index=False)
    roc.to_csv(destination / f"roc_{model}.csv", index=False)
    pr.to_csv(destination / f"pr_{model}.csv", index=False)
    metrics_frame = pd.DataFrame([{**metrics, "model": model}])
    metrics_path = destination / "metrics_summary.csv"
    if metrics_path.exists():
        existing = pd.read_csv(metrics_path)
        existing = existing[existing["model"] != model]
        metrics_frame = pd.concat([existing, metrics_frame], ignore_index=True)
    metrics_frame.sort_values("model").to_csv(metrics_path, index=False)
