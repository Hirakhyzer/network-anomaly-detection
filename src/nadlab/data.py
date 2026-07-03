from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Iterable

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "packets_in",
    "packets_out",
    "bytes_in",
    "bytes_out",
    "flow_count",
    "failed_logins",
    "dst_port_entropy",
    "protocol_diversity",
]

ANOMALY_TYPES = ("ddos", "port_scan", "brute_force", "exfiltration")


@dataclass(frozen=True)
class TelemetryConfig:
    n_hosts: int = 8
    n_steps: int = 900
    frequency_minutes: int = 5
    anomaly_fraction: float = 0.08
    normal_noise: float = 0.08
    seed: int = 42


def _positive(values: np.ndarray, floor: float = 0.0) -> np.ndarray:
    return np.maximum(values, floor)


def _inject_event(
    frame: pd.DataFrame,
    host_id: str,
    start: int,
    length: int,
    anomaly_type: str,
    severity: float,
) -> None:
    host_rows = frame.index[frame["host_id"] == host_id]
    selected = host_rows[start : start + length]
    if len(selected) == 0:
        return

    if anomaly_type == "ddos":
        frame.loc[selected, "packets_in"] *= 1 + 3.6 * severity
        frame.loc[selected, "bytes_in"] *= 1 + 2.8 * severity
        frame.loc[selected, "flow_count"] *= 1 + 2.2 * severity
        frame.loc[selected, "protocol_diversity"] *= 1 + 0.35 * severity
    elif anomaly_type == "port_scan":
        frame.loc[selected, "flow_count"] *= 1 + 3.1 * severity
        frame.loc[selected, "packets_out"] *= 1 + 1.5 * severity
        frame.loc[selected, "dst_port_entropy"] += 2.9 * severity
        frame.loc[selected, "protocol_diversity"] += 0.9 * severity
    elif anomaly_type == "brute_force":
        frame.loc[selected, "failed_logins"] += 18 * severity
        frame.loc[selected, "packets_in"] *= 1 + 0.65 * severity
        frame.loc[selected, "flow_count"] *= 1 + 0.9 * severity
    elif anomaly_type == "exfiltration":
        frame.loc[selected, "bytes_out"] *= 1 + 4.4 * severity
        frame.loc[selected, "packets_out"] *= 1 + 2.0 * severity
        frame.loc[selected, "flow_count"] *= 1 + 0.45 * severity
    else:
        raise ValueError(f"Unsupported anomaly type: {anomaly_type}")

    frame.loc[selected, "label"] = 1
    frame.loc[selected, "anomaly_type"] = anomaly_type
    frame.loc[selected, "severity"] = severity


def generate_synthetic_telemetry(config: TelemetryConfig | None = None) -> pd.DataFrame:
    """Generate labelled multi-host network telemetry with controlled anomalies.

    The data are synthetic and intended for reproducible methodology work, not
    claims about real enterprise traffic. Each host has a different baseline and
    a shared daily cycle, then labelled attack-shaped perturbations are injected.
    """
    config = config or TelemetryConfig()
    if config.n_hosts < 2 or config.n_steps < 50:
        raise ValueError("Use at least 2 hosts and 50 time steps.")
    if not 0 < config.anomaly_fraction < 0.45:
        raise ValueError("anomaly_fraction must be between 0 and 0.45.")

    rng = np.random.default_rng(config.seed)
    timeline = pd.date_range("2026-01-01", periods=config.n_steps, freq=f"{config.frequency_minutes}min")
    phase = np.linspace(0, 4 * np.pi, config.n_steps)
    daily = 1 + 0.18 * np.sin(phase) + 0.06 * np.cos(2 * phase)
    records: list[pd.DataFrame] = []

    for host_index in range(config.n_hosts):
        host_factor = rng.uniform(0.72, 1.35)
        noise = lambda scale: rng.normal(0, scale * config.normal_noise, config.n_steps)
        packets_in = _positive(230 * host_factor * daily * (1 + noise(1)), 5)
        packets_out = _positive(170 * host_factor * daily * (1 + noise(1)), 5)
        bytes_in = _positive(88000 * host_factor * daily * (1 + noise(1.3)), 100)
        bytes_out = _positive(69000 * host_factor * daily * (1 + noise(1.3)), 100)
        flow_count = _positive(46 * host_factor * daily * (1 + noise(1)), 1)
        failed_logins = _positive(rng.poisson(0.45 + host_index * 0.08, config.n_steps).astype(float), 0)
        dst_port_entropy = _positive(rng.normal(2.1 + host_index * 0.03, 0.18, config.n_steps), 0.1)
        protocol_diversity = _positive(rng.normal(3.0 + host_index * 0.04, 0.22, config.n_steps), 0.1)
        records.append(pd.DataFrame({
            "timestamp": timeline,
            "host_id": f"host-{host_index + 1:02d}",
            "packets_in": packets_in,
            "packets_out": packets_out,
            "bytes_in": bytes_in,
            "bytes_out": bytes_out,
            "flow_count": flow_count,
            "failed_logins": failed_logins,
            "dst_port_entropy": dst_port_entropy,
            "protocol_diversity": protocol_diversity,
            "label": 0,
            "anomaly_type": "normal",
            "severity": 0.0,
        }))

    frame = pd.concat(records, ignore_index=True)
    event_length = max(6, min(24, config.n_steps // 20))
    approx_events = ceil(config.n_hosts * config.n_steps * config.anomaly_fraction / event_length)
    for event_index in range(approx_events):
        host_id = f"host-{rng.integers(1, config.n_hosts + 1):02d}"
        start = int(rng.integers(20, max(21, config.n_steps - event_length - 1)))
        anomaly_type = ANOMALY_TYPES[event_index % len(ANOMALY_TYPES)]
        severity = float(rng.uniform(0.55, 1.0))
        _inject_event(frame, host_id, start, event_length, anomaly_type, severity)

    for column in FEATURE_COLUMNS:
        frame[column] = frame[column].astype(float).clip(lower=0)
    return frame.sort_values(["timestamp", "host_id"]).reset_index(drop=True)


def chronological_split(
    frame: pd.DataFrame,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by time, never random rows, to reduce temporal leakage."""
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("Fractions must be between 0 and 1.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train and validation fractions must leave a test period.")
    ordered = frame.sort_values("timestamp").reset_index(drop=True)
    timestamps = np.array(sorted(ordered["timestamp"].unique()))
    train_end = timestamps[int(len(timestamps) * train_fraction) - 1]
    validation_end = timestamps[int(len(timestamps) * (train_fraction + validation_fraction)) - 1]
    train = ordered[ordered["timestamp"] <= train_end].copy()
    validation = ordered[(ordered["timestamp"] > train_end) & (ordered["timestamp"] <= validation_end)].copy()
    test = ordered[ordered["timestamp"] > validation_end].copy()
    return train, validation, test


def anomaly_counts(frame: pd.DataFrame) -> pd.Series:
    return frame.loc[frame["label"] == 1, "anomaly_type"].value_counts().sort_index()


def validate_telemetry(frame: pd.DataFrame, required: Iterable[str] = FEATURE_COLUMNS) -> None:
    missing = set(required).difference(frame.columns)
    if missing:
        raise ValueError(f"Telemetry is missing columns: {sorted(missing)}")
    if not set(frame["label"].unique()).issubset({0, 1}):
        raise ValueError("label must contain only 0 and 1.")
    if frame[list(required)].isna().any().any():
        raise ValueError("Feature values cannot be missing.")
