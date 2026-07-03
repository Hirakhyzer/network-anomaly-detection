from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .data import FEATURE_COLUMNS, validate_telemetry


@dataclass
class WindowedDataset:
    x: np.ndarray
    y: np.ndarray
    timestamps: np.ndarray
    hosts: np.ndarray


def fit_normal_scaler(train_frame: pd.DataFrame) -> StandardScaler:
    """Fit scaling only on normal training observations to avoid label leakage."""
    validate_telemetry(train_frame)
    normal = train_frame.loc[train_frame["label"] == 0, FEATURE_COLUMNS]
    if normal.empty:
        raise ValueError("Training period has no normal observations for scaling.")
    return StandardScaler().fit(normal)


def transform_frame(frame: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    validate_telemetry(frame)
    transformed = frame.copy()
    transformed.loc[:, FEATURE_COLUMNS] = scaler.transform(transformed[FEATURE_COLUMNS])
    return transformed


def make_windows(frame: pd.DataFrame, window_size: int) -> WindowedDataset:
    """Return host-specific windows labelled anomalous if any row in the window is anomalous."""
    if window_size < 2:
        raise ValueError("window_size must be at least 2.")
    validate_telemetry(frame)
    ordered = frame.sort_values(["host_id", "timestamp"]).reset_index(drop=True)
    windows: list[np.ndarray] = []
    labels: list[int] = []
    timestamps: list[np.datetime64] = []
    hosts: list[str] = []
    for host, group in ordered.groupby("host_id", sort=True):
        values = group[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        group_labels = group["label"].to_numpy(dtype=np.int64)
        group_timestamps = group["timestamp"].to_numpy()
        for end in range(window_size - 1, len(group)):
            begin = end - window_size + 1
            windows.append(values[begin : end + 1])
            labels.append(int(group_labels[begin : end + 1].max()))
            timestamps.append(group_timestamps[end])
            hosts.append(str(host))
    if not windows:
        raise ValueError("Not enough rows per host for the requested window_size.")
    return WindowedDataset(
        x=np.stack(windows).astype(np.float32),
        y=np.asarray(labels, dtype=np.int64),
        timestamps=np.asarray(timestamps),
        hosts=np.asarray(hosts),
    )


def build_host_tensor(frame: pd.DataFrame, host_order: list[str] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Build time x host x feature tensor for graph-temporal experiments."""
    validate_telemetry(frame)
    hosts = host_order or sorted(frame["host_id"].unique().tolist())
    times = np.array(sorted(frame["timestamp"].unique()))
    tensor = np.zeros((len(times), len(hosts), len(FEATURE_COLUMNS)), dtype=np.float32)
    labels = np.zeros((len(times), len(hosts)), dtype=np.int64)
    time_to_index = {time: index for index, time in enumerate(times)}
    host_to_index = {host: index for index, host in enumerate(hosts)}
    for row in frame.itertuples(index=False):
        time_index = time_to_index[row.timestamp]
        host_index = host_to_index[row.host_id]
        tensor[time_index, host_index] = np.asarray([getattr(row, feature) for feature in FEATURE_COLUMNS], dtype=np.float32)
        labels[time_index, host_index] = int(row.label)
    return tensor, labels, times, hosts


def make_graph_windows(frame: pd.DataFrame, window_size: int, host_order: list[str] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    tensor, labels, times, hosts = build_host_tensor(frame, host_order)
    if len(tensor) < window_size:
        raise ValueError("Not enough time steps for graph windows.")
    windows = []
    window_labels = []
    end_times = []
    for end in range(window_size - 1, len(tensor)):
        begin = end - window_size + 1
        windows.append(tensor[begin : end + 1])
        window_labels.append(int(labels[begin : end + 1].max()))
        end_times.append(times[end])
    return np.stack(windows).astype(np.float32), np.asarray(window_labels, dtype=np.int64), np.asarray(end_times), hosts


def normal_only(dataset: WindowedDataset) -> np.ndarray:
    values = dataset.x[dataset.y == 0]
    if len(values) == 0:
        raise ValueError("No normal windows available.")
    return values
