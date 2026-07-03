import numpy as np

from nadlab.data import TelemetryConfig, chronological_split, generate_synthetic_telemetry
from nadlab.preprocessing import fit_normal_scaler, make_windows, normal_only, transform_frame


def test_windows_and_normal_scaler_are_shape_consistent():
    frame = generate_synthetic_telemetry(TelemetryConfig(n_hosts=3, n_steps=180, seed=11))
    train, validation, _ = chronological_split(frame)
    scaler = fit_normal_scaler(train)
    train_windows = make_windows(transform_frame(train, scaler), window_size=12)
    validation_windows = make_windows(transform_frame(validation, scaler), window_size=12)
    assert train_windows.x.ndim == 3
    assert train_windows.x.shape[-1] == 8
    assert train_windows.x.shape[0] == train_windows.y.shape[0]
    assert len(normal_only(train_windows)) > 0
    assert validation_windows.timestamps.shape[0] == validation_windows.x.shape[0]
