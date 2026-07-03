import numpy as np

from nadlab.data import TelemetryConfig, chronological_split, generate_synthetic_telemetry, validate_telemetry


def test_generator_creates_labelled_multihost_telemetry():
    frame = generate_synthetic_telemetry(TelemetryConfig(n_hosts=4, n_steps=180, anomaly_fraction=0.10, seed=7))
    validate_telemetry(frame)
    assert len(frame) == 4 * 180
    assert frame["host_id"].nunique() == 4
    assert set(frame["label"].unique()).issubset({0, 1})
    assert frame["label"].sum() > 0
    assert frame.loc[frame["label"] == 1, "anomaly_type"].nunique() >= 3


def test_chronological_split_has_no_time_overlap():
    frame = generate_synthetic_telemetry(TelemetryConfig(n_hosts=3, n_steps=150, seed=2))
    train, validation, test = chronological_split(frame)
    assert train["timestamp"].max() < validation["timestamp"].min()
    assert validation["timestamp"].max() < test["timestamp"].min()
    assert len(train) + len(validation) + len(test) == len(frame)
