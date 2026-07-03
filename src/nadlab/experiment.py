from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .baselines import (
    IsolationForestDetector,
    ZScoreDetector,
    fit_threshold,
    rolling_mad_scores,
    row_scores_to_window_scores,
)
from .data import TelemetryConfig, chronological_split, generate_synthetic_telemetry
from .metrics import curve_frames, evaluate_scores, save_result_tables
from .models import (
    GraphTemporalAutoencoder,
    LSTMAutoencoder,
    TransformerAutoencoder,
    correlation_adjacency,
)
from .preprocessing import (
    build_host_tensor,
    fit_normal_scaler,
    make_graph_windows,
    make_windows,
    normal_only,
    transform_frame,
)
from .training import fit_reconstruction_model, score_reconstruction_model, set_seed


SUPPORTED_MODELS = ("zscore", "rolling_mad", "isolation_forest", "lstm_ae", "transformer_ae", "graph_temporal_ae")


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def prepare_experiment(config: dict[str, Any]) -> dict[str, Any]:
    data_cfg = config["data"]
    telemetry = generate_synthetic_telemetry(TelemetryConfig(
        n_hosts=int(data_cfg["n_hosts"]),
        n_steps=int(data_cfg["n_steps"]),
        frequency_minutes=int(data_cfg["frequency_minutes"]),
        anomaly_fraction=float(data_cfg["anomaly_fraction"]),
        normal_noise=float(data_cfg["normal_noise"]),
        seed=int(config["seed"]),
    ))
    train_raw, validation_raw, test_raw = chronological_split(
        telemetry,
        train_fraction=float(data_cfg["train_fraction"]),
        validation_fraction=float(data_cfg["validation_fraction"]),
    )
    scaler = fit_normal_scaler(train_raw)
    train = transform_frame(train_raw, scaler)
    validation = transform_frame(validation_raw, scaler)
    test = transform_frame(test_raw, scaler)
    window_size = int(data_cfg["window_size"])
    train_windows = make_windows(train, window_size)
    validation_windows = make_windows(validation, window_size)
    test_windows = make_windows(test, window_size)
    return {
        "telemetry": telemetry,
        "train_raw": train_raw,
        "validation_raw": validation_raw,
        "test_raw": test_raw,
        "train": train,
        "validation": validation,
        "test": test,
        "train_windows": train_windows,
        "validation_windows": validation_windows,
        "test_windows": test_windows,
        "window_size": window_size,
    }


def _result(model: str, labels: np.ndarray, scores: np.ndarray, normal_validation_scores: np.ndarray, timestamps: np.ndarray, hosts: np.ndarray, output_dir: Path, quantile: float) -> dict[str, float]:
    threshold = fit_threshold(normal_validation_scores, quantile)
    metrics, predictions = evaluate_scores(labels, scores, threshold, timestamps, hosts)
    roc, pr = curve_frames(labels, scores, model)
    save_result_tables(output_dir, model, metrics, predictions, roc, pr)
    return {"model": model, **metrics}


def run_baseline(model_name: str, prepared: dict[str, Any], config: dict[str, Any], output_dir: Path) -> dict[str, float]:
    train_windows = prepared["train_windows"]
    validation_windows = prepared["validation_windows"]
    test_windows = prepared["test_windows"]
    quantile = float(config["evaluation"]["alert_threshold_quantile"])
    if model_name == "zscore":
        detector = ZScoreDetector().fit(normal_only(train_windows))
        validation_scores = detector.score_samples(validation_windows.x)
        test_scores = detector.score_samples(test_windows.x)
    elif model_name == "isolation_forest":
        detector = IsolationForestDetector(seed=int(config["seed"])).fit(normal_only(train_windows))
        validation_scores = detector.score_samples(validation_windows.x)
        test_scores = detector.score_samples(test_windows.x)
    elif model_name == "rolling_mad":
        validation_rows = rolling_mad_scores(prepared["validation_raw"], prepared["window_size"])
        test_rows = rolling_mad_scores(prepared["test_raw"], prepared["window_size"])
        validation_scores = row_scores_to_window_scores(prepared["validation_raw"], validation_rows, prepared["window_size"])
        test_scores = row_scores_to_window_scores(prepared["test_raw"], test_rows, prepared["window_size"])
    else:
        raise ValueError(f"Unsupported baseline: {model_name}")
    return _result(
        model_name, test_windows.y, test_scores,
        validation_scores[validation_windows.y == 0], test_windows.timestamps,
        test_windows.hosts, output_dir, quantile,
    )


def run_sequence_autoencoder(model_name: str, prepared: dict[str, Any], config: dict[str, Any], output_dir: Path) -> dict[str, float]:
    models = config["models"]
    train_windows = prepared["train_windows"]
    validation_windows = prepared["validation_windows"]
    test_windows = prepared["test_windows"]
    n_features = train_windows.x.shape[-1]
    if model_name == "lstm_ae":
        model = LSTMAutoencoder(n_features, int(models["hidden_dim"]), int(models["latent_dim"]), float(models["dropout"]))
    elif model_name == "transformer_ae":
        model = TransformerAutoencoder(
            n_features=n_features,
            hidden_dim=int(models["hidden_dim"]),
            latent_dim=int(models["latent_dim"]),
            heads=int(models["transformer_heads"]),
            layers=int(models["transformer_layers"]),
            dropout=float(models["dropout"]),
            max_steps=max(512, prepared["window_size"]),
        )
    else:
        raise ValueError(f"Unsupported sequence autoencoder: {model_name}")
    fit_reconstruction_model(
        model, normal_only(train_windows), normal_only(validation_windows),
        epochs=int(models["epochs"]), batch_size=int(models["batch_size"]),
        learning_rate=float(models["learning_rate"]),
    )
    validation_scores = score_reconstruction_model(model, validation_windows.x)
    test_scores = score_reconstruction_model(model, test_windows.x)
    return _result(
        model_name, test_windows.y, test_scores,
        validation_scores[validation_windows.y == 0], test_windows.timestamps,
        test_windows.hosts, output_dir, float(models["threshold_quantile"]),
    )


def run_graph_autoencoder(prepared: dict[str, Any], config: dict[str, Any], output_dir: Path) -> dict[str, float]:
    models = config["models"]
    window_size = prepared["window_size"]
    train_tensor, train_row_labels, _, hosts = build_host_tensor(prepared["train"])
    normal_times = train_row_labels.max(axis=1) == 0
    adjacency = correlation_adjacency(train_tensor[normal_times])
    train_x, train_y, _, graph_hosts = make_graph_windows(prepared["train"], window_size, hosts)
    validation_x, validation_y, validation_timestamps, _ = make_graph_windows(prepared["validation"], window_size, graph_hosts)
    test_x, test_y, test_timestamps, _ = make_graph_windows(prepared["test"], window_size, graph_hosts)
    model = GraphTemporalAutoencoder(
        n_features=train_x.shape[-1], adjacency=adjacency,
        hidden_dim=int(models["hidden_dim"]), dropout=float(models["dropout"]),
    )
    fit_reconstruction_model(
        model, train_x[train_y == 0], validation_x[validation_y == 0],
        epochs=int(models["epochs"]), batch_size=int(models["batch_size"]),
        learning_rate=float(models["learning_rate"]),
    )
    validation_scores = score_reconstruction_model(model, validation_x)
    test_scores = score_reconstruction_model(model, test_x)
    return _result(
        "graph_temporal_ae", test_y, test_scores,
        validation_scores[validation_y == 0], test_timestamps,
        np.repeat("graph-window", len(test_timestamps)), output_dir,
        float(models["threshold_quantile"]),
    )


def run_experiment(config: dict[str, Any], requested_model: str = "all") -> pd.DataFrame:
    set_seed(int(config["seed"]))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = prepare_experiment(config)
    prepared["telemetry"].to_csv(output_dir / "synthetic_telemetry.csv", index=False)
    requested = list(SUPPORTED_MODELS) if requested_model == "all" else [requested_model]
    unknown = set(requested).difference(SUPPORTED_MODELS)
    if unknown:
        raise ValueError(f"Unsupported models: {sorted(unknown)}")
    results: list[dict[str, float]] = []
    for model_name in requested:
        if model_name in {"zscore", "rolling_mad", "isolation_forest"}:
            results.append(run_baseline(model_name, prepared, config, output_dir))
        elif model_name in {"lstm_ae", "transformer_ae"}:
            results.append(run_sequence_autoencoder(model_name, prepared, config, output_dir))
        else:
            results.append(run_graph_autoencoder(prepared, config, output_dir))

    new_results = pd.DataFrame(results)
    metrics_path = output_dir / "metrics_summary.csv"
    if metrics_path.exists():
        existing = pd.read_csv(metrics_path)
        existing = existing[~existing["model"].isin(requested)]
        summary = pd.concat([existing, new_results], ignore_index=True)
    else:
        summary = new_results
    summary = summary.sort_values("f1", ascending=False).reset_index(drop=True)
    summary.to_csv(metrics_path, index=False)
    manifest = {
        "seed": config["seed"],
        "models": requested,
        "rows": int(len(prepared["telemetry"])),
        "anomaly_rate": float(prepared["telemetry"]["label"].mean()),
        "window_size": prepared["window_size"],
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return summary
