from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nadlab.data import TelemetryConfig, anomaly_counts, generate_synthetic_telemetry
from nadlab.experiment import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate labelled synthetic network telemetry.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default="data/synthetic_telemetry.csv")
    args = parser.parse_args()
    config = load_config(args.config)
    data = config["data"]
    frame = generate_synthetic_telemetry(TelemetryConfig(
        n_hosts=int(data["n_hosts"]),
        n_steps=int(data["n_steps"]),
        frequency_minutes=int(data["frequency_minutes"]),
        anomaly_fraction=float(data["anomaly_fraction"]),
        normal_noise=float(data["normal_noise"]),
        seed=int(config["seed"]),
    ))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"Saved {len(frame)} rows to {output}")
    print(f"Anomaly rate: {frame['label'].mean():.2%}")
    print(anomaly_counts(frame).to_string())


if __name__ == "__main__":
    main()
