from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nadlab.experiment import SUPPORTED_MODELS, load_config, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a reproducible network anomaly detection experiment.")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML experiment configuration.")
    parser.add_argument("--model", default="all", choices=["all", *SUPPORTED_MODELS], help="Detector or full benchmark.")
    parser.add_argument("--output-dir", default=None, help="Override output directory from configuration.")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.output_dir:
        config["output_dir"] = args.output_dir
    summary = run_experiment(config, args.model)
    print("\nExperiment complete. Metrics ranked by F1:\n")
    print(summary.to_string(index=False))
    print(f"\nSaved results to: {config['output_dir']}")


if __name__ == "__main__":
    main()
