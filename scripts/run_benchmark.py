from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nadlab.experiment import load_config, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run repeated-seed benchmark for statistical comparison.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--models", nargs="+", default=["zscore", "rolling_mad", "isolation_forest", "lstm_ae", "transformer_ae", "graph_temporal_ae"])
    parser.add_argument("--output-dir", default="results/benchmark")
    args = parser.parse_args()
    base = load_config(args.config)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    all_runs = []
    for seed in args.seeds:
        for model in args.models:
            config = deepcopy(base)
            config["seed"] = seed
            config["output_dir"] = str(output / f"seed_{seed}" / model)
            summary = run_experiment(config, model)
            summary["seed"] = seed
            all_runs.append(summary)
            print(f"Finished seed={seed}, model={model}")
    folds = pd.concat(all_runs, ignore_index=True)
    folds.to_csv(output / "fold_metrics.csv", index=False)
    aggregate = folds.groupby("model", as_index=False).agg(
        f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        roc_auc_mean=("roc_auc", "mean"), roc_auc_std=("roc_auc", "std"),
        pr_auc_mean=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std"),
        delay_mean=("detection_delay_minutes", "mean"),
    )
    aggregate.to_csv(output / "benchmark_summary.csv", index=False)
    print(f"Saved repeated-seed results to {output}")


if __name__ == "__main__":
    main()
