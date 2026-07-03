from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Create paper-ready comparison figures from result CSV files.")
    parser.add_argument("--results", default="results", help="Directory containing metrics_summary.csv and roc_*.csv files.")
    args = parser.parse_args()
    results = Path(args.results)
    figures = results / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(results / "metrics_summary.csv").sort_values("f1", ascending=False)

    figure, axis = plt.subplots(figsize=(9, 4.8))
    axis.bar(metrics["model"], metrics["f1"])
    axis.set_title("F1 score by detector")
    axis.set_xlabel("Detector")
    axis.set_ylabel("F1 score")
    axis.set_ylim(0, 1)
    axis.tick_params(axis="x", rotation=25)
    figure.tight_layout()
    figure.savefig(figures / "f1_comparison.png", dpi=220)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.8, 5.5))
    for path in sorted(results.glob("roc_*.csv")):
        curve = pd.read_csv(path)
        model = curve["model"].iloc[0]
        axis.plot(curve["fpr"], curve["tpr"], label=model)
    axis.plot([0, 1], [0, 1], linestyle="--", label="chance")
    axis.set_title("ROC comparison")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(figures / "roc_comparison.png", dpi=220)
    plt.close(figure)

    print(f"Saved figures to {figures}")


if __name__ == "__main__":
    main()
