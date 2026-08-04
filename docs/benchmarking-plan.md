# Benchmarking Plan

This document defines how to compare anomaly detection models in this repository.

## Goals

The benchmark should answer three questions:

1. Which model detects labelled anomaly periods most reliably?
2. Which model produces the lowest false-positive burden?
3. Which model remains stable across seeds and telemetry-generator settings?

## Models to compare

| Family | Example model |
|---|---|
| Statistical baseline | Z-score |
| Robust statistical baseline | Rolling median/MAD |
| Classical ML | Isolation Forest |
| Deep sequence model | LSTM autoencoder |
| Attention-based sequence model | Transformer autoencoder |
| Multi-host dependency model | Graph-temporal autoencoder |

## Core metrics

| Metric | Interpretation |
|---|---|
| Precision | Alert usefulness. |
| Recall | Anomaly coverage. |
| F1-score | Precision-recall balance. |
| ROC-AUC | Ranking quality across thresholds. |
| PR-AUC | Ranking quality under class imbalance. |
| False-positive rate | Operational alert burden. |
| Detection delay | Speed of detection after anomaly onset. |
| Missed runs | Complete failures to detect anomaly episodes. |

## Repeated-seed design

Use multiple seeds where possible:

```bash
python scripts/run_benchmark.py --seeds 42 43 44 45 46
```

For each model, report:

- Mean metric value.
- Standard deviation.
- Best and worst seed.
- Rank stability.
- Failure cases.

## Ablation ideas

Useful ablations include:

- Removing host identity features.
- Removing graph features.
- Changing anomaly severity.
- Varying window length.
- Changing validation threshold percentile.
- Comparing normal-only preprocessing with unsafe preprocessing to demonstrate leakage effects.

## Recommended tables

### Main benchmark table

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | Delay |
|---|---:|---:|---:|---:|---:|---:|---:|
| Z-score | | | | | | | |
| Rolling MAD | | | | | | | |
| Isolation Forest | | | | | | | |
| LSTM AE | | | | | | | |
| Transformer AE | | | | | | | |
| Graph-temporal AE | | | | | | | |

### Repeated-seed stability table

| Model | Mean F1 | Std F1 | Mean PR-AUC | Std PR-AUC | Rank stability |
|---|---:|---:|---:|---:|---|
| Z-score | | | | | |
| Rolling MAD | | | | | |
| Isolation Forest | | | | | |
| LSTM AE | | | | | |
| Transformer AE | | | | | |
| Graph-temporal AE | | | | | |

## Reporting rule

Do not report only the best model. Always include transparent baselines and explain when a complex model is not clearly better.
