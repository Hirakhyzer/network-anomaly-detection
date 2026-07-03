# Research protocol

## Research question

How do statistical baselines, Isolation Forest, sequence autoencoders, and graph-temporal reconstruction models compare when detecting controlled anomalies in multi-host network telemetry?

## Hypotheses

1. Sequence and graph-temporal reconstruction models will achieve stronger ranking quality than univariate statistical thresholds when attack signals are distributed across several features.
2. The graph-temporal model will be most useful when anomaly evidence depends on relationships among hosts.
3. Statistical baselines will remain important because they are fast, transparent, and may have lower detection delay for strong single-feature shifts.

## Experimental safeguards

- **Synthetic-data boundary:** All default data are generated. Results describe this generator and configuration, not real enterprise traffic.
- **Chronological splits:** Training, validation, and testing use non-overlapping time periods.
- **Normal-only fitting:** Feature scaling, anomaly detector fitting, and neural reconstruction training use normal training observations only.
- **Validation-only thresholding:** Alert thresholds are selected from normal validation scores. The test period is not used to choose a threshold.
- **Same evaluation unit:** Statistical and neural models are scored on comparable temporal windows.
- **Repeated seeds:** `scripts/run_benchmark.py` produces fold-level metrics for uncertainty and nonparametric comparisons.
- **No test-set model selection:** Pick the proposed architecture and hyperparameters before inspecting final test metrics. Report all tested models, including weaker results.

## Main outcomes

- Precision, recall, F1
- ROC-AUC and PR-AUC
- False-positive rate
- Detection delay in minutes
- Missed anomalous runs

## Ablations to run

1. Window sizes: 12, 24, and 48 observations.
2. Anomaly fraction: 0.03, 0.08, and 0.15.
3. Severity ranges: mild, medium, and high.
4. Host counts: 4, 8, and 16.
5. LSTM/Transformer hidden size and latent size.
6. Graph adjacency: learned correlation graph versus identity graph.

## Threats to validity

Synthetic telemetry can never represent all real network behavior. Attack shapes, class imbalance, host relationships, and temporal dynamics should be varied deliberately. A future external-validation stage should reproduce the pipeline with a suitable public dataset, document licensing and preprocessing, and retain the same no-leakage split strategy.
