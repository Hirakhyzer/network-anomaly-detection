# Trustworthy Network Anomaly Detection Lab

A reproducible cybersecurity and machine-learning research project using **Python, PyTorch, Scikit-learn, MATLAB, and Jupyter**. It compares transparent statistical baselines with LSTM, Transformer, and graph-temporal autoencoders on labelled synthetic multi-host network telemetry.

> The default dataset is synthetic. Results describe the configured generator and must not be presented as real enterprise-network performance.

## Research workflow

```text
Generate labelled telemetry
→ Create chronological train / validation / test periods
→ Fit preprocessing on normal training traffic only
→ Train or fit detector
→ Select threshold from normal validation scores
→ Evaluate the held-out test period
→ Export metrics and curves
→ Analyze figures and repeated-seed results in MATLAB
```

## Models

| Category | Methods |
| --- | --- |
| Statistical baselines | Z-score, host-aware rolling median/MAD |
| Classical machine learning | Isolation Forest |
| Deep time-series models | LSTM autoencoder, Transformer autoencoder |
| Multi-host dependency model | Graph-temporal autoencoder using a correlation graph |

## Metrics

Precision, recall, F1, ROC-AUC, PR-AUC, false-positive rate, detection delay, missed anomaly runs, ROC curves, and precision–recall curves.

## Install

Python 3.10 or newer is required.

```bash
python -m venv .venv
```

Windows activation:

```bat
.venv\Scripts\activate
```

Install packages:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run experiments

Generate only the dataset:

```bash
python scripts/generate_data.py
```

Run a quick statistical baseline:

```bash
python scripts/run_experiment.py --model zscore
```

Run an individual deep model:

```bash
python scripts/run_experiment.py --model lstm_ae
python scripts/run_experiment.py --model transformer_ae
python scripts/run_experiment.py --model graph_temporal_ae
```

Run every model:

```bash
python scripts/run_experiment.py --model all
```

Create Python figures:

```bash
python scripts/plot_results.py --results results
```

Run a repeated-seed benchmark for statistics:

```bash
python scripts/run_benchmark.py --seeds 42 43 44
```

## MATLAB analysis

After Python results exist:

```matlab
addpath('matlab')
analyze_results('results')
plot_roc_comparison('results')
statistical_significance_test('results/benchmark')
```

## Tests

```bash
pytest
```

The tests validate telemetry generation, chronological splits, windowing, baselines, metrics, and output shapes for all PyTorch models.

## Project map

```text
configs/       Reproducible experiment settings
src/nadlab/    Data, preprocessing, baselines, models, training, metrics, pipeline
scripts/       Data generation, experiments, repeated benchmarks, Python figures
matlab/        MATLAB analysis, ROC figures, Friedman comparison
notebooks/     Jupyter research walkthrough
paper/         Paper/report structure
/docs/         Research protocol, safeguards, ablation plan
tests/         Unit tests
```

## Academic integrity note

Use the report template and research protocol to document parameters, seeds, hardware, and limitations. Do not tune on the test period. For publication-oriented work, add external validation with a licensed public dataset while preserving the same chronological, no-leakage evaluation design.
