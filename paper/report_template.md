# Trustworthy Network Anomaly Detection: Reproducible Comparative Study

**Author:** Hira Khyzer  
**Affiliation:** [Department / University]  
**Date:** [Month Year]

## Abstract

State the operational problem, dataset boundary, evaluated methods, main evaluation protocol, headline result, and limitation in 150–250 words. Do not claim real-world generalization from the default synthetic data.

## 1. Introduction

- Why early detection of abnormal network behaviour matters.
- Why transparent baselines should be compared with deep learning.
- Research question and contributions.

## 2. Related work

Organize sources around statistical anomaly detection, Isolation Forest, recurrent autoencoders, Transformer-based time-series anomaly detection, and graph-based network monitoring.

## 3. Methodology

### 3.1 Telemetry generator and anomaly taxonomy

Describe hosts, features, sampling interval, DDoS, port-scan, brute-force, and exfiltration perturbations.

### 3.2 Leakage-aware evaluation protocol

Describe chronological splits, normal-only scaling/training, validation threshold selection, and the held-out test period.

### 3.3 Models

- Z-score baseline
- Rolling median/MAD baseline
- Isolation Forest
- LSTM autoencoder
- Transformer autoencoder
- Graph-temporal autoencoder

## 4. Experimental setup

Report seeds, hyperparameters, hardware, Python/PyTorch versions, MATLAB version, and all ablations.

## 5. Results

Insert `metrics_summary.csv`, ROC figures, PR curves, false-positive rate, and detection delay. For repeated seeds, report mean ± standard deviation and the Friedman/post-hoc comparison.

## 6. Discussion

Explain where baselines win, where representation learning helps, likely reasons, error patterns, operational trade-offs, and reproducibility limitations.

## 7. Ethics, security, and limitations

Clarify synthetic-data use, responsible evaluation, false-positive consequences, and why the system is not an autonomous incident-response tool.

## 8. Conclusion

Summarize the supported findings and concrete next steps for public-data or real-telemetry validation.

## Reproducibility appendix

Include the exact commit hash, config file, command lines, exported run manifest, and environment details.
