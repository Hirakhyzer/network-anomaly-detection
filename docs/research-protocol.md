# Research Protocol

This protocol is a checklist for running trustworthy network anomaly detection experiments.

## 1. Define the research question

Examples:

- Do graph-temporal representations improve anomaly detection over host-independent models?
- Are transparent rolling statistics competitive against deep autoencoders under controlled anomalies?
- How sensitive are detection results to threshold choice?
- Which model family produces fewer false positives at similar recall?

## 2. Fix the experiment configuration

Before running the test evaluation, record:

- Dataset generator settings or dataset version.
- Random seeds.
- Train/validation/test time ranges.
- Model hyperparameters.
- Threshold-selection rule.
- Hardware and software environment.

## 3. Preserve chronological separation

Use chronological splits rather than random splits unless the research question explicitly requires a different design.

Recommended split logic:

```text
Training period      → fit preprocessing and train detector
Validation period    → select threshold and tune only declared parameters
Test period          → final held-out evaluation only
```

## 4. Fit preprocessing safely

Preprocessing should be fitted on normal training traffic only. This is especially important for scalers, rolling statistics, feature normalization, and dimensionality reduction.

## 5. Train or fit detector

Each detector should receive the same split design and comparable input features unless the experiment is explicitly studying feature-set differences.

## 6. Select threshold

Threshold selection should be defined before final test evaluation. Examples:

- Use a percentile of normal validation anomaly scores.
- Use a validation operating point targeting a fixed false-positive rate.
- Use a domain-defined alert budget.

Do not choose a threshold by inspecting test labels.

## 7. Evaluate on test data

Recommended metrics:

- Precision.
- Recall.
- F1-score.
- ROC-AUC.
- PR-AUC.
- False-positive rate.
- Detection delay.
- Missed anomaly runs.

## 8. Repeat and compare

For research-quality comparison, run repeated seeds and summarize central tendency plus variability. Report model ranking stability rather than only one lucky run.

## 9. Report limitations

Every report should state:

- Whether the data is synthetic or real.
- Whether anomaly labels are simulated or externally validated.
- Whether the model was tested on external datasets.
- Whether results generalize beyond the configured generator.
- Any practical deployment constraints.

## 10. Avoid overclaiming

Synthetic-data performance is useful for controlled research and debugging, but it is not a substitute for real-world validation. Publication-ready claims should include external validation where licensing and ethics allow.
