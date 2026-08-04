# Contributing

Thank you for your interest in improving **Trustworthy Network Anomaly Detection Lab**.

This project welcomes contributions that improve reproducibility, research clarity, model quality, documentation, and test coverage.

## Good contribution areas

- Add new anomaly detection baselines.
- Improve graph-temporal modeling modules.
- Add dataset adapters for permitted public datasets.
- Improve threshold-selection strategies.
- Add reproducibility checks.
- Add tests for metrics and leakage-sensitive preprocessing.
- Improve documentation, diagrams, and research reporting templates.

## Contribution workflow

1. Open an issue describing the change.
2. Create a focused branch.
3. Add tests when the change affects behavior.
4. Run the test suite locally.
5. Open a pull request with a clear summary.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
pytest
```

## Research standards

When adding experiments, document:

- Dataset version or generator configuration.
- Random seeds.
- Train/validation/test split.
- Threshold-selection rule.
- Metrics reported.
- Limitations.

## Responsible security work

Do not contribute code that is designed to evade detection, hide malicious traffic, or support unauthorized activity. This project is for defensive research, education, and reproducible experimentation.
