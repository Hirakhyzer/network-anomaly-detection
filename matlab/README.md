# MATLAB analysis workflow

Run the Python experiments first so CSV outputs exist in `results/`.

```matlab
addpath('matlab')
analyze_results('results')
plot_roc_comparison('results')
```

For a repeated-seed comparison, first run:

```bash
python scripts/run_benchmark.py --seeds 42 43 44
```

Then run:

```matlab
statistical_significance_test('results/benchmark')
```

The MATLAB scripts read exported CSV data rather than rerunning Python models. This keeps the empirical record reproducible and makes figure generation independent of the training environment.
