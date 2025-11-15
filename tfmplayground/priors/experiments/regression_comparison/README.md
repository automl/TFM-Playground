# Regression Comparison Experiments

Compare different tabular priors on regression tasks.

## Structure

```
regression_comparison/
├── config.yaml              # Experiment configuration
├── utils.py                 # Shared utilities
├── 1_data_generation.py     # Generate synthetic data
├── data_analysis.py         # Analysis functions 
├── 2_run_analysis.py        # CLI for running analysis
└── results/
    ├── data/               # Generated HDF5 files
    └── reports/            # Analysis reports
```

## Usage

1. Edit `config.yaml` to configure experiments
2. Run `python 1_data_generation.py` to generate data
3. Run `python 2_run_analysis.py` to analyze and compare
4. Check `results/reports/` for comparison reports

Or use `./run.sh` to run both steps interactively.

## Output

- Data files: `results/data/prior_<name>_<params>.h5`
- Reports: `results/reports/<prior>_analysis_report.txt`
- Comparisons: `results/reports/comparison_<prior1>_vs_<prior2>.txt`

## Analysis Metrics

- Basic statistics (samples, sequences, features)
- Target distribution (mean, variance, skewness, kurtosis)
- Feature distributions and outlier detection
- Feature-target relationships (correlations, nonlinearity)
- Mutual information (nonlinear dependencies)
- Target scale and deviation
- Feature redundancy (collinearity)
- Noise characteristics (linear R²)
