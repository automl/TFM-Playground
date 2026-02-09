# Design Doc: Time Series Priors for TFM-Playground

**Author:** [Your Name]  
**Created:** [Date]  
**Status:** Draft  
**Reviewers:** [Names]

---

## Table of Contents

1. [Overview](#1-overview)
2. [Background & Motivation](#2-background--motivation)
3. [Goals & Non-Goals](#3-goals--non-goals)
4. [Design](#4-design)
5. [Alternatives Considered](#5-alternatives-considered)
6. [Implementation Plan](#6-implementation-plan)
7. [Testing Strategy](#7-testing-strategy)
8. [Open Questions](#8-open-questions)
9. [References](#9-references)

---

## 1. Overview

### One-Line Summary

Add time-series-aware synthetic data priors to TFM-Playground to improve nanoTabPFN performance on forecasting downstream tasks.

### Context

TFM-Playground currently supports TabICL, TICL, and TabPFN priors for generating synthetic pre-training data. None of these priors model temporal dependencies, making the resulting models suboptimal for forecasting tasks.

### Proposal

Introduce a new `timeseries` prior module that generates synthetic datasets with realistic temporal patterns including trends, seasonality, and autoregressive structure.

---

## 2. Background & Motivation

### Current State

The existing priors in TFM-Playground generate tabular data where:

- **Rows are i.i.d.**: Each sample is independent of others
- **No temporal structure**: Data has no concept of "before" or "after"
- **Random train/test splits**: Split position has no temporal meaning

This works well for classification and standard regression tasks, but fails to expose the model to patterns critical for forecasting:

| Pattern | Present in TabICL? | Needed for Forecasting? |
|---------|-------------------|------------------------|
| Non-linear relationships | ✅ Yes | ✅ Yes |
| Feature interactions | ✅ Yes | ✅ Yes |
| Temporal autocorrelation | ❌ No | ✅ Yes |
| Trends | ❌ No | ✅ Yes |
| Seasonality | ❌ No | ✅ Yes |
| Lag dependencies | ❌ No | ✅ Yes |

### Why This Matters

Forecasting is a major use case for tabular foundation models. Without exposure to temporal patterns during pre-training, models must learn these patterns entirely from limited downstream data, reducing transfer learning effectiveness.

### Prior Art

- **TabPFN/TabICL**: Use MLP-SCM and Tree-SCM for synthetic data (no temporal)
- **TimeGPT**: Pre-trained on real time series (not synthetic priors)
- **Lag-Llama**: Uses real time series with autoregressive structure

Our approach combines the synthetic prior paradigm (TabPFN-style) with temporal structure (time-series-style).

---

## 3. Goals & Non-Goals

### Goals

| ID | Goal | Priority |
|----|------|----------|
| G1 | Generate synthetic time series with temporal correlation | P0 |
| G2 | Support trends, seasonality, and AR patterns | P0 |
| G3 | Maintain compatibility with existing training pipeline | P0 |
| G4 | Enforce temporal train/test splits | P0 |
| G5 | Configurable complexity via hyperparameter sampling | P1 |
| G6 | Support lag feature generation | P1 |
| G7 | Enable mixing with existing priors | P2 |

### Non-Goals

| ID | Non-Goal | Rationale |
|----|----------|-----------|
| NG1 | Modify model architecture | Project constraint |
| NG2 | Multi-step sequence output | Adds complexity; defer to v2 |
| NG3 | Probabilistic forecasting | Start with point forecasts |
| NG4 | Real data augmentation | Focus on synthetic priors |

---

## 4. Design

### 4.1 System Architecture

```
                    ┌─────────────────────────────────────┐
                    │         Configuration Layer         │
                    │                                     │
                    │  config.py: Hyperparameter specs    │
                    └──────────────┬──────────────────────┘
                                   │
                                   ▼
    ┌──────────────────────────────────────────────────────────┐
    │                   Generation Layer                        │
    │                                                           │
    │  ┌─────────────────┐      ┌─────────────────────────┐   │
    │  │TemporalXSampler │─────▶│    TimeSeriesSCM        │   │
    │  │                 │      │                         │   │
    │  │ • Trends        │      │ • Temporal causality    │   │
    │  │ • Seasonality   │      │ • MLP transformation    │   │
    │  │ • AR processes  │      │ • Lag feature creation  │   │
    │  │ • Random walks  │      │                         │   │
    │  └─────────────────┘      └────────────┬────────────┘   │
    │                                        │                 │
    └────────────────────────────────────────┼─────────────────┘
                                             │
                                             ▼
    ┌──────────────────────────────────────────────────────────┐
    │                    Dataset Layer                          │
    │                                                           │
    │  ┌─────────────────────────────────────────────────┐     │
    │  │           ForecastPriorDataset                  │     │
    │  │                                                 │     │
    │  │  • Batch generation                             │     │
    │  │  • Temporal train/test splitting                │     │
    │  │  • Group/subgroup structure                     │     │
    │  └─────────────────────┬───────────────────────────┘     │
    │                        │                                  │
    └────────────────────────┼──────────────────────────────────┘
                             │
                             ▼
    ┌──────────────────────────────────────────────────────────┐
    │                   Interface Layer                         │
    │                                                           │
    │  ┌─────────────────────────────────────────────────┐     │
    │  │         TimeSeriesPriorDataLoader               │     │
    │  │                                                 │     │
    │  │  Output: {x, y, target_y, single_eval_pos}      │     │
    │  │  (Compatible with existing training loop)       │     │
    │  └─────────────────────────────────────────────────┘     │
    │                                                           │
    └──────────────────────────────────────────────────────────┘
```

### 4.2 Component Specifications

#### 4.2.1 TemporalXSampler

**Purpose**: Generate feature matrix X with temporal correlations.

**Input**:
- `seq_len`: Number of time steps
- `num_features`: Number of feature columns
- `config`: Sampling configuration

**Output**:
- `X`: Tensor of shape `[seq_len, num_features]`

**Temporal Patterns Supported**:

| Pattern | Description | Parameters |
|---------|-------------|------------|
| Trend | Systematic drift over time | type (linear/poly), strength |
| Seasonality | Periodic oscillation | period, amplitude, phase |
| AR(p) | Autoregressive process | order, coefficients |
| Random Walk | Integrated noise | drift, volatility |
| Level Shift | Sudden mean change | probability, magnitude |

**Composition Strategy**:

Each feature is generated as:
```
x[t] = trend[t] + seasonal[t] + ar[t] + noise[t]
```
where each component is optionally included based on sampled configuration.


#### 4.2.2 TimeSeriesSCM

**Purpose**: Generate (X, y) pairs with temporal causal structure.

**Key Invariant**: `y[t]` depends only on `X[t']` where `t' < t`

**Modes**:

| Mode | Formula | Use Case |
|------|---------|----------|
| direct | `y[t] = f(X[t-k:t-1])` | Standard forecasting |
| autoregressive | `y[t] = f(X[t-k:t-1], y[t-k:t-1])` | AR-style models |
| transformed | `y = g(MLP(X))` | Non-linear relationships |

**MLP Configuration**:
- Layers: 1-3 (sampled)
- Hidden dim: 16-128 (sampled)
- Activation: tanh, relu, or identity
- Noise: Gaussian, added after each layer

**Lag Feature Generation** (optional):
```
Original: X = [x₁[t], x₂[t], ...]
With lags: X' = [x₁[t], x₁[t-1], ..., x₁[t-k], x₂[t], ...]
```

This exposes the model to pre-engineered temporal features.


#### 4.2.3 ForecastPriorDataset

**Purpose**: Manage batch generation with proper temporal semantics.

**Batch Structure**:
```
Batch (B datasets)
├── Group 1 (shares temporal pattern type)
│   ├── Subgroup 1a (shares specific parameters)
│   │   ├── Dataset 1 (unique noise realization)
│   │   └── Dataset 2
│   └── Subgroup 1b
│       └── ...
└── Group 2
    └── ...
```

**Temporal Split Logic**:
```
Time axis:  [1, 2, 3, ..., t, ..., T]
            |←─ train ──→|←─ test ─→|

single_eval_pos = t (the temporal cutoff)
```

Unlike random splits, this ensures:
- Training data is always "past"
- Test data is always "future"
- No temporal leakage

**Split Ratio Sampling**:
- `min_train_ratio`: 0.5 (need sufficient history)
- `max_train_ratio`: 0.9 (need sufficient test data)


#### 4.2.4 Configuration System

**Fixed Hyperparameters** (constant across all generated data):
```python
DEFAULT_TS_FIXED_HP = {
    "max_lags": 10,
    "normalize": True,
    "min_seq_len": 50,
    "forecast_horizon": 1,
}
```

**Sampled Hyperparameters** (varied per batch/group):
```python
DEFAULT_TS_SAMPLED_HP = {
    # Trend
    "trend_type": meta_choice(["none", "linear", "quadratic"]),
    "trend_strength": meta_uniform(0.0, 2.0),
    
    # Seasonality  
    "seasonal_period": meta_choice([None, 7, 12, 24, 52]),
    "seasonal_amplitude": meta_uniform(0.0, 3.0),
    
    # Autoregression
    "ar_order": meta_choice([0, 1, 2, 3]),
    "ar_persistence": meta_uniform(0.1, 0.95),
    
    # Noise
    "noise_std": meta_log_uniform(0.01, 0.5),
    
    # Structure
    "include_lags": meta_choice([True, False]),
    "num_layers": meta_choice([1, 2, 3]),
}
```


### 4.3 Interface Contract

**Output Format** (must match existing priors):
```python
{
    "x": Tensor[batch_size, seq_len, num_features],
    "y": Tensor[batch_size, seq_len],
    "target_y": Tensor[batch_size, seq_len],
    "single_eval_pos": int
}
```

This ensures compatibility with the existing training loop in `train.py`.


### 4.4 File Structure

```
tfmplayground/priors/
├── __init__.py                 # Add: export TimeSeriesPriorDataLoader
├── dataloader.py               # Add: TimeSeriesPriorDataLoader class
├── main.py                     # Add: --lib timeseries option
│
└── timeseries/                 # NEW DIRECTORY
    ├── __init__.py             # Export public classes
    ├── temporal_sampler.py     # TemporalXSampler
    ├── timeseries_scm.py       # TimeSeriesSCM  
    ├── forecast_dataset.py     # ForecastPriorDataset
    └── config.py               # TS_FIXED_HP, TS_SAMPLED_HP
```

---

## 5. Alternatives Considered

### Alternative A: Modify Existing XSampler

**Approach**: Add temporal options to TabICL's XSampler.

**Pros**:
- Less code to write
- Reuses existing infrastructure

**Cons**:
- Requires forking TabICL (dependency management)
- XSampler design assumes i.i.d.; retrofitting is awkward
- Harder to test independently

**Decision**: Rejected — cleaner to build new module.


### Alternative B: Use GP with Temporal Kernel

**Approach**: Configure existing GP prior with periodic/Matérn kernels.

**Pros**:
- No new code for generation
- GPs naturally model correlations

**Cons**:
- Limited pattern types (hard to get trends + seasonality + AR)
- Computationally expensive for long sequences
- Less control over specific patterns

**Decision**: Rejected — insufficient flexibility.


### Alternative C: Use Real Time Series Data

**Approach**: Pre-train on real forecasting datasets instead of synthetic.

**Pros**:
- Guaranteed realistic patterns
- No synthetic-to-real gap

**Cons**:
- Limited data quantity
- Privacy/licensing concerns
- Doesn't follow TFM synthetic prior paradigm

**Decision**: Rejected — out of scope; could complement but not replace.


### Alternative D: Sequence-to-Sequence Output

**Approach**: Output y as `[y[t+1], y[t+2], ..., y[t+h]]` instead of `y[t+1]`.

**Pros**:
- Direct multi-step forecasting
- More realistic task formulation

**Cons**:
- Requires changing output format
- May need architecture changes
- Adds complexity

**Decision**: Deferred to v2 — start simple.

---

## 6. Implementation Plan

### Phase 1: Foundation (Week 1)

| Day | Task | Deliverable |
|-----|------|-------------|
| 1 | Set up module structure | Empty `timeseries/` package |
| 2 | Implement basic samplers (AR, trend) | `temporal_sampler.py` v1 |
| 3 | Add seasonality, random walk | `temporal_sampler.py` v2 |
| 4 | Implement TimeSeriesSCM | `timeseries_scm.py` |
| 5 | Unit tests + visual verification | `tests/`, notebook |

### Phase 2: Integration (Week 2)

| Day | Task | Deliverable |
|-----|------|-------------|
| 1 | Implement ForecastPriorDataset | `forecast_dataset.py` |
| 2 | Create configuration system | `config.py` |
| 3 | Add TimeSeriesPriorDataLoader | `dataloader.py` update |
| 4 | Update CLI | `main.py` update |
| 5 | Integration tests | End-to-end verification |

### Phase 3: Evaluation (Week 3)

| Day | Task | Deliverable |
|-----|------|-------------|
| 1 | Create training script | `pretrain_forecasting.py` |
| 2 | Train model on TS priors | Model checkpoint |
| 3 | Create evaluation script | `eval_forecasting.py` |
| 4 | Run benchmarks | Results table |
| 5 | Documentation | README, this doc finalized |

---

## 7. Testing Strategy

### Unit Tests

| Component | Test | Verification |
|-----------|------|--------------|
| TemporalXSampler | AR output | Check autocorrelation > 0 at lag 1 |
| TemporalXSampler | Trend output | Check positive slope in linear regression |
| TemporalXSampler | Seasonal output | Check periodicity via FFT |
| TimeSeriesSCM | Output shape | Assert correct dimensions |
| TimeSeriesSCM | Temporal causality | Verify y[t] independent of X[t+1:] |
| ForecastPriorDataset | Split validity | Assert split respects time order |

### Integration Tests

| Test | Description |
|------|-------------|
| DataLoader iteration | Generate 100 batches without error |
| Training smoke test | Train 1 epoch, verify loss decreases |
| Output format | Verify matches expected schema |

### Visual Tests

| Test | What to Check |
|------|---------------|
| Time series plots | Visible trends, seasonality |
| ACF plots | Decay pattern for AR processes |
| Scatter plots | X vs y relationship |

### Benchmark Tests

| Comparison | Metric | Success Criterion |
|------------|--------|-------------------|
| TS prior vs TabICL prior | Forecasting R² | TS prior ≥ TabICL |
| TS prior vs naive baseline | Forecasting MAE | TS prior < naive |

---

## 8. Open Questions

| # | Question | Options | Status |
|---|----------|---------|--------|
| 1 | Should lag features be on by default? | Yes / No / Configurable | Leaning: Configurable, default Yes |
| 2 | What AR order range? | [1-3] / [1-5] / [1-10] | Leaning: [1-5] |
| 3 | Include level shifts/regime changes? | Yes / No | Leaning: No (v1), Yes (v2) |
| 4 | Mix with existing priors? | Always / Optional / Never | Leaning: Optional |
| 5 | Which eval datasets? | ETT / M4 / Synthetic | Leaning: Start synthetic |

---

## 9. References

1. TabPFN Paper: Hollmann et al., 2023
2. TabICL Paper: Qu et al., 2025  
3. TFM-Playground Repo: https://github.com/automl/TFM-Playground
4. TabICL Repo: https://github.com/soda-inria/tabicl
5. Time Series Components: Hyndman & Athanasopoulos, "Forecasting: Principles and Practice"

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| AR(p) | Autoregressive process of order p |
| SCM | Structural Causal Model |
| Prior | Generative mechanism for synthetic data |
| single_eval_pos | Index separating train from test data |
| Lag features | Past values of a variable used as features |

---

## Appendix B: Example Generated Data

*[To be added: visualizations of generated time series showing trends, seasonality, and AR patterns]*

---

## Change Log

| Date | Author | Change |
|------|--------|--------|
| TBD | TBD | Initial draft |
