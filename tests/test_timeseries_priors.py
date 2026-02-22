"""Unit tests for time series priors.

Run with: pytest tests/test_timeseries_priors.py -v
"""

import pytest
import torch


# =============================================================================
# Test 1: Imports
# =============================================================================

def test_imports():
    """Verify all time series prior modules can be imported."""
    from tfmplayground.priors.timeseries import (
        TemporalXSampler,
        TimeSeriesSCM,
        ForecastPriorDataset,
        DEFAULT_TS_FIXED_HP,
        TS_PRIOR_PRESETS,
    )
    from tfmplayground.priors import TimeSeriesPriorDataLoader
    
    assert TemporalXSampler is not None
    assert TimeSeriesSCM is not None
    assert ForecastPriorDataset is not None
    assert DEFAULT_TS_FIXED_HP is not None
    assert TS_PRIOR_PRESETS is not None
    assert TimeSeriesPriorDataLoader is not None


# =============================================================================
# Test 2: TemporalXSampler Shape
# =============================================================================

def test_temporal_sampler_shape():
    """Verify TemporalXSampler produces correct output shape."""
    from tfmplayground.priors.timeseries import TemporalXSampler
    
    seq_len = 100
    num_features = 5
    
    sampler = TemporalXSampler(
        seq_len=seq_len,
        num_features=num_features,
        device="cpu",
    )
    
    X = sampler.sample()
    
    assert X.shape == (seq_len, num_features), f"Expected ({seq_len}, {num_features}), got {X.shape}"
    assert X.dtype == torch.float32, f"Expected float32, got {X.dtype}"


# =============================================================================
# Test 3: TimeSeriesSCM Shape
# =============================================================================

def test_timeseries_scm_shape():
    """Verify TimeSeriesSCM produces correct output shapes."""
    from tfmplayground.priors.timeseries import TemporalXSampler, TimeSeriesSCM
    
    seq_len = 100
    num_features = 5
    max_lags = 3
    
    # Generate input features
    sampler = TemporalXSampler(seq_len=seq_len, num_features=num_features, device="cpu")
    X_input = sampler.sample()
    
    # Generate targets
    scm = TimeSeriesSCM(
        seq_len=seq_len,
        num_features=num_features,
        device="cpu",
        max_lags=max_lags,
    )
    
    X_output, y = scm.generate(X_input)
    
    # X should have lag features added
    expected_features = num_features + (num_features * max_lags)
    assert X_output.shape[0] == seq_len, f"Expected seq_len {seq_len}, got {X_output.shape[0]}"
    assert X_output.shape[1] == expected_features, f"Expected {expected_features} features, got {X_output.shape[1]}"
    
    # y should be 1D with seq_len
    assert y.shape == (seq_len,), f"Expected y shape ({seq_len},), got {y.shape}"


# =============================================================================
# Test 4: ForecastPriorDataset Batch Format
# =============================================================================

def test_forecast_dataset_batch_format():
    """Verify ForecastPriorDataset produces correct batch format."""
    from tfmplayground.priors.timeseries import ForecastPriorDataset, DEFAULT_TS_FIXED_HP
    
    batch_size = 4
    min_seq_len = 50
    max_seq_len = 64
    min_features = 2
    max_features = 5
    
    dataset = ForecastPriorDataset(
        batch_size=batch_size,
        min_seq_len=min_seq_len,
        max_seq_len=max_seq_len,
        min_features=min_features,
        max_features=max_features,
        device="cpu",
        prior_type="mixed",
        fixed_hp=DEFAULT_TS_FIXED_HP,
    )
    
    batch = dataset.get_batch()
    
    # Check required keys
    required_keys = {"x", "y", "target_y", "single_eval_pos"}
    assert set(batch.keys()) == required_keys, f"Expected keys {required_keys}, got {set(batch.keys())}"
    
    # Check shapes
    x = batch["x"]
    y = batch["y"]
    target_y = batch["target_y"]
    single_eval_pos = batch["single_eval_pos"]
    
    assert x.dim() == 3, f"x should be 3D (batch, seq, feat), got {x.dim()}D"
    assert y.dim() == 2, f"y should be 2D (batch, seq), got {y.dim()}D"
    assert target_y.dim() == 2, f"target_y should be 2D (batch, seq), got {target_y.dim()}D"
    
    assert x.shape[0] == batch_size, f"Expected batch_size {batch_size}, got {x.shape[0]}"
    assert y.shape[0] == batch_size
    assert target_y.shape[0] == batch_size
    
    # Check single_eval_pos is valid
    seq_len = x.shape[1]
    assert isinstance(single_eval_pos, int), f"single_eval_pos should be int, got {type(single_eval_pos)}"
    assert 0 < single_eval_pos < seq_len, f"single_eval_pos {single_eval_pos} not in valid range (0, {seq_len})"


# =============================================================================
# Test 5: DataLoader Iteration
# =============================================================================

def test_dataloader_iteration():
    """Verify TimeSeriesPriorDataLoader can be iterated."""
    from tfmplayground.priors import TimeSeriesPriorDataLoader
    
    num_steps = 3
    batch_size = 2
    
    loader = TimeSeriesPriorDataLoader(
        num_steps=num_steps,
        batch_size=batch_size,
        num_datapoints_min=50,
        num_datapoints_max=64,
        min_features=2,
        max_features=5,
        device="cpu",
        prior_type="mixed",
    )
    
    # Check length
    assert len(loader) == num_steps, f"Expected len {num_steps}, got {len(loader)}"
    
    # Iterate through all batches
    batch_count = 0
    for batch in loader:
        assert "x" in batch
        assert "y" in batch
        assert "target_y" in batch
        assert "single_eval_pos" in batch
        batch_count += 1
    
    assert batch_count == num_steps, f"Expected {num_steps} batches, got {batch_count}"


# =============================================================================
# Test 6: No NaNs in Output
# =============================================================================

def test_no_nans_in_output():
    """Verify generated data contains no NaN or Inf values."""
    from tfmplayground.priors import TimeSeriesPriorDataLoader
    
    loader = TimeSeriesPriorDataLoader(
        num_steps=5,
        batch_size=4,
        num_datapoints_min=50,
        num_datapoints_max=64,
        min_features=2,
        max_features=5,
        device="cpu",
        prior_type="mixed",
    )
    
    for batch in loader:
        x = batch["x"]
        y = batch["y"]
        target_y = batch["target_y"]
        
        assert not torch.isnan(x).any(), "Found NaN in x"
        assert not torch.isnan(y).any(), "Found NaN in y"
        assert not torch.isnan(target_y).any(), "Found NaN in target_y"
        
        assert not torch.isinf(x).any(), "Found Inf in x"
        assert not torch.isinf(y).any(), "Found Inf in y"
        assert not torch.isinf(target_y).any(), "Found Inf in target_y"


# =============================================================================
# Test 7: All Prior Types Work
# =============================================================================

@pytest.mark.parametrize("prior_type", ["trend", "seasonal", "ar", "random_walk", "mixed"])
def test_all_prior_types(prior_type):
    """Verify all prior types can generate data without error."""
    from tfmplayground.priors import TimeSeriesPriorDataLoader
    
    loader = TimeSeriesPriorDataLoader(
        num_steps=2,
        batch_size=2,
        num_datapoints_min=50,
        num_datapoints_max=64,
        min_features=2,
        max_features=5,
        device="cpu",
        prior_type=prior_type,
    )
    
    # Should complete without error
    for batch in loader:
        assert batch["x"].shape[0] == 2
        assert not torch.isnan(batch["x"]).any()


# =============================================================================
# Run tests directly
# =============================================================================

if __name__ == "__main__":
    print("Running time series prior tests...\n")
    
    tests = [
        ("Imports", test_imports),
        ("TemporalXSampler Shape", test_temporal_sampler_shape),
        ("TimeSeriesSCM Shape", test_timeseries_scm_shape),
        ("ForecastPriorDataset Batch Format", test_forecast_dataset_batch_format),
        ("DataLoader Iteration", test_dataloader_iteration),
        ("No NaNs in Output", test_no_nans_in_output),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            print(f"✓ {name}")
            passed += 1
        except Exception as e:
            print(f"✗ {name}: {e}")
            failed += 1
    
    # Parametrized test
    print("\nTesting all prior types:")
    for prior_type in ["trend", "seasonal", "ar", "random_walk", "mixed"]:
        try:
            test_all_prior_types(prior_type)
            print(f"  ✓ {prior_type}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {prior_type}: {e}")
            failed += 1
    
    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
