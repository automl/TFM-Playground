"""Evaluation script for forecasting models.

Compares a trained model against an untrained baseline on synthetic time series data.
"""

import argparse
import torch
import numpy as np
from pfns.bar_distribution import FullSupportBarDistribution

from tfmplayground.model import NanoTabPFNModel
from tfmplayground.priors.timeseries import ForecastPriorDataset, DEFAULT_TS_FIXED_HP
from tfmplayground.utils import get_default_device, set_randomness_seed


def load_model(weights_path: str, device: torch.device) -> NanoTabPFNModel:
    """Load a trained model from weights file."""
    checkpoint = torch.load(weights_path, map_location=device)
    
    # Get architecture from checkpoint or use defaults
    if 'architecture' in checkpoint:
        arch = checkpoint['architecture']
        model = NanoTabPFNModel(
            num_attention_heads=arch['num_attention_heads'],
            embedding_size=arch['embedding_size'],
            mlp_hidden_size=arch['mlp_hidden_size'],
            num_layers=arch['num_layers'],
            num_outputs=arch['num_outputs'],
        )
        model.load_state_dict(checkpoint['model'])
    else:
        # Assume it's just state dict with default architecture
        model = NanoTabPFNModel(
            num_attention_heads=6,
            embedding_size=192,
            mlp_hidden_size=768,
            num_layers=6,
            num_outputs=100,
        )
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model


def create_untrained_model(reference_model: NanoTabPFNModel, device: torch.device) -> NanoTabPFNModel:
    """Create an untrained model with same architecture."""
    model = NanoTabPFNModel(
        num_attention_heads=reference_model.num_attention_heads,
        embedding_size=reference_model.embedding_size,
        mlp_hidden_size=reference_model.mlp_hidden_size,
        num_layers=reference_model.num_layers,
        num_outputs=reference_model.num_outputs,
    )
    model.to(device)
    model.eval()
    return model


def generate_test_data(n_samples: int, device: torch.device, seed: int = 42):
    """Generate synthetic test data."""
    set_randomness_seed(seed)
    
    dataset = ForecastPriorDataset(
        batch_size=n_samples,
        min_seq_len=50,
        max_seq_len=64,  # Keep small for memory
        min_features=2,
        max_features=5,
        device=device,
        prior_type="mixed",
        fixed_hp=DEFAULT_TS_FIXED_HP,
    )
    
    batch = dataset.get_batch()
    return batch


def run_inference(model: NanoTabPFNModel, batch: dict, device: torch.device) -> torch.Tensor:
    """Run model inference on a batch."""
    x = batch['x'].to(device)
    y = batch['y'].to(device)
    single_eval_pos = batch['single_eval_pos']
    
    # Normalize y (same as training)
    y_train = y[:, :single_eval_pos]
    y_mean = y_train.mean(dim=1, keepdim=True)
    y_std = y_train.std(dim=1, keepdim=True) + 1e-8
    y_norm = (y_train - y_mean) / y_std
    
    with torch.no_grad():
        # Model expects (x, y) tuple
        data = (x, y_norm)
        output = model(data, single_eval_pos=single_eval_pos)
    
    # Output is logits over buckets, take mean prediction
    # For simplicity, use argmax bucket as prediction
    pred_buckets = output.argmax(dim=-1)  # (batch, seq_len - single_eval_pos)
    
    # Convert bucket indices to values (approximate)
    # Buckets span normalized range, roughly -4 to 4
    bucket_values = torch.linspace(-4, 4, model.num_outputs, device=device)
    pred_norm = bucket_values[pred_buckets]
    
    # Denormalize
    predictions = pred_norm * y_std + y_mean
    
    return predictions


def compute_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> dict:
    """Compute evaluation metrics."""
    pred = predictions.cpu().numpy().flatten()
    true = targets.cpu().numpy().flatten()
    
    # Remove NaN/Inf
    mask = np.isfinite(pred) & np.isfinite(true)
    pred = pred[mask]
    true = true[mask]
    
    if len(pred) == 0:
        return {'r2': float('nan'), 'mae': float('nan'), 'rmse': float('nan')}
    
    # R²
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # MAE
    mae = np.mean(np.abs(true - pred))
    
    # RMSE
    rmse = np.sqrt(np.mean((true - pred) ** 2))
    
    return {'r2': r2, 'mae': mae, 'rmse': rmse}


def main():
    parser = argparse.ArgumentParser(description="Evaluate forecasting model")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model weights")
    parser.add_argument("--n-samples", type=int, default=50, help="Number of test samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    device = get_default_device()
    print(f"Using device: {device}")
    
    # Load trained model
    print(f"\nLoading trained model from: {args.model}")
    trained_model = load_model(args.model, device)
    
    # Create untrained baseline
    print("Creating untrained baseline model...")
    baseline_model = create_untrained_model(trained_model, device)
    
    # Generate test data
    print(f"\nGenerating {args.n_samples} test samples...")
    test_batch = generate_test_data(args.n_samples, device, args.seed)
    
    single_eval_pos = test_batch['single_eval_pos']
    targets = test_batch['target_y'][:, single_eval_pos:]
    
    print(f"Sequence length: {test_batch['x'].shape[1]}")
    print(f"Train/test split at: {single_eval_pos}")
    print(f"Forecasting {targets.shape[1]} steps ahead")
    
    # Run inference
    print("\nRunning inference...")
    
    print("  - Trained model...")
    pred_trained = run_inference(trained_model, test_batch, device)
    
    print("  - Baseline model...")
    pred_baseline = run_inference(baseline_model, test_batch, device)
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics_trained = compute_metrics(pred_trained, targets)
    metrics_baseline = compute_metrics(pred_baseline, targets)
    
    # Print results
    print("\n" + "=" * 60)
    print("FORECASTING EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nTest samples: {args.n_samples}")
    print(f"Random seed: {args.seed}")
    print()
    print(f"{'Model':<20} {'R²':>10} {'MAE':>10} {'RMSE':>10}")
    print("-" * 52)
    print(f"{'TS-Prior (trained)':<20} {metrics_trained['r2']:>10.4f} {metrics_trained['mae']:>10.4f} {metrics_trained['rmse']:>10.4f}")
    print(f"{'Baseline (untrained)':<20} {metrics_baseline['r2']:>10.4f} {metrics_baseline['mae']:>10.4f} {metrics_baseline['rmse']:>10.4f}")
    print("-" * 52)
    
    r2_improvement = metrics_trained['r2'] - metrics_baseline['r2']
    print(f"\nR² Improvement: {r2_improvement:+.4f}")
    
    if metrics_trained['r2'] > metrics_baseline['r2']:
        print("Winner: TS-Prior (trained) ✓")
    else:
        print("Winner: Baseline (untrained)")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
