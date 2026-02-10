"""Pretraining script for time series forecasting using temporal priors."""

import argparse

import torch
from pfns.bar_distribution import FullSupportBarDistribution
from sklearn.metrics import r2_score

from tfmplayground.callbacks import ConsoleLoggerCallback
from tfmplayground.model import NanoTabPFNModel
from tfmplayground.priors import TimeSeriesPriorDataLoader, DEFAULT_TS_FIXED_HP, TS_PRIOR_PRESETS
from tfmplayground.train import train
from tfmplayground.utils import get_default_device, set_randomness_seed

parser = argparse.ArgumentParser(description="Pretrain nanoTabPFN for time series forecasting")

# Model architecture
parser.add_argument("--heads", type=int, default=6, help="number of attention heads")
parser.add_argument("--embeddingsize", type=int, default=192, help="embedding size")
parser.add_argument("--hiddensize", type=int, default=768, help="MLP hidden layer size")
parser.add_argument("--layers", type=int, default=6, help="number of transformer layers")

# Training settings
parser.add_argument("--batchsize", type=int, default=8, help="batch size")
parser.add_argument("--accumulate", type=int, default=1, help="gradient accumulation steps")
parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
parser.add_argument("--steps", type=int, default=100, help="steps per epoch")
parser.add_argument("--epochs", type=int, default=1000, help="number of epochs")
parser.add_argument("--loadcheckpoint", type=str, default=None, help="checkpoint to resume from")
parser.add_argument("--runname", type=str, default="nanoTabPFN-forecasting", help="run name")

# Prior settings
parser.add_argument("--priortype", type=str, default="mixed",
                    choices=["trend", "seasonal", "ar", "random_walk", "mixed"],
                    help="type of temporal pattern to generate")
parser.add_argument("--preset", type=str, default=None,
                    choices=list(TS_PRIOR_PRESETS.keys()),
                    help="use a predefined prior preset (overrides --priortype)")
parser.add_argument("--minseqlen", type=int, default=50, help="minimum sequence length")
parser.add_argument("--maxseqlen", type=int, default=512, help="maximum sequence length")
parser.add_argument("--minfeatures", type=int, default=1, help="minimum number of features")
parser.add_argument("--maxfeatures", type=int, default=20, help="maximum number of features")

# Output settings
parser.add_argument("--n_buckets", type=int, default=100, help="number of buckets for regression")
parser.add_argument("--saveweights", type=str, default="nanotabpfn_forecasting_weights.pth",
                    help="path to save trained weights")
parser.add_argument("--savebuckets", type=str, default="nanotabpfn_forecasting_buckets.pth",
                    help="path to save bucket edges")

args = parser.parse_args()

set_randomness_seed(2402)

device = get_default_device()

# Load checkpoint if provided
ckpt = None
if args.loadcheckpoint:
    ckpt = torch.load(args.loadcheckpoint)

# Determine prior configuration
if args.preset:
    preset_config = TS_PRIOR_PRESETS[args.preset]
    prior_type = preset_config.get("prior_type", "mixed")
    fixed_hp = {**DEFAULT_TS_FIXED_HP, **preset_config.get("fixed_hp", {})}
    print(f"Using preset '{args.preset}' with prior_type='{prior_type}'")
else:
    prior_type = args.priortype
    fixed_hp = DEFAULT_TS_FIXED_HP
    print(f"Using prior_type='{prior_type}' with default hyperparameters")

# Create prior data loader
prior = TimeSeriesPriorDataLoader(
    num_steps=args.steps,
    batch_size=args.batchsize,
    num_datapoints_min=args.minseqlen,
    num_datapoints_max=args.maxseqlen,
    min_features=args.minfeatures,
    max_features=args.maxfeatures,
    device=device,
    prior_type=prior_type,
    fixed_hp=fixed_hp,
)

# Create model
model = NanoTabPFNModel(
    num_attention_heads=args.heads,
    embedding_size=args.embeddingsize,
    mlp_hidden_size=args.hiddensize,
    num_layers=args.layers,
    num_outputs=args.n_buckets,
)

if ckpt:
    model.load_state_dict(ckpt['model'])

# Create bucket edges for regression output
# For time series, we use a fixed range since we normalize targets
bucket_edges = torch.linspace(-4.0, 4.0, args.n_buckets + 1).to(device)
torch.save(bucket_edges, args.savebuckets)

dist = FullSupportBarDistribution(bucket_edges)


class ForecastingLoggerCallback(ConsoleLoggerCallback):
    """Callback for logging forecasting training progress."""
    
    def on_epoch_end(self, epoch: int, epoch_time: float, loss: float, model, **kwargs):
        print(
            f"epoch {epoch:5d} | time {epoch_time:5.2f}s | mean loss {loss:5.4f}",
            flush=True
        )


callbacks = [ForecastingLoggerCallback()]

print(f"\n{'='*60}")
print(f"Time Series Forecasting Pretraining")
print(f"{'='*60}")
print(f"Prior type: {prior_type}")
print(f"Sequence length: {args.minseqlen} - {args.maxseqlen}")
print(f"Features: {args.minfeatures} - {args.maxfeatures}")
print(f"Batch size: {args.batchsize}")
print(f"Steps per epoch: {args.steps}")
print(f"Epochs: {args.epochs}")
print(f"{'='*60}\n")

trained_model, loss = train(
    model=model,
    prior=prior,
    criterion=dist,
    epochs=args.epochs,
    accumulate_gradients=args.accumulate,
    lr=args.lr,
    device=device,
    callbacks=callbacks,
    ckpt=ckpt,
    run_name=args.runname,
)

torch.save(trained_model.to('cpu').state_dict(), args.saveweights)
print(f"\nModel saved to {args.saveweights}")
print(f"Bucket edges saved to {args.savebuckets}")
