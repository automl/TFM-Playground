"""Preentrenamiento. Cada experimento es un comando; los valores por defecto son la línea base.

Ejemplo, desde la raíz del repo:
    nohup python -u experiments/pretrain.py --gpu 6 --run-name mix_scm --prior-type mix_scm > workdir/mix_scm.log 2>&1 &
"""

import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"  # antes de importar torch

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from tfmplayground.callbacks import ConsoleLoggerCallback
from tfmplayground.external_priors import TabICLPriorDataLoader
from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.train import WideningConfig, train

parser = argparse.ArgumentParser()
parser.add_argument("--run-name", required=True)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--prior-type", default="mlp_scm")  # mlp_scm, mix_scm, tree_scm, graph_scm
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch-size", type=int, default=2)
parser.add_argument("--accumulate", type=int, default=4)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--add-features-max", type=int, default=5000)
parser.add_argument("--prob-no-widening", type=float, default=0.3)
parser.add_argument("--missing-rate-max", type=float, default=0.1)
args = parser.parse_args()

os.chdir(Path(__file__).resolve().parents[1])  # workdir/ siempre en la raíz del repo
device = torch.device(f"cuda:{args.gpu}")

# Guarda los parámetros del run junto a sus checkpoints
run_dir = Path("workdir") / args.run_name
run_dir.mkdir(parents=True, exist_ok=True)
(run_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

MAX_CLASSES = 10

prior = TabICLPriorDataLoader(
    num_steps=1000,  # batches por época
    batch_size=args.batch_size,
    num_datapoints_min=40,
    num_datapoints_max=300,
    min_features=50,
    max_features=350,
    max_num_classes=MAX_CLASSES,
    device=device,
    prior_type=args.prior_type,
    log_seq_len=True,
    min_train_size=0.3,
    max_train_size=0.9,
)

model = NanoTabPFNModel(
    embedding_size=192,
    num_attention_heads=6,
    mlp_hidden_size=768,
    num_layers=12,
    num_outputs=MAX_CLASSES,
).to(device)
model.gradient_checkpointing = True

widening = WideningConfig(
    add_features_min=200,
    add_features_max=args.add_features_max,
    sparsity_max=0.05,
    noise_max=1.0,
    include_original_prob=0.5,
    max_cats=20,
    prob_no_widening=args.prob_no_widening,
)

trained_model, loss = train(
    model=model,
    prior=prior,
    criterion=nn.CrossEntropyLoss(),
    epochs=args.epochs,
    accumulate_gradients=args.accumulate,
    lr=args.lr,
    device=device,
    callbacks=[ConsoleLoggerCallback()],
    run_name=args.run_name,
    missing_rate_max=args.missing_rate_max,
    widening=widening,
    amp_dtype=torch.bfloat16,
    warmup_steps=500,
    snapshot_every=5,
    log_every=50,
)
print(f"Done. Checkpoint: workdir/{args.run_name}/latest_checkpoint.pth")
