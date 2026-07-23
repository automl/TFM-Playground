"""
Experiment: does adding topo-encoded column-name text embeddings help NanoTabPFN?

Uses the Do-PFN SCM prior (via TopoAwareDataLoader) to sample binary-target
datasets whose columns are named `topo_{rank}_{node}_f{feat_idx}`, so the causal
position of every feature is readable from its name.

Two classifiers are trained on the *identical* set of X sampled datasets:
    1. baseline   - a plain NanoTabPFN model (column names ignored)
    2. text       - a NanoTabPFN model with a ColumnEmbeddingProjector that adds
                    sentence-transformer embeddings of the topo column names to
                    every feature cell (ConTextTab-style).

Both share the same trunk initialisation (the text model's extra projector is the
only difference), so any accuracy gap is attributable to the text signal.

Finally both are evaluated on a *separate* set of 30 held-out SCM datasets.

Requirements:
    pip install sentence-transformers      # the `text` optional-dependency extra
    (first run downloads the ~90MB all-MiniLM-L6-v2 model)

Run from this directory (tfmplayground/priors/scm_prior/) so that
`from data_loader import ...` resolves, exactly like generate_data.py:
    cd tfmplayground/priors/scm_prior
    python topo_text_experiment.py                 # generate + save datasets, then run
    python topo_text_experiment.py --use-saved     # reuse saved datasets (skips the encoder)

Generated datasets are saved to --data-dir (default: ./topo_datasets/) as
train_datasets.pt / eval_datasets.pt so subsequent runs can skip regeneration.
"""
import argparse
import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import nn

from data_loader import make_dataloader  # TopoAwareDataLoader factory (this dir)

from tfmplayground.callbacks import ConsoleLoggerCallback
from tfmplayground.model import NanoTabPFNModel
from tfmplayground.text_encoder import ColumnTextEncoder
from tfmplayground.train import train
from tfmplayground.utils import get_default_device, set_randomness_seed

# ─────────────────────────── config (smoke test) ──────────────────────────
# Every dataset is used exactly once. The pool is split into EPOCHS disjoint
# chunks of (X_TRAIN_DATASETS // EPOCHS) datasets; each epoch trains on a fresh
# chunk, so nothing repeats. EPOCHS only controls the granularity of the
# per-epoch progress log — total gradient steps == X_TRAIN_DATASETS either way.
X_TRAIN_DATASETS = 10000     # distinct SCM datasets, each visited once
EPOCHS           = 100      # disjoint chunks => X_TRAIN_DATASETS // EPOCHS per epoch
N_EVAL_DATASETS  = 30      # held-out datasets for the comparison

EMBEDDING_SIZE   = 64
NUM_LAYERS       = 3
NUM_HEADS        = 4
HIDDEN_SIZE      = 256
NUM_OUTPUTS      = 2       # binary classification
    
TRAIN_SEED       = 0       # dataloader seed for the training set
EVAL_SEED        = 12345   # different seed => disjoint held-out datasets
INIT_SEED        = 2402    # model-init / training rng

LR               = 1e-3
# ───────────────────────────────────────────────────────────────────────────


def cache_datasets(num_datasets: int, seed: int, encoder: ColumnTextEncoder) -> list[dict]:
    """Sample `num_datasets` SCM datasets once and pre-compute their column embeddings.

    batch_size=1 => each step is one independent SCM (== one dataset). The topo
    column names are encoded with the frozen sentence encoder and stored as
    `column_embeddings` of shape (1, F, D) so the train loop can consume them.
    """
    loader = make_dataloader(num_steps=num_datasets, batch_size=1, seed=seed)
    step = max(1, num_datasets // 10)                    # print ~10 progress updates
    batches = []
    for i, b in enumerate(loader, start=1):
        names = b["column_names"]                       # list[str], length F
        col_emb = encoder.encode(names)                 # (F, D) on encoder device
        if i % step == 0 or i == num_datasets:
            print(f"  cached {i}/{num_datasets} datasets", flush=True)
        batches.append({
            "x": b["x"].float(),                         # (1, N, F)
            "y": b["y"].float(),                         # (1, N, 1) binarized {0,1}
            "target_y": b["target_y"].float(),
            "single_eval_pos": int(b["single_eval_pos"]),
            "column_embeddings": col_emb.unsqueeze(0),   # (1, F, D)
            "column_names": names,
        })
    return batches


def get_datasets(num_datasets: int, seed: int, encoder, path: str, use_saved: bool,
                 tag: str) -> list[dict]:
    """Load pre-generated datasets from `path`, or sample fresh ones and save them.

    Each dataset is a dict of tensors + metadata (see cache_datasets); the whole
    list is (de)serialised with torch.save/torch.load. When reused, the sentence
    encoder is not needed at all (embeddings are already baked in).
    """
    if use_saved and os.path.exists(path):
        print(f"loading {tag} datasets from {path} ...")
        batches = torch.load(path, map_location="cpu", weights_only=False)
        print(f"  loaded {len(batches)} {tag} datasets")
        return batches

    print(f"sampling {num_datasets} {tag} datasets (seed={seed}) ...")
    batches = cache_datasets(num_datasets, seed, encoder)
    torch.save(batches, path)
    print(f"  saved {len(batches)} {tag} datasets -> {path}")
    return batches


class ChunkedLoader:
    """In-memory DataLoader that hands out one disjoint chunk of datasets per epoch.

    The train loop calls `iter(prior)` once per epoch; each call advances an
    internal pointer, so over `epochs` epochs every dataset is visited exactly
    once (never repeated). Exposes the attributes the train loop relies on
    (`num_steps`, `__len__`, `__iter__`). A fresh instance per model (built from
    the same batch list) guarantees both models train on byte-identical data.
    """

    def __init__(self, batches: list[dict], epochs: int):
        self.num_steps = len(batches) // epochs          # datasets per epoch
        usable = self.num_steps * epochs                 # drop any remainder
        self.batches = batches[:usable]
        self._pointer = 0

    def __iter__(self):
        start = self._pointer
        self._pointer += self.num_steps
        return iter(self.batches[start:start + self.num_steps])

    def __len__(self):
        return self.num_steps


def build_models(text_embedding_dim: int) -> tuple[NanoTabPFNModel, NanoTabPFNModel]:
    """Build baseline + text models sharing an identical trunk initialisation."""
    set_randomness_seed(INIT_SEED)
    common = dict(
        embedding_size=EMBEDDING_SIZE,
        num_attention_heads=NUM_HEADS,
        mlp_hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_outputs=NUM_OUTPUTS,
    )
    baseline = NanoTabPFNModel(**common)
    text = NanoTabPFNModel(**common, text_embedding_dim=text_embedding_dim)
    # Copy the shared trunk so the ONLY difference at init is the (new) projector.
    # strict=False leaves `column_projector.*` at its own initialisation.
    missing, unexpected = text.load_state_dict(baseline.state_dict(), strict=False)
    assert not unexpected, f"unexpected keys when copying trunk: {unexpected}"
    assert all(k.startswith("column_projector") for k in missing), missing
    return baseline, text


@torch.no_grad()
def evaluate(model: NanoTabPFNModel, batches: list[dict], device, use_text: bool) -> dict:
    """Direct-model evaluation (mirrors training exactly, no sklearn preprocessing).

    For each dataset we feed the full x with train-only y and read off logits for
    the test rows, then compute accuracy and (when both classes are present) ROC-AUC.
    """
    model.eval()
    accs, aucs = [], []
    for b in batches:
        x = b["x"].to(device)                       # (1, N, F)
        y = b["y"].to(device)                       # (1, N, 1)
        sep = b["single_eval_pos"]
        col_emb = b["column_embeddings"].to(device) if use_text else None

        logits = model((x, y[:, :sep]), single_eval_pos=sep,
                       column_embeddings=col_emb)[0]  # (n_test, 2)
        proba = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        preds = logits.argmax(dim=-1).cpu().numpy()
        true = y[0, sep:, 0].cpu().numpy().astype(int)

        accs.append(accuracy_score(true, preds))
        if len(np.unique(true)) > 1:                # AUC undefined for single-class splits
            aucs.append(roc_auc_score(true, proba))
    return {
        "accuracy": float(np.mean(accs)),
        "roc_auc": float(np.mean(aucs)) if aucs else float("nan"),
        "n_auc": len(aucs),
        "per_dataset_acc": accs,          # per-dataset accuracy, for head-to-head wins
    }


def parse_args():
    p = argparse.ArgumentParser(description="Topo column-name text-embedding experiment.")
    p.add_argument("--data-dir", default="topo_datasets",
                   help="directory to save/load the generated datasets (default: topo_datasets)")
    p.add_argument("--use-saved", action="store_true",
                   help="reuse datasets already saved in --data-dir instead of regenerating "
                        "(falls back to generating + saving if they are absent)")
    return p.parse_args()


def main():
    args = parse_args()
    device = get_default_device()
    print(f"device: {device}")

    os.makedirs(args.data_dir, exist_ok=True)
    train_path = os.path.join(args.data_dir, "train_datasets.pt")
    eval_path = os.path.join(args.data_dir, "eval_datasets.pt")

    # The sentence encoder is only needed when generating fresh datasets; if we
    # can reuse both saved files, skip loading it entirely.
    reuse = args.use_saved and os.path.exists(train_path) and os.path.exists(eval_path)
    encoder = None
    if not reuse:
        print("loading frozen sentence encoder (all-MiniLM-L6-v2) ...")
        encoder = ColumnTextEncoder(device="cpu")   # embeddings are tiny; cpu is fine

    train_batches = get_datasets(X_TRAIN_DATASETS, TRAIN_SEED, encoder, train_path,
                                 args.use_saved, "train")
    eval_batches = get_datasets(N_EVAL_DATASETS, EVAL_SEED, encoder, eval_path,
                                args.use_saved, "eval")

    dim = train_batches[0]["column_embeddings"].shape[-1]   # text embedding dim
    print(f"text embedding dim: {dim}")
    # Peek at the topo column names of the first dataset for a sanity check.
    print(f"example columns: {train_batches[0]['column_names'][:6]} "
          f"(F={len(train_batches[0]['column_names'])})")

    baseline, text = build_models(text_embedding_dim=dim)
    criterion = nn.CrossEntropyLoss()

    # Each model gets its own ChunkedLoader over the SAME batch list, so both see
    # identical data; every dataset is visited exactly once across the epochs.
    print(f"\n=== training BASELINE ({EPOCHS} epochs x "
          f"{len(train_batches) // EPOCHS} datasets, each used once) ===")
    baseline, _ = train(model=baseline, prior=ChunkedLoader(train_batches, EPOCHS),
                        criterion=criterion, epochs=EPOCHS, lr=LR, device=device,
                        callbacks=[ConsoleLoggerCallback()], run_name="topo_baseline")

    print(f"\n=== training TEXT ({EPOCHS} epochs x "
          f"{len(train_batches) // EPOCHS} datasets, each used once) ===")
    text, _ = train(model=text, prior=ChunkedLoader(train_batches, EPOCHS),
                    criterion=criterion, epochs=EPOCHS, lr=LR, device=device,
                    callbacks=[ConsoleLoggerCallback()], run_name="topo_text")

    print(f"\n=== evaluation on {N_EVAL_DATASETS} held-out datasets ===")
    base_metrics = evaluate(baseline, eval_batches, device, use_text=False)
    text_metrics = evaluate(text, eval_batches, device, use_text=True)

    print(f"\n{'model':<10}{'accuracy':>12}{'roc_auc':>12}")
    print("-" * 34)
    print(f"{'baseline':<10}{base_metrics['accuracy']:>12.4f}{base_metrics['roc_auc']:>12.4f}")
    print(f"{'text':<10}{text_metrics['accuracy']:>12.4f}{text_metrics['roc_auc']:>12.4f}")
    print("-" * 34)
    d_acc = text_metrics["accuracy"] - base_metrics["accuracy"]
    d_auc = text_metrics["roc_auc"] - base_metrics["roc_auc"]
    print(f"{'Δ text':<10}{d_acc:>+12.4f}{d_auc:>+12.4f}")
    print(f"(ROC-AUC averaged over {text_metrics['n_auc']}/{N_EVAL_DATASETS} "
          f"datasets with both classes in the test split)")

    # Head-to-head: on how many of the eval datasets does each model win (by accuracy)?
    text_wins = base_wins = ties = 0
    for a_base, a_text in zip(base_metrics["per_dataset_acc"], text_metrics["per_dataset_acc"]):
        if a_text > a_base:
            text_wins += 1
        elif a_base > a_text:
            base_wins += 1
        else:
            ties += 1
    print(f"\nhead-to-head over {N_EVAL_DATASETS} eval datasets (by accuracy):")
    print(f"  text wins    : {text_wins}")
    print(f"  baseline wins: {base_wins}")
    print(f"  ties         : {ties}")


if __name__ == "__main__":
    main()
