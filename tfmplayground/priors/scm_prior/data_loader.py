"""
Topo-aware DataLoader for the Do-PFN SCM prior.

Wraps ObservationalDataLoader to additionally output:
  - topo_ranks : float tensor of shape (num_features,)
      Normalized topological rank [0, 1] for each feature column in x.
      Features from a node at topo position k get rank k / (num_nodes - 1).
  - column_names : list[str]
      Human-readable names like "topo_0_x3_f0", encoding causal position.

This is the bridge between the causal graph structure and the model's
feature encoder: the column-name prefix convention becomes a numeric
rank that the TopoTabPFN model can condition on.
"""
from __future__ import annotations

from typing import Any, Dict, List

import networkx as nx
import torch

from dopfnprior.dataloaders.observational_dataloader import ObservationalDataLoader
from dopfnprior.configs.default_config import prior_config as DEFAULT_PRIOR_CONFIG


class TopoAwareDataLoader(ObservationalDataLoader):
    """
    Extends ObservationalDataLoader to also return topological rank
    information for each feature column.

    Parameters
    ----------
    Same as ObservationalDataLoader, plus:
    binarize_targets : bool
        If True (default), binarize continuous SCM targets at the per-dataset
        median so the batch is compatible with cross-entropy classification loss.
    """

    def __init__(self, *args, binarize_targets: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.binarize_targets = binarize_targets

    # ------------------------------------------------------------------
    # Core override
    # ------------------------------------------------------------------

    def batch_function(self) -> Dict[str, Any]:
        batch = super().batch_function()

        scm = batch["scm"]
        graph = scm.dag                          # nx.DiGraph
        topo_order: List[str] = scm._topo        # already computed in SCM.__init__
        n_nodes = len(topo_order)

        # Build topo_ranks and column_names in the same order that
        # ObservationalDataLoader concatenates features into batch['x'].
        # That order is exactly scm._topo (see scm.propagate → xs dict).
        topo_ranks: List[float] = []
        column_names: List[str] = []

        for rank, v in enumerate(topo_order):
            n_vis = graph.nodes[v]["n_visible"]
            if v == "y":
                # The first visible feature of 'y' became the regression target;
                # remaining features (if any) appear in x.
                n_vis = max(0, n_vis - 1)
            normalized_rank = rank / max(1, n_nodes - 1)
            for j in range(n_vis):
                topo_ranks.append(normalized_rank)
                column_names.append(f"topo_{rank}_{v}_f{j}")

        batch["topo_ranks"] = torch.tensor(topo_ranks, dtype=torch.float32)
        batch["column_names"] = column_names

        # Binarize continuous targets at the per-dataset median so the data
        # is compatible with classification models (cross-entropy).
        if self.binarize_targets:
            y = batch["y"]  # (B, N, 1)  continuous
            median = y.median(dim=1, keepdim=True).values
            batch["y"] = (y > median).float()
            # target_y is what the train loop optimizes (cast to long for
            # cross-entropy); it MUST hold the binarized labels too, otherwise
            # training regresses onto the continuous target cast to int and the
            # classifier collapses to always predicting class 0.
            batch["target_y"] = batch["y"]

        return batch


# ------------------------------------------------------------------
# Convenience factory
# ------------------------------------------------------------------

def make_dataloader(
    num_steps: int = 200,
    batch_size: int = 4,
    seed: int = 42,
    prior_config: Dict | None = None,
) -> TopoAwareDataLoader:
    """Create a TopoAwareDataLoader with sensible demo-scale defaults."""
    cfg = prior_config if prior_config is not None else DEFAULT_PRIOR_CONFIG
    return TopoAwareDataLoader(
        num_steps=num_steps,
        batch_size=batch_size,
        prior_config=cfg,
        seed=seed,
        binarize_targets=True,
    )
