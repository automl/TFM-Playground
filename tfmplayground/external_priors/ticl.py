"""DataLoader and configuration for TICL-based priors."""

import torch
from ticl.dataloader import PriorDataLoader as TICLPriorDataset
from ticl.priors import BooleanConjunctionPrior, ClassificationAdapterPrior, GPPrior, MLPPrior, StepFunctionPrior
from torch.utils.data import DataLoader


def _get_ticl_prior_config(prior_type: str) -> dict:
    """Return the default kwargs for MLPPrior, GPPrior, or classification priors.

    Args:
        prior_type: Type of TICL prior ('mlp', 'gp', 'classification_adapter', etc.)
    """

    if prior_type == "mlp":
        return {
            "sampling": "uniform",
            "num_layers": 2,
            "prior_mlp_hidden_dim": 64,
            "prior_mlp_activations": torch.nn.Tanh,
            "noise_std": 0.05,
            "prior_mlp_dropout_prob": 0.0,
            "init_std": 1.0,
            "prior_mlp_scale_weights_sqrt": True,
            "block_wise_dropout": False,
            "is_causal": False,
            "num_causes": 0,
            "y_is_effect": False,
            "pre_sample_causes": False,
            "pre_sample_weights": False,
            "random_feature_rotation": True,
            "add_uninformative_features": False,
            "sort_features": False,
            "in_clique": False,
        }
    elif prior_type == "gp":
        return {
            "sampling": "uniform",
            "noise": 1e-3,
            "outputscale": 3.0,
            "lengthscale": 1.0,
        }
    elif prior_type == "classification_adapter":
        return {
            "balanced": False,
            "output_multiclass_ordered_p": 0.1,
            "multiclass_type": "rank",
            "categorical_feature_p": 0.15,
            "nan_prob_no_reason": 0.05,
            "nan_prob_a_reason": 0.03,
            "set_value_to_nan": 0.9,
            "num_features_sampler": "uniform",
            "pad_zeros": False,
            "feature_curriculum": False,
        }
    elif prior_type == "boolean_conjunctions":
        return {"max_rank": 20, "max_fraction_uninformative": 0.3, "p_uninformative": 0.3, "verbose": False}
    elif prior_type == "step_function":
        return {
            "max_steps": 1,
            "sampling": "uniform",
        }
    else:
        raise ValueError(f"Unsupported TICL prior type: {prior_type}")


def build_ticl_prior(
    prior_type: str, base_prior: str = None, max_num_classes: int = None
) -> MLPPrior | GPPrior | ClassificationAdapterPrior | BooleanConjunctionPrior | StepFunctionPrior:
    """Builds a TICL prior based on the prior type string using the defaults in this module.

    Args:
        prior_type: Type of TICL prior ('mlp', 'gp', 'classification_adapter', etc.)
        base_prior: Base regression prior for composite priors (e.g., 'mlp' or 'gp' for classification_adapter)
        max_num_classes: Maximum number of classes for classification priors
    """

    cfg = _get_ticl_prior_config(prior_type)

    if prior_type == "mlp":
        return MLPPrior(cfg)
    elif prior_type == "gp":
        return GPPrior(cfg)
    elif prior_type == "classification_adapter":
        if base_prior is None:
            base_prior = "mlp"  # default to MLP
        # build the base regression prior
        base_prior_obj = build_ticl_prior(base_prior)

        # we equate them rather than treating num_classes as a separate parameter because:
        # - max_num_classes serves as the upper bound for TICL's internal sampling
        # - even with num_classes set to a constant, TICL's class_sampler_f() will internally
        #   vary the actual number of classes (50% chance of 2, 50% chance of uniform(2, num_classes))
        cfg["max_num_classes"] = max_num_classes
        cfg["num_classes"] = max_num_classes
        return ClassificationAdapterPrior(base_prior_obj, **cfg)
    elif prior_type == "boolean_conjunctions":
        return BooleanConjunctionPrior(hyperparameters=cfg)
    elif prior_type == "step_function":
        return StepFunctionPrior(cfg)
    else:
        raise ValueError(f"Unsupported TICL prior type: {prior_type}")


class TICLPriorDataLoader(DataLoader):
    """DataLoader sampling synthetic prior data from TICL's PriorDataLoader.

    Args:
        prior (Any): A TICL prior object supporting get_batch.
        num_steps (int): Number of batches per epoch.
        batch_size (int): Number of functions sampled per batch.
        num_datapoints_max (int): Number of datapoints sampled per function.
        num_features (int): Dimensionality of x vectors.
        device (torch.device): Target device for tensors.
        min_eval_pos (int, optional): Minimum evaluation position in the sequence.
    """

    def __init__(
        self,
        prior,
        num_steps: int,
        batch_size: int,
        num_datapoints_max: int,
        num_features: int,
        min_eval_pos: int,
        device: torch.device,
    ):
        self.num_steps = num_steps
        self.device = device

        self.pd = TICLPriorDataset(
            prior=prior,
            num_steps=num_steps,
            batch_size=batch_size,
            min_eval_pos=min_eval_pos,
            n_samples=num_datapoints_max,
            device=device,
            num_features=num_features,
        )

    def ticl_to_ours(self, d):
        (info, x, y), target_y, train_test_split_index = d
        x = x.permute(1, 0, 2)
        y = y.permute(1, 0)
        target_y = target_y.permute(1, 0)

        return dict(
            x=x.to(self.device),
            y=y.to(self.device),
            target_y=target_y.to(self.device),  # target_y is identical to y (for downstream compatibility)
            train_test_split_index=train_test_split_index,
        )

    def __iter__(self):
        return (self.ticl_to_ours(batch) for batch in self.pd)

    def __len__(self):
        return self.num_steps
