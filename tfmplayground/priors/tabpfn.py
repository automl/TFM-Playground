"""DataLoader and configuration for TabPFN v1-based priors."""

import torch
from tabpfn_prior import TabPFNPriorDataLoader  # noqa: F401


def _get_tabpfn_prior_config(prior_type: str) -> dict:
    """Return the default kwargs for TabPFN priors.

    Args:
        prior_type: Type of TabPFN prior ('mlp', 'gp', 'prior_bag')
    Note:
        gp_mix is included in the library wrapper but lacks implementation so its not included here
    """

    if prior_type == "mlp":
        return {
            "sampling": "uniform",
            "num_layers": 2,
            "prior_mlp_hidden_dim": 64,
            "prior_mlp_activations": lambda: torch.nn.ReLU(),
            "mix_activations": False,
            "noise_std": 0.1,
            "prior_mlp_dropout_prob": 0.0,
            "init_std": 1.0,
            "prior_mlp_scale_weights_sqrt": True,
            "random_feature_rotation": True,
            "is_causal": False,
            "num_causes": 0,
            "y_is_effect": False,
            "pre_sample_causes": False,
            "pre_sample_weights": False,
            "block_wise_dropout": False,
            "add_uninformative_features": False,
            "sort_features": False,
            "in_clique": False,
        }
    elif prior_type == "gp":
        return {
            "noise": 0.1,
            "outputscale": 1.0,
            "lengthscale": 0.2,
            "is_binary_classification": False,
            "normalize_by_used_features": True,
            "order_y": False,
            "sampling": "uniform",
        }
    elif prior_type == "prior_bag":
        # prior bag combines MLP and GP priors
        mlp_config = _get_tabpfn_prior_config("mlp")
        gp_config = _get_tabpfn_prior_config("gp")
        return {
            **mlp_config,
            **gp_config,
            "prior_bag_exp_weights_1": 2.0,  # GP gets weight 1.0, MLP gets this value (default 2.0).
        }
    else:
        raise ValueError(f"Unsupported TabPFN prior type: {prior_type}")


def build_tabpfn_prior(prior_type: str, max_classes: int) -> dict:
    """Builds TabPFN prior configuration with appropriate settings for regression or classification.

    Args:
        prior_type: Type of TabPFN prior ('mlp', 'gp', 'prior_bag')
        max_classes: Maximum number of classes

    Returns:
        dict with 'flexible', 'max_num_classes', and 'prior_config' keys
    """
    is_regression = max_classes == 0

    return {
        "flexible": not is_regression,  # false for regression, true for classification
        "max_num_classes": 2
        if is_regression
        else max_classes,  # library weirdly requires >=2 regardless of regression or classification
        # num_classes parameter in the library code is equated to max_num_classes
        # so its not varied separately here
        "prior_config": {
            **_get_tabpfn_prior_config(prior_type),
        },
    }
