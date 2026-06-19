
import torch
from tabpfn_prior import TabPFNPriorDataLoader  # noqa: F401


def _get_tabpfn_prior_config(prior_type):

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
        mlp_config = _get_tabpfn_prior_config("mlp")
        gp_config = _get_tabpfn_prior_config("gp")
        return {
            **mlp_config,
            **gp_config,
            "prior_bag_exp_weights_1": 2.0,
        }
    else:
        raise ValueError(f"Unsupported TabPFN prior type: {prior_type}")


def build_tabpfn_prior(prior_type, max_classes):
    is_regression = max_classes == 0

    return {
        "flexible": not is_regression,
        "max_num_classes": 2
        if is_regression
        else max_classes,
        "prior_config": {
            **_get_tabpfn_prior_config(prior_type),
        },
    }
