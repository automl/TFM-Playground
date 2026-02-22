"""Configuration module for all priors."""

import torch


def get_ticl_prior_config(prior_type: str) -> dict:
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
        return {
            "max_rank": 20,
            "max_fraction_uninformative": 0.3,
            "p_uninformative": 0.3,
            "verbose": False,
        }
    elif prior_type == "step_function":
        return {
            "max_steps": 1,
            "sampling": "uniform",
        }
    else:
        raise ValueError(f"Unsupported TICL prior type: {prior_type}")


def get_tabpfn_prior_config(prior_type: str) -> dict:
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
        mlp_config = get_tabpfn_prior_config("mlp")
        gp_config = get_tabpfn_prior_config("gp")
        return {
            **mlp_config,
            **gp_config,
            "prior_bag_exp_weights_1": 2.0,  # GP gets weight 1.0, MLP gets this value (default 2.0).
        }
    else:
        raise ValueError(f"Unsupported TabPFN prior type: {prior_type}")
