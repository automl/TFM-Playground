"""Configuration module for all priors."""

import torch

def get_ticl_prior_config(prior_type: str) -> dict:
    """Return the default kwargs for MLPPrior, GPPrior, or classification priors.
    
    Args:
        prior_type: Type of TICL prior ('mlp', 'gp', 'classification_adapter', etc.)
    """
    
    if prior_type == "mlp":
        return {
            "pre_sample_causes": True,
            "sampling": "normal",
            "prior_mlp_scale_weights_sqrt": True,
            "random_feature_rotation": True,
            "num_layers": {
                "distribution": "meta_gamma",
                "max_alpha": 2,
                "max_scale": 3,
                "round": True,
                "lower_bound": 2,
            },
            "prior_mlp_hidden_dim": {
                "distribution": "meta_gamma",
                "max_alpha": 3,
                "max_scale": 100,
                "round": True,
                "lower_bound": 4,
            },
            "prior_mlp_dropout_prob": {
                "distribution": "meta_beta",
                "scale": 0.6,
                "min": 0.1,
                "max": 5.0,
            },
            "init_std": {
                "distribution": "log_uniform",
                "min": 1e-2,
                "max": 12,
            },
            "noise_std": {
                "distribution": "log_uniform",
                "min": 1e-4,
                "max": 0.5,
            },
            "num_causes": {
                "distribution": "meta_gamma",
                "max_alpha": 3,
                "max_scale": 7,
                "round": True,
                "lower_bound": 2,
            },
            "is_causal": {
                "distribution": "meta_choice",
                "choice_values": [True, False],
            },
            "pre_sample_weights": {
                "distribution": "meta_choice",
                "choice_values": [True, False],
            },
            "y_is_effect": {
                "distribution": "meta_choice",
                "choice_values": [True, False],
            },
            "prior_mlp_activations": {
                "distribution": "meta_choice",
                "choice_values": [
                    torch.nn.Tanh,
                    torch.nn.Identity,
                    torch.nn.ReLU,
                ],
            },
            "block_wise_dropout": {
                "distribution": "meta_choice",
                "choice_values": [True, False],
            },
            "sort_features": {
                "distribution": "meta_choice",
                "choice_values": [True, False],
            },
            "in_clique": {
                "distribution": "meta_choice",
                "choice_values": [True, False],
            },
            "add_uninformative_features": False,
        }

    elif prior_type == "gp":
        return {
            "outputscale": {
                "distribution": "log_uniform",
                "min": 1e-5,
                "max": 8,
            },
            "lengthscale": {
                "distribution": "log_uniform",
                "min": 1e-5,
                "max": 8,
            },
            "noise": {
                "distribution": "meta_choice",
                "choice_values": [0.00001, 0.0001, 0.01],
            },
            "sampling": "normal",
        }

    elif prior_type == "classification_adapter":
        return {
            "max_num_classes": 10,
            "num_classes": {
                "distribution": "uniform_int",
                "min": 2,
                "max": 10,
            },
            "balanced": False,
            "output_multiclass_ordered_p": 0.0,
            "multiclass_max_steps": 10,
            "multiclass_type": "rank",
            "categorical_feature_p": 0.2,
            "nan_prob_no_reason": 0.0,
            "nan_prob_a_reason": 0.0,
            "set_value_to_nan": 0.9,
            "num_features_sampler": "uniform",
            "pad_zeros": True,
            "feature_curriculum": False,
        }

    elif prior_type == "boolean_conjunctions":
        return {
            "max_fraction_uninformative": 0.5,
            "p_uninformative": 0.5,
        }

    elif prior_type == "step_function":
        return {
            "max_steps": 1,
            "sampling": "uniform",
        }

    else:
        raise ValueError(f"Unsupported TICL prior type: {prior_type}")
    

TABPFN_DIFFERENTIABLE = True

def get_tabpfn_prior_config(prior_type: str) -> dict:
    """Return the default kwargs for TabPFN priors.
    
    Args:
        prior_type: Type of TabPFN prior ('mlp', 'gp', 'prior_bag')
    Note:
        gp_mix is included in the library wrapper but lacks implementation so its not included here
    """
    
    if prior_type == "mlp":
        return {
            'sampling': 'uniform',
            'num_layers': 2,
            'prior_mlp_hidden_dim': 64,
            'prior_mlp_activations': lambda: torch.nn.ReLU(),
            'mix_activations': False,
            'noise_std': 0.1,
            'prior_mlp_dropout_prob': 0.0,
            'init_std': 1.0,
            'prior_mlp_scale_weights_sqrt': True,
            'random_feature_rotation': True,
            'is_causal': False,
            'num_causes': 0,
            'y_is_effect': False,
            'pre_sample_causes': False,
            'pre_sample_weights': False,
            'block_wise_dropout': False,
            'add_uninformative_features': False,
            'sort_features': False,
            'in_clique': False,
        }
    elif prior_type == "gp":
        return {
            'noise': 1e-3,
            'outputscale': 3.0,
            'lengthscale': 1.0,
            'is_binary_classification': False,
            'normalize_by_used_features': True,
            'order_y': False,
            'sampling': 'uniform',
        }
    elif prior_type == "prior_bag":
        # prior bag combines MLP and GP priors
        mlp_config = get_tabpfn_prior_config("mlp")
        gp_config = get_tabpfn_prior_config("gp")
        return {
            **mlp_config,
            **gp_config,
            "prior_bag_exp_weights_1": 2.0, # GP gets weight 1.0, MLP gets this value (default 2.0).
        }
    else:
        raise ValueError(f"Unsupported TabPFN prior type: {prior_type}")


def get_tabpfn_differentiable_prior_config(prior_type: str) -> dict:
    """Return ranged differentiable hyperparameter configs for TabPFN priors.

    These dictionaries are consumed only when ``differentiable=True`` is passed.
    """

    if prior_type == "mlp":
        return {
            "num_layers": {
                "distribution": "meta_gamma",
                "max_alpha": 2,
                "max_scale": 3,
                "round": True,
                "lower_bound": 2,
            },
            "prior_mlp_hidden_dim": {
                "distribution": "meta_gamma",
                "max_alpha": 3,
                "max_scale": 100,
                "round": True,
                "lower_bound": 4,
            },
            "prior_mlp_dropout_prob": {
                "distribution": "meta_beta",
                "scale": 0.6,
                "min": 0.1,
                "max": 5.0,
            },
            "noise_std": {
                "distribution": "meta_trunc_norm_log_scaled",
                "max_mean": 0.3,
                "min_mean": 1e-4,
                "round": False,
                "lower_bound": 0.0,
            },
            "init_std": {
                "distribution": "meta_trunc_norm_log_scaled",
                "max_mean": 10.0,
                "min_mean": 1e-2,
                "round": False,
                "lower_bound": 0.0,
            },
            "num_causes": {
                "distribution": "meta_gamma",
                "max_alpha": 3,
                "max_scale": 7,
                "round": True,
                "lower_bound": 2,
            },
            "is_causal": {
                "distribution": "meta_choice",
                "choice_values": [True, False],
            },
            "pre_sample_weights": {
                "distribution": "meta_choice",
                "choice_values": [True, False],
            },
            "y_is_effect": {
                "distribution": "meta_choice",
                "choice_values": [True, False],
            },
            "sampling": {
                "distribution": "meta_choice",
                "choice_values": ["normal", "mixed"],
            },
            "prior_mlp_activations": {
                "distribution": "meta_choice_mixed",
                "choice_values": [torch.nn.Tanh, torch.nn.Identity, torch.nn.ReLU],
            },
            "block_wise_dropout": {
                "distribution": "meta_choice",
                "choice_values": [True, False],
            },
            "sort_features": {
                "distribution": "meta_choice",
                "choice_values": [True, False],
            },
            "in_clique": {
                "distribution": "meta_choice",
                "choice_values": [True, False],
            },
        }

    elif prior_type == "gp":
        return {
            "outputscale": {
                "distribution": "meta_trunc_norm_log_scaled",
                "max_mean": 10.0,
                "min_mean": 1e-5,
                "round": False,
                "lower_bound": 0.0,
            },
            "lengthscale": {
                "distribution": "meta_trunc_norm_log_scaled",
                "max_mean": 10.0,
                "min_mean": 1e-5,
                "round": False,
                "lower_bound": 0.0,
            },
            "noise": {
                "distribution": "meta_choice",
                "choice_values": [1e-5, 1e-4, 1e-2],
            },
        }

    elif prior_type == "prior_bag":
        mlp_config = get_tabpfn_differentiable_prior_config("mlp")
        gp_config = get_tabpfn_differentiable_prior_config("gp")
        return {
            **mlp_config,
            **gp_config,
            "prior_bag_exp_weights_1": {
                "distribution": "uniform",
                "min": 2.0,
                "max": 10.0,
            },
        }

    else:
        raise ValueError(f"Unsupported TabPFN prior type: {prior_type}")


def get_tabforest_prior_config(prior_type: str) -> dict:
    """Return the default kwargs for TabForestPFN generators.
    
    These generators are vendored from TabForestPFN (den Breejen et al.)
    and produce classification-only synthetic tabular data.
    
    Args:
        prior_type: Type of TabForestPFN prior ('forest', 'neighbor', 'cuts')
    """
    
    if prior_type == "forest":
        return {
            'base_size': 1000,       # number of random training points for the decision tree
            'n_estimators': 1,       # number of trees (unused in current impl, kept for API compat)
            'min_depth': 1,          # minimum tree depth
            'max_depth': 25,         # maximum tree depth
            'categorical_x': True,   # whether to randomly convert some features to categorical
        }
    elif prior_type == "neighbor":
        return {
            'min_neighbors': 2,      # minimum number of anchor points for KNN
            'max_neighbors': 2048,   # maximum number of anchor points for KNN
        }
    elif prior_type == "cuts":
        return {
            'n_cuts': 1000,          # number of random hyperplane cuts
        }
    else:
        raise ValueError(f"Unsupported TabForestPFN prior type: {prior_type}")
