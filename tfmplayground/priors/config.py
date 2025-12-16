"""Configuration module for TICL priors."""

import torch

# extensible prior registry: add new libraries here
PRIOR_REGISTRY = {
    "ticl": {
        "priors": ["mlp", "gp", "classification_adapter", "boolean_conjunctions", "step_function"],
        "classification_priors": ["classification_adapter", "boolean_conjunctions", "step_function"],
        "composite_priors": ["classification_adapter"],
        "default_prior": "mlp",
    },
    "tabicl": {
        "priors": ["mlp_scm", "tree_scm", "mix_scm", "dummy"],
        "classification_priors": ["mlp_scm", "tree_scm", "mix_scm", "dummy"],  # TabICL is classification-only
        "composite_priors": [],
        "default_prior": "mix_scm",
    }
}

def get_available_libraries():
    """Return list of available library names."""
    return list(PRIOR_REGISTRY.keys())

def get_priors_for_lib(lib: str):
    """Return available priors for a given library."""
    if lib not in PRIOR_REGISTRY:
        raise ValueError(f"Unknown library: '{lib}'. Available: {', '.join(get_available_libraries())}")
    return PRIOR_REGISTRY[lib]["priors"]

def is_classification_prior(lib: str, prior_type: str):
    """Check if a prior is a classification prior."""
    if lib not in PRIOR_REGISTRY:
        return False
    return prior_type in PRIOR_REGISTRY[lib]["classification_priors"]

def is_composite_prior(lib: str, prior_type: str):
    """Check if a prior requires a base prior."""
    if lib not in PRIOR_REGISTRY:
        return False
    return prior_type in PRIOR_REGISTRY[lib]["composite_priors"]

def get_default_prior(lib: str):
    """Get the default prior type for a library."""
    if lib not in PRIOR_REGISTRY:
        raise ValueError(f"Unknown library: '{lib}'. Available: {', '.join(get_available_libraries())}")
    return PRIOR_REGISTRY[lib]["default_prior"]

def get_ticl_prior_config(prior_type: str, max_num_classes: int = None) -> dict:
    """Return the default kwargs for MLPPrior, GPPrior, or classification priors."""
    
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
            "max_num_classes": 50,  # this is a global upper bound for how many output classes the generator or model can handle;
                                    # setting it > 0 keeps classification mode active (0 = regression),
                                    # and actual tasks will use up to this many classes depending on num_classes and random sampling
            "num_classes": max_num_classes,
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
          'max_rank': 20,
          'max_fraction_uninformative': 0.3,
          'p_uninformative': 0.3,
          'verbose': False
        }
    elif prior_type == "step_function":
        return {
            "max_steps": 1,
            "sampling": "uniform",
        }
    else:
        raise ValueError(f"Unsupported TICL prior type: {prior_type}")
