"""Unit tests for config.py"""

import unittest
import torch
from tfmplayground.priors.config import get_ticl_prior_config


class TestGetTICLPriorConfig(unittest.TestCase):
    """Test cases for get_ticl_prior_config function."""

    def test_mlp_config(self):
        """Test MLP prior configuration."""
        config = get_ticl_prior_config("mlp")

        self.assertIsInstance(config, dict)
        self.assertEqual(config["sampling"], "uniform")
        self.assertEqual(config["num_layers"], 2)
        self.assertEqual(config["prior_mlp_hidden_dim"], 64)
        self.assertEqual(config["prior_mlp_activations"], torch.nn.Tanh)
        self.assertEqual(config["noise_std"], 0.05)
        self.assertEqual(config["prior_mlp_dropout_prob"], 0.0)
        self.assertIn("init_std", config)
        self.assertIn("random_feature_rotation", config)

    def test_gp_config(self):
        """Test GP prior configuration."""
        config = get_ticl_prior_config("gp")

        self.assertIsInstance(config, dict)
        self.assertEqual(config["sampling"], "uniform")
        self.assertEqual(config["noise"], 1e-3)
        self.assertEqual(config["outputscale"], 3.0)
        self.assertEqual(config["lengthscale"], 1.0)

    def test_classification_adapter_config(self):
        """Test classification adapter prior configuration."""
        config = get_ticl_prior_config("classification_adapter")

        self.assertIsInstance(config, dict)
        self.assertEqual(config["max_num_classes"], 50)
        self.assertIsNone(config["num_classes"])
        self.assertFalse(config["balanced"])
        self.assertIn("output_multiclass_ordered_p", config)
        self.assertIn("multiclass_type", config)
        self.assertIn("categorical_feature_p", config)

    def test_classification_adapter_with_max_classes(self):
        """Test classification adapter with custom max_num_classes."""
        max_classes = 10
        config = get_ticl_prior_config(
            "classification_adapter", max_num_classes=max_classes
        )

        self.assertEqual(config["num_classes"], max_classes)
        self.assertEqual(config["max_num_classes"], 50)

    def test_boolean_conjunctions_config(self):
        """Test boolean conjunctions prior configuration."""
        config = get_ticl_prior_config("boolean_conjunctions")

        self.assertIsInstance(config, dict)
        self.assertEqual(config["max_rank"], 20)
        self.assertEqual(config["max_fraction_uninformative"], 0.3)
        self.assertEqual(config["p_uninformative"], 0.3)
        self.assertFalse(config["verbose"])

    def test_step_function_config(self):
        """Test step function prior configuration."""
        config = get_ticl_prior_config("step_function")

        self.assertIsInstance(config, dict)
        self.assertEqual(config["max_steps"], 1)
        self.assertEqual(config["sampling"], "uniform")

    def test_invalid_prior_type(self):
        """Test that invalid prior type raises ValueError."""
        with self.assertRaises(ValueError) as context:
            get_ticl_prior_config("invalid_prior_type")

        self.assertIn("Unsupported TICL prior type", str(context.exception))

    def test_all_configs_return_dict(self):
        """Test that all valid prior types return dictionaries."""
        prior_types = [
            "mlp",
            "gp",
            "classification_adapter",
            "boolean_conjunctions",
            "step_function",
        ]

        for prior_type in prior_types:
            with self.subTest(prior_type=prior_type):
                config = get_ticl_prior_config(prior_type)
                self.assertIsInstance(config, dict)
                self.assertGreater(len(config), 0)


if __name__ == "__main__":
    unittest.main()
