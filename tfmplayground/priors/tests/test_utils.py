"""Unit tests for utils.py"""

import os
import tempfile
import unittest
import h5py
import numpy as np
import torch
from tfmplayground.priors.utils import build_ticl_prior, dump_prior_to_h5


class TestBuildTICLPrior(unittest.TestCase):
    """Test cases for build_ticl_prior function."""

    def test_build_mlp_prior(self):
        """Test building MLP prior."""
        prior = build_ticl_prior("mlp")

        self.assertIsNotNone(prior)
        # Check that it's a valid prior object
        self.assertTrue(hasattr(prior, "hyperparameters") or hasattr(prior, "__dict__"))

    def test_build_gp_prior(self):
        """Test building GP prior."""
        prior = build_ticl_prior("gp")

        self.assertIsNotNone(prior)

    def test_build_classification_adapter_default(self):
        """Test building classification adapter with default base prior (MLP)."""
        prior = build_ticl_prior("classification_adapter", max_num_classes=10)

        self.assertIsNotNone(prior)

    def test_build_classification_adapter_with_gp(self):
        """Test building classification adapter with GP base prior."""
        prior = build_ticl_prior(
            "classification_adapter", base_prior_type="gp", max_num_classes=5
        )

        self.assertIsNotNone(prior)

    def test_build_boolean_conjunctions(self):
        """Test building boolean conjunctions prior."""
        prior = build_ticl_prior("boolean_conjunctions")

        self.assertIsNotNone(prior)

    def test_build_step_function(self):
        """Test building step function prior."""
        prior = build_ticl_prior("step_function")

        self.assertIsNotNone(prior)

    def test_invalid_prior_type(self):
        """Test that invalid prior type raises ValueError."""
        with self.assertRaises(ValueError) as context:
            build_ticl_prior("nonexistent_prior")

        self.assertIn("Unsupported TICL prior type", str(context.exception))

    def test_classification_adapter_with_custom_max_classes(self):
        """Test classification adapter respects max_num_classes parameter."""
        max_classes = 15
        prior = build_ticl_prior("classification_adapter", max_num_classes=max_classes)

        self.assertIsNotNone(prior)


class TestDumpPriorToH5(unittest.TestCase):
    """Test cases for dump_prior_to_h5 function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.device = torch.device("cpu")

    def tearDown(self):
        """Clean up test files."""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _create_mock_prior_data(
        self, batch_size, max_seq_len, max_features, num_batches=2
    ):
        """Create mock prior data for testing."""
        for _ in range(num_batches):
            x = torch.randn(batch_size, max_seq_len, max_features)
            y = torch.randn(batch_size, max_seq_len)
            single_eval_pos = max_seq_len // 2

            yield {"x": x, "y": y, "single_eval_pos": single_eval_pos}

    def test_dump_regression_prior(self):
        """Test dumping regression prior data to H5."""
        save_path = os.path.join(self.temp_dir, "test_regression.h5")
        batch_size = 4
        max_seq_len = 100
        max_features = 10

        mock_data = self._create_mock_prior_data(batch_size, max_seq_len, max_features)

        dump_prior_to_h5(
            prior=mock_data,
            max_classes=None,
            batch_size=batch_size,
            save_path=save_path,
            problem_type="regression",
            max_seq_len=max_seq_len,
            max_features=max_features,
        )

        # Verify file was created
        self.assertTrue(os.path.exists(save_path))

        # Verify file contents
        with h5py.File(save_path, "r") as f:
            self.assertIn("X", f)
            self.assertIn("y", f)
            self.assertIn("single_eval_pos", f)
            self.assertIn("problem_type", f)
            self.assertIn("num_features", f)
            self.assertIn("num_datapoints", f)

            # Check shapes
            self.assertEqual(f["X"].shape[1], max_seq_len)
            self.assertEqual(f["X"].shape[2], max_features)
            self.assertEqual(f["y"].shape[1], max_seq_len)

            # Check problem type
            self.assertEqual(f["problem_type"][()].decode("utf-8"), "regression")

    def test_dump_classification_prior(self):
        """Test dumping classification prior data to H5."""
        save_path = os.path.join(self.temp_dir, "test_classification.h5")
        batch_size = 4
        max_seq_len = 100
        max_features = 10
        max_classes = 5

        mock_data = self._create_mock_prior_data(batch_size, max_seq_len, max_features)

        dump_prior_to_h5(
            prior=mock_data,
            max_classes=max_classes,
            batch_size=batch_size,
            save_path=save_path,
            problem_type="classification",
            max_seq_len=max_seq_len,
            max_features=max_features,
        )

        # Verify file was created
        self.assertTrue(os.path.exists(save_path))

        # Verify file contents
        with h5py.File(save_path, "r") as f:
            self.assertIn("max_num_classes", f)
            self.assertEqual(f["max_num_classes"][0], max_classes)
            self.assertEqual(f["problem_type"][()].decode("utf-8"), "classification")

    def test_dump_handles_tensor_single_eval_pos(self):
        """Test that dump handles single_eval_pos as both int and tensor."""
        save_path = os.path.join(self.temp_dir, "test_tensor_eval.h5")
        batch_size = 2
        max_seq_len = 50
        max_features = 5

        def mock_data_with_tensor():
            x = torch.randn(batch_size, max_seq_len, max_features)
            y = torch.randn(batch_size, max_seq_len)
            single_eval_pos = torch.tensor(25)  # Tensor instead of int

            yield {"x": x, "y": y, "single_eval_pos": single_eval_pos}

        dump_prior_to_h5(
            prior=mock_data_with_tensor(),
            max_classes=None,
            batch_size=batch_size,
            save_path=save_path,
            problem_type="regression",
            max_seq_len=max_seq_len,
            max_features=max_features,
        )

        # Verify it didn't crash and file was created
        self.assertTrue(os.path.exists(save_path))

    def test_dump_with_padding(self):
        """Test that data is correctly padded to max dimensions."""
        save_path = os.path.join(self.temp_dir, "test_padding.h5")
        batch_size = 2
        max_seq_len = 100
        max_features = 20

        # Create data smaller than max dimensions
        actual_seq_len = 50
        actual_features = 10

        def small_mock_data():
            x = torch.randn(batch_size, actual_seq_len, actual_features)
            y = torch.randn(batch_size, actual_seq_len)
            yield {"x": x, "y": y, "single_eval_pos": 25}

        dump_prior_to_h5(
            prior=small_mock_data(),
            max_classes=None,
            batch_size=batch_size,
            save_path=save_path,
            problem_type="regression",
            max_seq_len=max_seq_len,
            max_features=max_features,
        )

        with h5py.File(save_path, "r") as f:
            # Verify data was padded to max dimensions
            self.assertEqual(f["X"].shape[1], max_seq_len)
            self.assertEqual(f["X"].shape[2], max_features)

            # Verify original data dimensions are recorded
            self.assertEqual(f["num_features"][0], actual_features)
            self.assertEqual(f["num_datapoints"][0], actual_seq_len)


if __name__ == "__main__":
    unittest.main()
