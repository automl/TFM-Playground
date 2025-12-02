"""Unit tests for dataloader.py"""

import os
import tempfile
import unittest
import h5py
import numpy as np
import torch
from tfmplayground.priors.dataloader import (
    PriorDataLoader,
    PriorDumpDataLoader,
    TabICLPriorDataLoader,
    TICLPriorDataLoader,
)


class TestPriorDataLoader(unittest.TestCase):
    """Test cases for PriorDataLoader class."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.num_steps = 5
        self.batch_size = 4
        self.num_datapoints_max = 100
        self.num_features = 10

    def _mock_get_batch(self, batch_size, num_datapoints_max, num_features):
        """Mock get_batch function for testing."""
        x = torch.randn(batch_size, num_datapoints_max, num_features)
        y = torch.randn(batch_size, num_datapoints_max)
        return {
            "x": x,
            "y": y,
            "target_y": y,
            "single_eval_pos": num_datapoints_max // 2,
        }

    def test_initialization(self):
        """Test PriorDataLoader initialization."""
        loader = PriorDataLoader(
            get_batch_function=self._mock_get_batch,
            num_steps=self.num_steps,
            batch_size=self.batch_size,
            num_datapoints_max=self.num_datapoints_max,
            num_features=self.num_features,
            device=self.device,
        )

        self.assertEqual(loader.num_steps, self.num_steps)
        self.assertEqual(loader.batch_size, self.batch_size)
        self.assertEqual(loader.num_datapoints_max, self.num_datapoints_max)
        self.assertEqual(loader.num_features, self.num_features)

    def test_iteration(self):
        """Test iterating through PriorDataLoader."""
        loader = PriorDataLoader(
            get_batch_function=self._mock_get_batch,
            num_steps=self.num_steps,
            batch_size=self.batch_size,
            num_datapoints_max=self.num_datapoints_max,
            num_features=self.num_features,
            device=self.device,
        )

        batch_count = 0
        for batch in loader:
            self.assertIn("x", batch)
            self.assertIn("y", batch)
            self.assertIn("single_eval_pos", batch)
            self.assertEqual(batch["x"].shape[0], self.batch_size)
            batch_count += 1

        self.assertEqual(batch_count, self.num_steps)

    def test_length(self):
        """Test __len__ method."""
        loader = PriorDataLoader(
            get_batch_function=self._mock_get_batch,
            num_steps=self.num_steps,
            batch_size=self.batch_size,
            num_datapoints_max=self.num_datapoints_max,
            num_features=self.num_features,
            device=self.device,
        )

        self.assertEqual(len(loader), self.num_steps)


class TestPriorDumpDataLoader(unittest.TestCase):
    """Test cases for PriorDumpDataLoader class."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "test_prior.h5")

        # Create a mock H5 file
        self._create_mock_h5_file()

    def tearDown(self):
        """Clean up test files."""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _create_mock_h5_file(
        self, num_batches=10, batch_size=4, max_seq_len=100, max_features=10
    ):
        """Create a mock H5 file for testing."""
        total_samples = num_batches * batch_size

        with h5py.File(self.test_file, "w") as f:
            # Create datasets
            f.create_dataset(
                "X", data=np.random.randn(total_samples, max_seq_len, max_features)
            )
            f.create_dataset("y", data=np.random.randn(total_samples, max_seq_len))
            f.create_dataset(
                "single_eval_pos",
                data=np.full(total_samples, max_seq_len // 2, dtype=np.int32),
            )
            f.create_dataset(
                "num_features",
                data=np.full(total_samples, max_features, dtype=np.int32),
            )
            f.create_dataset(
                "num_datapoints",
                data=np.full(total_samples, max_seq_len, dtype=np.int32),
            )
            f.create_dataset("max_num_classes", data=np.array([5]))
            f.create_dataset(
                "problem_type", data="classification", dtype=h5py.string_dtype()
            )

    def test_initialization(self):
        """Test PriorDumpDataLoader initialization."""
        loader = PriorDumpDataLoader(
            filename=self.test_file, num_steps=5, batch_size=4, device=self.device
        )

        self.assertEqual(loader.filename, self.test_file)
        self.assertEqual(loader.num_steps, 5)
        self.assertEqual(loader.batch_size, 4)
        self.assertIsNotNone(loader.max_num_classes)
        self.assertEqual(loader.problem_type, "classification")

    def test_iteration(self):
        """Test iterating through PriorDumpDataLoader."""
        num_steps = 3
        batch_size = 4

        loader = PriorDumpDataLoader(
            filename=self.test_file,
            num_steps=num_steps,
            batch_size=batch_size,
            device=self.device,
        )

        batch_count = 0
        for batch in loader:
            self.assertIn("x", batch)
            self.assertIn("y", batch)
            self.assertIn("target_y", batch)
            self.assertIn("single_eval_pos", batch)
            self.assertEqual(batch["x"].shape[0], batch_size)
            batch_count += 1

        self.assertEqual(batch_count, num_steps)

    def test_length(self):
        """Test __len__ method."""
        loader = PriorDumpDataLoader(
            filename=self.test_file, num_steps=7, batch_size=4, device=self.device
        )

        self.assertEqual(len(loader), 7)

    def test_pointer_wraps_around(self):
        """Test that pointer wraps around when reaching end of file."""
        # Create small file
        small_file = os.path.join(self.temp_dir, "small_prior.h5")
        self._create_mock_h5_file_at_path(small_file, num_batches=2, batch_size=2)

        # Request more steps than available
        loader = PriorDumpDataLoader(
            filename=small_file, num_steps=5, batch_size=2, device=self.device
        )

        # Should not crash and wrap around
        batch_count = sum(1 for _ in loader)
        self.assertEqual(batch_count, 5)

    def _create_mock_h5_file_at_path(
        self, path, num_batches=10, batch_size=4, max_seq_len=100, max_features=10
    ):
        """Create a mock H5 file at specific path."""
        total_samples = num_batches * batch_size

        with h5py.File(path, "w") as f:
            f.create_dataset(
                "X", data=np.random.randn(total_samples, max_seq_len, max_features)
            )
            f.create_dataset("y", data=np.random.randn(total_samples, max_seq_len))
            f.create_dataset(
                "single_eval_pos",
                data=np.full(total_samples, max_seq_len // 2, dtype=np.int32),
            )
            f.create_dataset(
                "num_features",
                data=np.full(total_samples, max_features, dtype=np.int32),
            )
            f.create_dataset(
                "num_datapoints",
                data=np.full(total_samples, max_seq_len, dtype=np.int32),
            )
            f.create_dataset(
                "problem_type", data="regression", dtype=h5py.string_dtype()
            )

    def test_starting_index(self):
        """Test loading with custom starting_index."""
        loader = PriorDumpDataLoader(
            filename=self.test_file,
            num_steps=2,
            batch_size=4,
            device=self.device,
            starting_index=8,  # Start from second batch
        )

        self.assertEqual(loader.pointer, 8)

    def test_regression_file_no_max_classes(self):
        """Test loading regression file without max_num_classes."""
        reg_file = os.path.join(self.temp_dir, "regression.h5")
        self._create_mock_h5_file_at_path(reg_file, num_batches=5)

        loader = PriorDumpDataLoader(
            filename=reg_file, num_steps=2, batch_size=4, device=self.device
        )

        self.assertIsNone(loader.max_num_classes)
        self.assertEqual(loader.problem_type, "regression")


class TestTabICLPriorDataLoader(unittest.TestCase):
    """Test cases for TabICLPriorDataLoader class."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")

    def test_initialization(self):
        """Test TabICLPriorDataLoader initialization."""
        loader = TabICLPriorDataLoader(
            num_steps=5,
            batch_size=4,
            num_datapoints_min=10,
            num_datapoints_max=100,
            min_features=5,
            max_features=20,
            max_num_classes=10,
            device=self.device,
        )

        self.assertEqual(loader.num_steps, 5)
        self.assertEqual(loader.batch_size, 4)
        self.assertEqual(loader.max_num_classes, 10)
        self.assertIsNotNone(loader.pd)

    def test_length(self):
        """Test __len__ method."""
        loader = TabICLPriorDataLoader(
            num_steps=8,
            batch_size=4,
            num_datapoints_min=10,
            num_datapoints_max=100,
            min_features=5,
            max_features=20,
            max_num_classes=10,
            device=self.device,
        )

        self.assertEqual(len(loader), 8)

    def test_tabicl_to_ours_conversion(self):
        """Test tabicl_to_ours conversion method."""
        loader = TabICLPriorDataLoader(
            num_steps=1,
            batch_size=4,
            num_datapoints_min=10,
            num_datapoints_max=100,
            min_features=5,
            max_features=20,
            max_num_classes=10,
            device=self.device,
        )

        # Mock TabICL data format
        x = torch.randn(4, 100, 20)
        y = torch.randn(4, 100)
        active_features = torch.tensor([15, 15, 15, 15])
        seqlen = torch.tensor([80, 80, 80, 80])
        train_size = torch.tensor([50, 50, 50, 50])

        mock_data = (x, y, active_features, seqlen, train_size)

        result = loader.tabicl_to_ours(mock_data)

        self.assertIn("x", result)
        self.assertIn("y", result)
        self.assertIn("target_y", result)
        self.assertIn("single_eval_pos", result)
        self.assertEqual(result["single_eval_pos"], 50)


class TestTICLPriorDataLoader(unittest.TestCase):
    """Test cases for TICLPriorDataLoader class."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")

    def _create_mock_prior(self):
        """Create a mock TICL prior."""

        class MockPrior:
            def __call__(self, *args, **kwargs):
                return None

        return MockPrior()

    def test_initialization(self):
        """Test TICLPriorDataLoader initialization."""
        mock_prior = self._create_mock_prior()

        loader = TICLPriorDataLoader(
            prior=mock_prior,
            num_steps=5,
            batch_size=4,
            num_datapoints_max=100,
            num_features=10,
            device=self.device,
            min_eval_pos=10,
        )

        self.assertEqual(loader.num_steps, 5)
        self.assertIsNotNone(loader.pd)

    def test_length(self):
        """Test __len__ method."""
        mock_prior = self._create_mock_prior()

        loader = TICLPriorDataLoader(
            prior=mock_prior,
            num_steps=6,
            batch_size=4,
            num_datapoints_max=100,
            num_features=10,
            device=self.device,
        )

        self.assertEqual(len(loader), 6)

    def test_ticl_to_ours_conversion(self):
        """Test ticl_to_ours conversion method."""
        mock_prior = self._create_mock_prior()

        loader = TICLPriorDataLoader(
            prior=mock_prior,
            num_steps=1,
            batch_size=4,
            num_datapoints_max=100,
            num_features=10,
            device=self.device,
        )

        # Mock TICL data format (note: different shape from ours)
        info = {}
        x = torch.randn(100, 4, 10)  # (seq_len, batch, features)
        y = torch.randn(100, 4)  # (seq_len, batch)
        target_y = torch.randn(100, 4)
        single_eval_pos = 50

        mock_data = ((info, x, y), target_y, single_eval_pos)

        result = loader.ticl_to_ours(mock_data)

        self.assertIn("x", result)
        self.assertIn("y", result)
        self.assertIn("target_y", result)
        self.assertIn("single_eval_pos", result)

        # Verify shapes were transposed correctly
        self.assertEqual(result["x"].shape, (4, 100, 10))  # (batch, seq_len, features)
        self.assertEqual(result["y"].shape, (4, 100))  # (batch, seq_len)
        self.assertEqual(result["single_eval_pos"], single_eval_pos)


if __name__ == "__main__":
    unittest.main()
