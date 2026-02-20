"""Data loading utilities for tabular priors."""

from typing import Any, Callable, Dict, Iterator, Union
import warnings

import h5py
import numpy as np
import torch
from tabicl.prior.dataset import PriorDataset as TabICLPriorDataset
# from ticl.dataloader import PriorDataLoader as TICLPriorDataset
from torch.utils.data import DataLoader

from gtfm.utils.adj import remove_axis

class PriorDataLoader(DataLoader):
    """Generic DataLoader for synthetic data generation using a get_batch function.

    Args:
        get_batch_function (Callable): A function returning batches of data.
        num_steps (int): Number of batches per epoch.
        batch_size (int): Number of functions per batch.
        num_datapoints_max (int): Max sequence length per function.
        num_features (int): Number of input features.
        device (torch.device): Device to move tensors to.
    """

    def __init__(
        self,
        get_batch_function: Callable[..., Dict[str, Union[torch.Tensor, int]]],
        num_steps: int,
        batch_size: int,
        num_datapoints_max: int,
        num_features: int,
        device: torch.device,
    ):
        self.get_batch_function = get_batch_function
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.num_datapoints_max = num_datapoints_max
        self.num_features = num_features
        self.device = device

    def __iter__(self) -> Iterator[Dict[str, Union[torch.Tensor, int]]]:
        return iter(
            self.get_batch_function(self.batch_size, self.num_datapoints_max, self.num_features)
            for _ in range(self.num_steps)
        )

    def __len__(self) -> int:
        return self.num_steps


class PriorDumpDataLoader(DataLoader):
    """DataLoader that loads synthetic prior data from an HDF5 dump.

    Args:
        filename (str): Path to the HDF5 file.
        num_steps (int): Number of batches per epoch.
        batch_size (int): Batch size.
        device (torch.device): Device to load tensors onto.
        starting_index (int): Starting index for the data pointer.
        max_density (float, optional): Only load datasets with density <= this value.
        min_density (float, optional): Only load datasets with density >= this value.
        n_samples (int, optional): Number of samples to use from the valid density range.
                                   Samples are shuffled once at initialization to avoid bias.
        random_seed (int, optional): Random seed for reproducible sample shuffling.
        
    Note:
        The density filtering parameters (max_density, min_density, n_samples, random_seed)
        must be either all None or all set. Partial specification will raise a ValueError.
        Requires the HDF5 file to be sorted by density when using density filtering.
    """
    def __init__(
        self, 
        filename, 
        num_steps, 
        batch_size, 
        device, 
        starting_index=0, 
        max_density=None,
        min_density=None,
        n_samples=None,
        random_seed=None
    ):
        self.filename = filename
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.max_density = max_density
        self.min_density = min_density
        self.n_samples = n_samples
        self.random_seed = random_seed
        self.device = device

        # Validate density parameters
        self._validate_density_parameters()
        
        with h5py.File(self.filename, "r") as f:
            self.num_datapoints_max = f['X'].shape[0]
            
            # Apply density filtering
            if self.max_density is not None or self.min_density is not None:
                all_densities = f["density"][:]
                self.shuffled_indices = self._compute_shuffled_indices(all_densities)
                self._log_density_info(all_densities)
            else:
                # No filtering, use all samples
                if self.n_samples is not None:
                    if self.n_samples > self.num_datapoints_max:
                        raise ValueError(
                            f"n_samples ({self.n_samples}) cannot be greater than "
                            f"total available samples ({self.num_datapoints_max})"
                        )
                    # Randomly sample and shuffle n_samples from all available data
                    rng = np.random.RandomState(self.random_seed)
                    self.shuffled_indices = rng.choice(
                        self.num_datapoints_max, 
                        size=self.n_samples, 
                        replace=False
                    )
                    rng.shuffle(self.shuffled_indices)
                    print(f"Randomly selected and shuffled {self.n_samples} samples from {self.num_datapoints_max} total samples")
                else:
                    # Use all samples, optionally shuffle them
                    self.shuffled_indices = np.arange(self.num_datapoints_max)
                    if self.random_seed is not None:
                        rng = np.random.RandomState(self.random_seed)
                        rng.shuffle(self.shuffled_indices)

            if "max_num_classes" in f:
                self.max_num_classes = f["max_num_classes"][0]
            else:
                self.max_num_classes = None
            self.problem_type = f["problem_type"][()].decode("utf-8")
            self.has_num_datapoints = "num_datapoints" in f
            _, self.stored_max_seq_len, self.stored_max_num_features = f["X"].shape
        
        # Initialize pointer for iteration
        self.pointer = starting_index % len(self.shuffled_indices)
    
    def _validate_density_parameters(self):
        """Validate density filtering parameters."""
        # Check that density filtering parameters are used consistently
        density_params = [self.max_density, self.min_density, self.n_samples, self.random_seed]
        density_params_set = [p is not None for p in density_params]
        
        if any(density_params_set) and not all(density_params_set):
            param_names = ['max_density', 'min_density', 'n_samples', 'random_seed']
            set_params = [name for name, is_set in zip(param_names, density_params_set) if is_set]
            unset_params = [name for name, is_set in zip(param_names, density_params_set) if not is_set]
            raise ValueError(
                f"Density filtering parameters must be either all None or all set. "
                f"Currently set: {set_params}. Missing: {unset_params}."
            )
        
        # If all are None, no further validation needed
        if not any(density_params_set):
            return
        
        # At this point, all parameters are set, so validate their values
        if not (0 <= self.max_density <= 1):
            raise ValueError(f"max_density must be between 0 and 1, got {self.max_density}")
        
        if not (0 <= self.min_density <= 1):
            raise ValueError(f"min_density must be between 0 and 1, got {self.min_density}")
        
        if self.min_density > self.max_density:
            raise ValueError(
                f"min_density ({self.min_density}) cannot be greater than "
                f"max_density ({self.max_density})"
            )

    def _compute_shuffled_indices(self, all_densities):
        """Compute shuffled indices based on density filtering and n_samples.
        
        The indices are shuffled once at initialization to avoid bias toward
        lower densities while maintaining fast slicing access during iteration.
        
        Args:
            all_densities: Array of density values from the HDF5 file.
            
        Returns:
            Array of shuffled indices to use for sampling.
        """
        # Check if data is sorted
        if not np.all(all_densities[:-1] <= all_densities[1:]):
            raise ValueError(
                f"The HDF5 file '{self.filename}' is not sorted by density. "
                "The PriorDumpDataLoader with a density filter requires the data to be pre-sorted. "
                "Please regenerate the dataset using the sorted dump function."
            )

        # Find the range of valid indices based on density
        if self.min_density is not None:
            start_idx = int(np.searchsorted(all_densities, self.min_density, side='left'))
        else:
            start_idx = 0
        
        if self.max_density is not None:
            end_idx = int(np.searchsorted(all_densities, self.max_density, side='right'))
        else:
            end_idx = len(all_densities)
        
        # Validate the range
        if start_idx >= end_idx:
            raise ValueError(
                f"No datasets found with density in range [{self.min_density}, {self.max_density}]. "
                f"Available density range: [{all_densities.min():.4f}, {all_densities.max():.4f}]"
            )
        
        # Get all candidate indices in the density range
        candidate_indices = np.arange(start_idx, end_idx)
        num_candidates = len(candidate_indices)
        
        # Handle n_samples parameter
        if self.n_samples is not None:
            if self.n_samples > num_candidates:
                raise ValueError(
                    f"Insufficient samples after density filtering: "
                    f"found {num_candidates} samples, but n_samples={self.n_samples}. "
                    f"Density range: [{self.min_density}, {self.max_density}]. "
                    f"Consider relaxing the density constraints or reducing n_samples."
                )
            
            # Randomly sample n_samples from the valid range
            rng = np.random.RandomState(self.random_seed)
            shuffled_indices = rng.choice(
                candidate_indices, 
                size=self.n_samples, 
                replace=False
            )
            # Shuffle to avoid any remaining bias
            rng.shuffle(shuffled_indices)
        else:
            # Use all candidates and shuffle them
            shuffled_indices = candidate_indices.copy()
            rng = np.random.RandomState(self.random_seed)
            rng.shuffle(shuffled_indices)
        
        # Warn if we don't have enough samples for the requested batches
        total_samples_needed = self.num_steps * self.batch_size
        if len(shuffled_indices) < total_samples_needed:
            warnings.warn(
                f"Number of valid samples ({len(shuffled_indices)}) is less than "
                f"total samples needed ({total_samples_needed} = {self.num_steps} steps × "
                f"{self.batch_size} batch_size). Data will be reused multiple times per epoch.",
                UserWarning
            )
        
        return shuffled_indices

    def _log_density_info(self, all_densities):
        """Log information about the density filtering."""
        num_valid_samples = len(self.shuffled_indices)
        # Get density range from the shuffled indices
        densities_selected = all_densities[self.shuffled_indices]
        min_density_actual = densities_selected.min()
        max_density_actual = densities_selected.max()
        
        if self.n_samples is not None:
            print(
                f"Density filter active: Randomly selected and shuffled {num_valid_samples} samples "
                f"from density range [{min_density_actual:.4f}, {max_density_actual:.4f}]"
            )
        else:
            print(
                f"Density filter active: Using {num_valid_samples} shuffled samples "
                f"with density range [{min_density_actual:.4f}, {max_density_actual:.4f}]"
            )

    def __iter__(self):
        with h5py.File(self.filename, "r") as f:
            for _ in range(self.num_steps):
                
                if self.pointer + self.batch_size > len(self.shuffled_indices):
                    # Wrap around to the start
                    print(
                        """Finished iteration over valid samples! """
                        """Will start reusing the same data with different splits now."""
                    )
                    self.pointer = 0

                # Get the actual indices for this batch
                batch_indices = self.shuffled_indices[self.pointer : self.pointer + self.batch_size]

                # Load data using fancy indexing (unavoidable with shuffled indices)
                num_features = f["num_features"][batch_indices].max()
                if self.has_num_datapoints:
                    num_datapoints_batch = f["num_datapoints"][batch_indices]
                    max_seq_in_batch = int(num_datapoints_batch.max())
                else:
                    max_seq_in_batch = int(self.stored_max_seq_len)

                x = torch.from_numpy(f["X"][batch_indices, :max_seq_in_batch, :num_features])
                y = torch.from_numpy(f["y"][batch_indices, :max_seq_in_batch])
                adj = torch.from_numpy(f['adj'][batch_indices,])

                if num_features != self.stored_max_num_features:
                    # We cut down the padded features to the max number of features in **the current** batch. 
                    # Therefore, we need to cut down the adjacency matrix accordingly. 
                    # The features in adj are stored as (features | padded features | target node).
                    adj = remove_axis(adj, list(range(num_features, adj.shape[1] - 1)))

                single_eval_pos = f["single_eval_pos"][batch_indices]

                self.pointer += self.batch_size

                yield dict(
                    x=x.to(self.device),
                    y=y.to(self.device),
                    target_y=y.to(self.device),  # target_y is identical to y (for downstream compatibility)
                    single_eval_pos=single_eval_pos[0].item(),
                    adj=adj.to(self.device),
                )

    def __len__(self):
        return self.num_steps


class TabICLPriorDataLoader(DataLoader):
    """DataLoader sampling synthetic prior data on-the-fly from TabICL's PriorDataset.

    Args:
        num_steps (int): Number of batches to generate per epoch.
        batch_size (int): Number of functions per batch.
        num_datapoints_min (int): Minimum number of datapoints per function.
        num_datapoints_max (int): Maximum number of datapoints per function.
        min_features (int): Minimum number of features in x.
        max_features (int): Maximum number of features in x.
        max_num_classes (int): Maximum number of classes (for classification tasks).
        device (torch.device): Target device for tensors.
    """

    def __init__(
        self,
        num_steps: int,
        batch_size: int,
        num_datapoints_min: int,
        num_datapoints_max: int,
        min_features: int,
        max_features: int,
        max_num_classes: int,
        device: torch.device,
        scm_fixed_hp: Dict[str, Any],
        scm_sampled_hp: Dict[str, Any],
    ):
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.num_datapoints_min = num_datapoints_min
        self.num_datapoints_max = num_datapoints_max
        self.min_features = min_features
        self.max_features = max_features
        self.max_num_classes = max_num_classes
        self.device = device

        self.pd = TabICLPriorDataset(
            batch_size=batch_size,
            batch_size_per_gp=batch_size,
            min_features=min_features,
            max_features=max_features,
            max_classes=max_num_classes,
            min_seq_len=num_datapoints_min,
            max_seq_len=num_datapoints_max,
            scm_fixed_hp=scm_fixed_hp,
            scm_sampled_hp=scm_sampled_hp,
            # device=device,
        )

    def tabicl_to_ours(self, d):
        x, y, active_features, seqlen, train_size, adj, priors = d
        density = torch.tensor([p.density for p in priors])
        if (active_features != active_features[0]).any():
            print("Warning: Varying active features within a batch is not supported. ")
            return None # skip batches with varying active features for now
        active_features = active_features[
            0
        ].item()  # should be all the same since we use batch_size_per_gp=batch_size (not true in practice!)
        x = x[:, :, :active_features]
        if (train_size != train_size[0]).any():
            print("Warning: Varying train sizes within a batch is not supported. ")
            return None # skip batches with varying train sizes for now
        single_eval_pos = train_size[0].item()  # should be all the same since we use batch_size_per_gp=batch_size
        return dict(
            x=x.to(self.device),
            y=y.to(self.device),
            target_y=y.to(self.device),  # target_y is identical to y (for downstream compatibility)
            single_eval_pos=single_eval_pos,
            adj=adj.to(self.device),
            density=density.to(self.device),
            priors=priors,
        )

    def __iter__(self):
        # Quick ugly fix to avoid None batches, which come from varying active_features/train_size in TabICL's PriorDataset
        # don't understand why that happens when batch_size_per_gp == batch_size
        # return iter(self.tabicl_to_ours(next(self.pd)) for _ in range(self.num_steps))
        generator  = (self.tabicl_to_ours(next(self.pd)) for _ in range(self.num_steps))
        return (batch for batch in generator if batch is not None)

    def __len__(self):
        return self.num_steps


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
        device: torch.device,
        min_eval_pos: int = 10,
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
        (info, x, y), target_y, single_eval_pos = d
        x = x.permute(1, 0, 2)
        y = y.permute(1, 0)
        target_y = target_y.permute(1, 0)

        return dict(
            x=x.to(self.device),
            y=y.to(self.device),
            target_y=target_y.to(self.device),  # target_y is identical to y (for downstream compatibility)
            single_eval_pos=single_eval_pos,
        )

    def __iter__(self):
        return (self.ticl_to_ours(batch) for batch in self.pd)

    def __len__(self):
        return self.num_steps