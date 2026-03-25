"""Data loading utilities for tabular priors."""

import os
from typing import Any, Callable, Dict, Iterator, Union

import h5py
import torch
from tabicl.prior.dataset import PriorDataset as TabICLPriorDataset
# from ticl.dataloader import PriorDataLoader as TICLPriorDataset
from gcfm.priordata_processing.Datasets.ObservationalDataset import ObservationalDataset as GCFMPriorDataset
from torch.utils.data import DataLoader
import networkx as nx

from gtfm.graph.torch_moral import calculate_density
from gtfm.utils.adj import move_axis, remove_axis

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
    """
    def __init__(self, filename, num_steps, batch_size, device, starting_index=0):
        self.filename = filename
        self.num_steps = num_steps
        self.batch_size = batch_size
        with h5py.File(self.filename, "r", swmr=True) as f:
            self.num_datapoints_max = f['X'].shape[0]
            if "max_num_classes" in f:
                self.max_num_classes = f["max_num_classes"][0]
            else:
                self.max_num_classes = None
            self.problem_type = f["problem_type"][()].decode("utf-8")
            self.has_num_datapoints = "num_datapoints" in f
            _, self.stored_max_seq_len, self.stored_max_num_features = f["X"].shape
        self.device = device
        self.pointer = starting_index

    def __iter__(self):
        with h5py.File(self.filename, "r", swmr=True) as f:
            for _ in range(self.num_steps):
                end = self.pointer + self.batch_size

                num_features = f["num_features"][self.pointer : end].max()
                if self.has_num_datapoints:
                    num_datapoints_batch = f["num_datapoints"][self.pointer:end]
                    max_seq_in_batch = int(num_datapoints_batch.max())
                else:
                    max_seq_in_batch = int(self.stored_max_seq_len)

                x = torch.from_numpy(f["X"][self.pointer:end, :max_seq_in_batch, :num_features])
                y = torch.from_numpy(f["y"][self.pointer:end, :max_seq_in_batch])
                # adj = torch.from_numpy(f['adj'][self.pointer:end,])

                # if num_features != self.stored_max_num_features:
                    # We cut down the padded features to the max number of features in **the current** batch. Therefore, we need to cut down the adjacency matrix accordingly. The features in adj are stored as (features | padded features | target node).
                    # adj = remove_axis(adj, list(range(num_features, adj.shape[1] - 1)))

                single_eval_pos = f["single_eval_pos"][self.pointer : end]

                self.pointer += self.batch_size
                if self.pointer >= f["X"].shape[0]:
                    print(
                        """Finished iteration over all stored datasets! """
                        """Will start reusing the same data with different splits now."""
                    )
                    self.pointer = 0

                yield dict(
                    x=x.to(self.device),
                    y=y.to(self.device),
                    target_y=y.to(self.device),  # target_y is identical to y (for downstream compatibility)
                    single_eval_pos=single_eval_pos[0].item(),
                    # adj=adj.to(self.device),
                    adj = None,
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

class GCFMDataLoader(DataLoader):
    """DataLoader sampling synthetic prior data from GCFM's PriorDataLoader.

    Each iteration of GCFM's PriorDataset yields a single sample, so this
    DataLoader draws `batch_size` individual samples and stacks them into
    a batched dict per step.

    Args:
        config (Dict[str, Any]): GCFM configuration containing 'scm_config',
            'preprocessing_config', and 'dataset_config'.
        batch_size (int): Number of functions sampled per batch.
        num_steps (int): Number of batches per epoch.
        device (torch.device): Target device for tensors.
    """

    # STACK_KEYS = {"x", "y", "adj", "density"}
    STACK_KEYS = {"x", "y"}

    def __init__(
        self,
        config: Dict[str, Any],
        batch_size: int,
        num_steps: int,
        device: torch.device,
        extra_checks: bool = False,
    ):
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.device = device
        self._global_idx = 0
        self.extra_checks = extra_checks

        self.pd = GCFMPriorDataset(
            scm_config=config["scm_config"],
            preprocessing_config=config["preprocessing_config"],
            dataset_config=config["dataset_config"],
            seed=None,
        )

    def gcfm_to_ours(self, sample):
        """Convert a single GCFM sample to our dict format."""
        x_train, y_train, x_test, y_test, graph_info, dataset_info = sample

        x = torch.cat([x_train, x_test], dim=0)
        y = torch.cat([y_train, y_test], dim=0).squeeze(-1)
        # adj = graph_info["moral_matrix_padded"] 
        # density = graph_info["moral_density"]
        
        scm = graph_info["scm"]
        processor = graph_info["processor"]

        # if self.extra_checks:
        #     ordered_nodes: list[int] = processor.kept_feature_indices + [processor.selected_target_feature]
        #     p = len(ordered_nodes)

        #     adj1 = move_axis(adj, src=adj.shape[0]-1, dst = processor.selected_target_feature)[:p, :p]
        #     adj2 = nx.adjacency_matrix(nx.moral_graph(scm.dag.g), nodelist=ordered_nodes).todense() 

        return dict(
            x=x,
            y=y,
            # adj=adj,
            # density=density,
            single_eval_pos=dataset_info["number_train_samples"],
            scm=scm,
            processor=processor,
        )

    def _collate(self, dicts):
        """Stack a list of single-sample dicts into a batched dict."""
        single_eval_positions = [d["single_eval_pos"] for d in dicts]
        if len(set(single_eval_positions)) > 1:
            raise ValueError("Varying train sizes within a batch is not supported.")

        batch = {
            k: torch.stack([d[k] for d in dicts]).to(self.device)
            for k in self.STACK_KEYS
        }
        batch["target_y"] = batch["y"]  # downstream compatibility
        batch["single_eval_pos"] = single_eval_positions[0]
        batch["scm"] = [d["scm"] for d in dicts]
        batch["processor"] = [d["processor"] for d in dicts]

        return batch

    def __iter__(self):
        def generate():
            for _ in range(self.num_steps):
                samples = [
                    self.gcfm_to_ours(self.pd[self._global_idx + i])
                    for i in range(self.batch_size)
                ]
                self._global_idx += self.batch_size
                batch = self._collate(samples)
                if batch is not None:
                    yield batch

        return generate()

    def __len__(self):
        return self.num_steps


# class GCFMDataLoader(DataLoader):
#     """DataLoader sampling synthetic prior data from GCFM's PriorDataLoader.

#     Args:
#         prior (Any): A GCFM prior object supporting get_batch.
#         num_steps (int): Number of batches per epoch.
#         batch_size (int): Number of functions sampled per batch.
#         num_datapoints_max (int): Number of datapoints sampled per function.
#         num_features (int): Dimensionality of x vectors.
#         device (torch.device): Target device for tensors.
#     """

#     def __init__(
#         self,
#         config: Dict[str, Any],
#         batch_size: int,
#         num_steps: int,
#         device: torch.device,
#     ):
#         self.batch_size = batch_size
#         self.num_steps = num_steps
#         self.device = device

#         self.pd = GCFMPriorDataset(
#             scm_config = config['scm_config'],
#             preprocessing_config = config['preprocessing_config'],
#             dataset_config = config['dataset_config'],
#             seed = None,
#         )

#     def gcfm_to_ours(self, d):
#         x_train, y_train, x_test, y_test, graph_info, dataset_info = d

#         x = torch.cat([x_train, x_test], dim=0)
#         y = torch.cat([y_train, y_test], dim=0)
#         single_eval_pos = dataset_info['number_train_samples']
#         adj = graph_info['moral_matrix'] # select moral_matrix instead of adj_matrix
#         density = calculate_density(adj)

#         return dict(
#             x=x.to(self.device),
#             y=y.to(self.device),
#             target_y=y.to(self.device),  # target_y is identical to y (for downstream compatibility)
#             single_eval_pos=single_eval_pos,
#             adj=adj.to(self.device),
#             density=density.to(self.device),
#         )
    


#     def __iter__(self):
#         return iter(self.gcfm_to_ours(next(self.pd)) for _ in range(self.num_steps))

#     def __len__(self):
#         return self.num_steps