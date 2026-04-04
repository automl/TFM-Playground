
"""
Real Data Prior - Episode Generator (Phase 3)

Purpose:
Generate synthetic in-context learning episodes from cached real-world tabular datasets.
Each episode consists of a randomly sampled dataset, a selected target column,
and a randomly sampled subset of rows formatted for few-shot learning.

This phase consumes:
- Phase 1 cache (.npz datasets + metadata)
- Phase 2 pool files (train / fallback dataset lists)

What this module does:
- Loads cached datasets lazily with LRU caching
- Selects datasets from a predefined pool
- Selects target columns (original or random depending on mode)
- Samples rows to construct an episode sequence
- Splits episode into context and query via single_eval_pos
- Standardizes features using context-only statistics (no leakage)
- Formats targets for classification or regression
- Caps number of classes to model constraints
- Pads or subsamples features to fixed dimensionality
- Returns torch tensors ready for training

Supported modes:
    classification_only
        Uses each dataset’s original classification target.

    regression_only
        Uses each dataset’s original regression target.

    mixed_random_target
        Randomly selects a column matching the requested task type.
        Optionally falls back to a curated fallback pool if needed.

Example usage:
    loader = RealDataPriorDataLoader(
        cache_dir="data/cache",
        train_pool_file="data/pools/train.txt",
        num_steps=1000,
        min_seq_len=64,
        max_seq_len=256,
        max_features=100,
        device=torch.device("cuda"),
        task_type="classification",
        mode="mixed"
    )

Output per iteration:
    {
        "x": Tensor [batch, seq_len, max_features],
        "y": Tensor [batch, seq_len],
        "target_y": Tensor [batch, seq_len],
        "single_eval_pos": int
    }
"""

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

# model constraint: maximum number of classes the model can handle
MODEL_MAX_CLASSES = 10

@dataclass
class EpisodeConfig:
    cache_dir: str
    train_pool_file: str
    num_steps: int  # num steps per epoch
    task_mode: str  # "classification_only", "regression_only", or "mixed_random_target"

    # parser overrides these values based on other args, but we need them in the config for episode generation logic
    min_seq_len: int = 64
    max_seq_len: int = 512
    max_features: int = 100
    max_classes: int = 10
    base_seed: int = 0

    # single eval pos is choosen to be between these two fractions
    min_eval_pos_frac: float = 0.25
    max_eval_pos_frac: float = 0.75

    # clip features after standardization to avoid extreme values that can destabilize training
    clip_features: bool = True
    clip_value: float = 10.0

    # heuristic for classification column detection in mixed_random_target mode: if a column is categorical or has few unique values, treat as classification
    unique_count_threshold: int = 10
    npz_cache_size: int = 16

    # retry limits for mixed_random_target mode
    max_dataset_retries: int = 3
    max_resample_retries: int = 3

    # for mixed_random_target: which task type to force ("classification" or "regression")
    mixed_target_type: Optional[str] = None

    # optional fallback pool file for mixed_random_target mode
    # must be explicitly provided, otherwise try a couple times and give error when target column not found
    fallback_pool_file: Optional[str] = None
    _VALID_TASK_MODES: ClassVar[set[str]] = {"classification_only", "regression_only", "mixed_random_target"}

class LRUCache:
    """
    small cache for loaded arrays.
    stores (data, is_categorical, unique_counts, original_target_idx) everything is keyed by dataset_id.
    basically an OrderedDict with a max size that pops the least recently used item when full.
    """

    def __init__(self, maxsize: int = 16):
        self.maxsize = maxsize
        self.cache: OrderedDict[str, tuple[np.ndarray, np.ndarray, np.ndarray, int]] = OrderedDict()

    def get(self, key: str) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, int]]:
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key: str, value: tuple[np.ndarray, np.ndarray, np.ndarray, int]) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
            self.cache[key] = value
            return
        if len(self.cache) >= self.maxsize:
            self.cache.popitem(last=False)
        self.cache[key] = value


class RealDataPrior(Dataset):
    """
    Torch Dataset that returns one synthetic in context episode per index.

    """

    def __init__(self, config: EpisodeConfig):
        self.config = config
        self.pool_ids = self._load_pool_ids()
        if not self.pool_ids:
            raise ValueError(f"No valid datasets found in pool file: {config.train_pool_file}")
        
        # load fallback pool only if in mixed_random_target mode
        if config.task_mode == "mixed_random_target":
            self.fallback_pool_ids = self._load_fallback_pool_ids()
        else:
            self.fallback_pool_ids = []
        self.npz_cache = LRUCache(maxsize=config.npz_cache_size)

    def _npz_path(self, dataset_id: str) -> Path:
        return Path(self.config.cache_dir) / "datasets" / f"openml_{dataset_id}.npz"

    def _load_pool_ids(self) -> list[str]:
        """load dataset IDs from pool file, and filter to those that have cached npz files. returns list of valid dataset IDs."""
        pool_path = Path(self.config.train_pool_file)
        if not pool_path.exists():
            raise FileNotFoundError(f"Pool file not found: {pool_path}")

        with open(pool_path) as f:
            ids = [line.strip() for line in f if line.strip()]

        valid_ids: list[str] = []
        for did in ids:
            if self._npz_path(did).exists():
                valid_ids.append(did)

        return valid_ids

    def _load_fallback_pool_ids(self) -> list[str]:
        """
        load fallback pool IDs from explicitly configured file.
        only called when in mixed_random_target mode.
        returns empty list if fallback_pool_file not set or doesn't exist.
        """
        cfg = self.config
        
        # must be explicitly set
        if cfg.fallback_pool_file is None:
            return []
        
        fallback_path = Path(cfg.fallback_pool_file)
        if not fallback_path.exists():
            return []
        
        with open(fallback_path) as f:
            ids = [line.strip() for line in f if line.strip()]
        
        valid_ids: list[str] = []
        for did in ids:
            if self._npz_path(did).exists():
                valid_ids.append(did)
        
        return valid_ids

    def _load_dataset(self, dataset_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        # check cache first
        cached = self.npz_cache.get(dataset_id)
        if cached is not None:
            return cached
        
        # load if not in cache
        npz_path = self._npz_path(dataset_id)
        with np.load(npz_path) as f:
            data = f["data"]  # flat matrix [n_rows, n_cols]
            is_categorical = f["column_is_categorical"].astype(bool)
            unique_counts = f["column_unique_counts"].astype(int)
            original_target_idx = int(f["original_target_idx"])

        out = (data, is_categorical, unique_counts, original_target_idx)
        self.npz_cache.put(dataset_id, out)
        return out
    
    def _stable_hash_seed(self, *parts: object) -> int:
        "ensures the same seed is generated for the same episode index and dataset_id across runs, regardless of other config changes. includes base_seed for global seed control."
        s = "|".join([str(self.config.base_seed)] + [str(p) for p in parts])
        h = hashlib.sha1(s.encode("utf8")).hexdigest()
        return int(h[:8], 16)

    def _get_episode_rng(self, idx: int, dataset_id: str) -> np.random.Generator:
        # generate a stable RNG for this episode based on index and dataset_id
        seed = self._stable_hash_seed("episode", idx, "dataset", dataset_id)
        return np.random.default_rng(seed)

    def _sample_dataset_id(self, rng: np.random.Generator) -> str:
        # sample a random dataset ID from the pool using the provided RNG
        j = int(rng.integers(0, len(self.pool_ids)))
        return self.pool_ids[j]

    def _pick_single_eval_pos(self, rng: np.random.Generator, seq_len: int) -> int:
        # pick a single evaluation position in the episode, between min_eval_pos_frac and max_eval_pos_frac of the sequence length, using the provided RNG.
        cfg = self.config
        if seq_len <= 2:
            return 1
        min_eval = max(1, int(seq_len * cfg.min_eval_pos_frac))
        max_eval = min(seq_len - 1, int(seq_len * cfg.max_eval_pos_frac))
        if min_eval >= max_eval:
            return seq_len // 2
        return int(rng.integers(min_eval, max_eval + 1))

    def _standardize_in_place(self, X_ep: np.ndarray, single_eval_pos: int) -> np.ndarray:
        # sanitize non-finite values before computing statistics to avoid reduction overflow warnings
        np.nan_to_num(X_ep, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # take the 'context' portion of the episode up to single_eval_pos
        train_X = X_ep[:single_eval_pos]
        # compute mean and std, and standardize the entire episode using those statistics for no leakage
        train_X64 = train_X.astype(np.float64, copy=False)
        means = train_X64.mean(axis=0, keepdims=True)
        stds = train_X64.std(axis=0, keepdims=True)
        # avoid division by zero for constant features by setting std to 1.0 in those cases
        stds = np.where(stds == 0.0, 1.0, stds)
        # standardize
        X_ep = (X_ep.astype(np.float64, copy=False) - means) / stds
        # safety guard/could be redundant
        X_ep = np.nan_to_num(X_ep, nan=0.0, posinf=0.0, neginf=0.0)

        # optional clipping to avoid extreme values that can destabilize training 
        if self.config.clip_features:
            X_ep = np.clip(X_ep, -self.config.clip_value, self.config.clip_value)

        return X_ep.astype(np.float32, copy=False)

    def _format_target(self, y_ep: np.ndarray, single_eval_pos: int, is_classification: bool) -> np.ndarray:
        """formats y_ep according to task type. 
        for classification, remaps classes to 0..C-1 and returns int64. 
        for regression, standardizes using train portion stats and returns float32."""
        
        if is_classification:
            y_ep = y_ep.astype(np.int64)
            
            classes = np.unique(y_ep)
            # remap to 0..max_classes - 1
            classes_sorted = np.sort(classes)
            class_to_idx = {c: i for i, c in enumerate(classes_sorted)}
            # replace with the new remapped values
            y_idx = np.array([class_to_idx[v] for v in y_ep], dtype=np.int64)
            # no label permutation needed: each episode picks its own dataset + target column and remaps classes independently
            return y_idx.astype(np.float32)

        # regression
        # rescale using mean and std of the 'context' portion to avoid leakage
        y_ep = np.nan_to_num(y_ep, nan=0.0, posinf=0.0, neginf=0.0)
        train_y = y_ep[:single_eval_pos]
        train_y64 = train_y.astype(np.float64, copy=False)
        y_mean = float(train_y64.mean())
        y_std = float(train_y64.std())
        if y_std == 0.0:
            y_std = 1.0
        y_scaled = (y_ep.astype(np.float64, copy=False) - y_mean) / y_std
        y_scaled = np.nan_to_num(y_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        return y_scaled.astype(np.float32)

    def _subsample_features(self, rng: np.random.Generator, X_ep: np.ndarray) -> np.ndarray:
        """if there are more features than max_features, randomly subsample features using the provided RNG.
        if fewer, return as-is (padding is handled downstream by dump_prior_to_h5)."""
        cfg = self.config
        f = X_ep.shape[1]
        if f > cfg.max_features:
            idx = rng.choice(f, size=cfg.max_features, replace=False)
            X_ep = X_ep[:, idx]
        return X_ep

    def _validate_task_mode(self) -> None:
        cfg = self.config
        if cfg.task_mode not in cfg._VALID_TASK_MODES:
            raise ValueError(
                f"Unknown task_mode '{cfg.task_mode}'. "
                f"Must be one of {sorted(cfg._VALID_TASK_MODES)}"
            )
        if cfg.task_mode == "mixed_random_target" and cfg.mixed_target_type not in ("classification", "regression"):
            raise ValueError(
                "mixed_random_target requires mixed_target_type to be 'classification' or 'regression'."
            )

    def _pick_target_col_mixed(self, rng: np.random.Generator, is_categorical: np.ndarray, unique_counts: np.ndarray, n_cols: int) -> tuple[Optional[int], bool]:
        """select a target column for mixed_random_target mode.
        iterates over all columns in random order and returns the first column
        that matches the requested task type (classification or regression)."""
        cfg = self.config
        classification_wanted = (cfg.mixed_target_type == "classification")

        for candidate in rng.permutation(n_cols):
            c = int(candidate)
            # classification if categorical or has few unique values, regression otherwise
            candidate_is_cls = bool(is_categorical[c]) or (int(unique_counts[c]) <= cfg.unique_count_threshold)
            if candidate_is_cls == classification_wanted:
                return c, candidate_is_cls

        return None, classification_wanted

    def _select_dataset_and_target(self, idx: int, initial_rng: np.random.Generator) -> tuple[np.random.Generator, np.ndarray, int, bool]:
        """
        selects a dataset and a valid target column for one episode.
        in single-target modes, the dataset’s original target column is used.
        in "mixed_random_target" mode, randomly searches columns within sampled datasets until it finds one matching the requested task type
        or retries up to `max_dataset_retries` datasets from the main pool

        if no suitable column is found and a fallback pool is provided,
        selects a dataset from the fallback pool and uses its original target.
        on the 'assumption' that the fallback pool is curated to contain the desired target types.
        """
        cfg = self.config

        # sample a dataset id
        dataset_id = self._sample_dataset_id(initial_rng)
        target_col: Optional[int] = None
        is_classification: Optional[bool] = None

        for _ in range(cfg.max_dataset_retries):
            rng = self._get_episode_rng(idx, dataset_id)
            # load the dataset
            data, is_categorical, unique_counts, original_target_idx = self._load_dataset(dataset_id)

            if cfg.task_mode == "mixed_random_target":
                # randomly samples columns until it finds one that matches the requested task type, or exhausts the columns.
                # if no column found after max retries returns none and the task it was looking for
                target_col, is_cls = self._pick_target_col_mixed(rng, is_categorical, unique_counts, data.shape[1])
                if target_col is None:
                    # sample a new one retry
                    dataset_id = self._sample_dataset_id(initial_rng)
                    continue
                is_classification = is_cls
            else:
                # target column is the original target column
                target_col = int(original_target_idx)
                is_classification = (cfg.task_mode == "classification_only")

            return rng, data, int(target_col), bool(is_classification)

        # try fallback pool
        if self.fallback_pool_ids:
            # Note: fallback pool is assumed to have the necessary target column as its intended target
            fallback_idx = int(initial_rng.integers(0, len(self.fallback_pool_ids)))
            dataset_id = self.fallback_pool_ids[fallback_idx]
            rng = self._get_episode_rng(idx, dataset_id)
            data, is_categorical, unique_counts, original_target_idx = self._load_dataset(dataset_id)

            target_col = int(original_target_idx)
            is_classification = (cfg.mixed_target_type == "classification")
            return rng, data, target_col, bool(is_classification)

        raise RuntimeError(
            f"Could not find any {cfg.mixed_target_type} column after "
            f"{cfg.max_dataset_retries} dataset attempts. "
            "Dataset pool may not contain suitable columns for the requested task type."
        )

    def _sample_rows_and_build_xy(
        self,
        rng: np.random.Generator,
        data: np.ndarray,
        target_col: int,
        is_classification: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """ Builds one episode by randomly sampling rows.
        - Splits data into X and y.
        - Samples a sequence length and rows without replacement.
        - For classification, ensures at least two classes if possible.
        Returns (X_ep, y_ep).
        """
        cfg = self.config

        # extract target column from the 'full' matrix
        y_use = data[:, target_col]
        # use the rest as features
        X_use = np.delete(data, target_col, axis=1)
        n_rows = X_use.shape[0]

        # pick a random sequence length for this episode, between min_seq_len and max_seq_len
        # but capped at the number of available rows
        seq_len = int(rng.integers(cfg.min_seq_len, cfg.max_seq_len + 1))
        seq_len = min(seq_len, n_rows)

        # randomly sample the rows without replacement to build the episode
        rows = rng.choice(n_rows, size=seq_len, replace=False)

        # subsampled episode data
        X_ep = X_use[rows].astype(np.float32, copy=True)
        y_ep = y_use[rows].astype(np.int64 if is_classification else np.float32, copy=True)

        # safeguard against sampling only one class for classification episodes
        if is_classification:
            for _ in range(cfg.max_resample_retries):
                if np.unique(y_ep).size >= 2:
                    break
                rows = rng.choice(n_rows, size=seq_len, replace=False)
                X_ep = X_use[rows].astype(np.float32, copy=True)
                y_ep = y_use[rows].astype(np.int64, copy=True)

        return X_ep, y_ep

    def _cap_classes_in_place(self, y_ep: np.ndarray) -> None:
        """mutates y_ep to respect maximum classes using an 'other' bucket."""
        effective_max = min(self.config.max_classes, MODEL_MAX_CLASSES)
        classes, counts = np.unique(y_ep, return_counts=True)
        if classes.size <= effective_max:
            return

        top_k = classes[np.argsort(-counts)[: effective_max - 1]]
        mask = ~np.isin(y_ep, top_k)

        other = int(y_ep.max()) + 1
        y_ep[mask] = other

    def _sample_episode(self, idx: int, initial_rng: np.random.Generator) -> dict[str, torch.Tensor | int]:
        self._validate_task_mode()

        rng, data, target_col, is_classification = self._select_dataset_and_target(idx, initial_rng)

        X_ep, y_ep = self._sample_rows_and_build_xy(rng, data, target_col, is_classification)

        if is_classification:
            self._cap_classes_in_place(y_ep)

        single_eval_pos = self._pick_single_eval_pos(rng, X_ep.shape[0])

        X_ep = self._standardize_in_place(X_ep, single_eval_pos)
        X_ep = self._subsample_features(rng, X_ep)

        y_ep = self._format_target(y_ep, single_eval_pos, is_classification)

        x_tensor = torch.from_numpy(X_ep)
        y_tensor = torch.from_numpy(y_ep)

        return {
            "x": x_tensor,
            "y": y_tensor,
            "target_y": y_tensor.clone(),
            "single_eval_pos": int(single_eval_pos),
        }

    def __len__(self) -> int:
        return self.config.num_steps

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        seed = self._stable_hash_seed("initial", idx)
        initial_rng = np.random.default_rng(seed)
        return self._sample_episode(idx, initial_rng)

class RealDataPriorDataLoader:
    """
    Thin wrapper that yields dict batches on a device fitting the style of the synthetic data loaders
    """
    def __init__(
        self,
        cache_dir: str,
        train_pool_file: str,
        num_steps: int,
        min_seq_len: int,
        max_seq_len: int,
        max_features: int,
        device: torch.device,
        task_type: str,
        mode: str = "only",
        batch_size: int = 1, # required for model's loss function otherwise padding effects the loss calculation
        base_seed: int = 0,
        npz_cache_size: int = 16,
        num_workers: int = 0,
        fallback_pool_file: Optional[str] = None,
    ):
        # support 2 modes: randomly sampling the target column until it is not degenerate or 
        # sampling the original target column
        if mode == "only":
            internal_task_mode = f"{task_type}_only"
        else:  # mode == "mixed"
            internal_task_mode = "mixed_random_target"

        self.config = EpisodeConfig(
            cache_dir=cache_dir,
            train_pool_file=train_pool_file,
            num_steps=num_steps,
            min_seq_len=min_seq_len,
            max_seq_len=max_seq_len,
            max_features=max_features,
            task_mode=internal_task_mode,
            mixed_target_type=task_type,
            base_seed=base_seed,
            npz_cache_size=npz_cache_size,
            fallback_pool_file=fallback_pool_file,
        )

        self.device = device
        self.dataset = RealDataPrior(self.config)

        self._dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    def __iter__(self):
        for batch in self._dataloader:
            yield {
                "x": batch["x"].to(self.device),
                "y": batch["y"].to(self.device),
                "target_y": batch["target_y"].to(self.device),
                "single_eval_pos": batch["single_eval_pos"],
            }

    def __len__(self) -> int:
        return len(self._dataloader)
