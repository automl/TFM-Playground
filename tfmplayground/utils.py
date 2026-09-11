import math
import os
import random
import uuid
from dataclasses import asdict
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np
import torch
from torch import nn

from tfmplayground import models
from tfmplayground.configs import models as model_configs
from tfmplayground.configs.training import ExperimentConfig
from tfmplayground.models.base import TabularFoundationModel

if TYPE_CHECKING:
    from tfmplayground.priors.base import Prior


def set_randomness_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_default_device():
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    if torch.cuda.is_available():
        device = "cuda"
    return device


def compute_bucket_borders(num_buckets: int, ys: torch.Tensor) -> torch.Tensor:
    """
    decides equal mass bucket borders from ys

    inspired by pfns.model.bar_distribution get_bucket_borders
    """
    ys = torch.as_tensor(ys, dtype=torch.float32).flatten()
    ys = ys[torch.isfinite(ys)]

    if ys.numel() < num_buckets:
        raise ValueError(f"{ys.numel()} targets cannot make {num_buckets} buckets")

    n = (ys.numel() // num_buckets) * num_buckets
    ys = ys[:n]
    ys_per_bucket = n // num_buckets

    ys_sorted, _ = torch.sort(ys)

    chunks = ys_sorted.reshape(num_buckets, ys_per_bucket)
    interiors = (chunks[:-1, -1] + chunks[1:, 0]) / 2

    min_outer = ys_sorted[0].unsqueeze(0)
    max_outer = ys_sorted[-1].unsqueeze(0)

    borders = torch.cat([min_outer, interiors, max_outer])

    if borders.numel() != num_buckets + 1:
        raise ValueError(f"{borders.numel()} borders cannot make {num_buckets} buckets")
    if borders.numel() != torch.unique_consecutive(borders).numel():
        raise ValueError("targets repeat, at least one bucket has no width")

    return borders


def make_bucket_borders(
    prior: "Prior",
    num_buckets: int,
    batch_size: int,
    min_targets: int,
) -> torch.Tensor:
    """
    finds bucket borders from targets that prior gives
    """
    normalized_targets = []
    collected = 0
    while collected < min_targets:
        _, y_train, _, y_test = prior.batch(batch_size)
        y_train = y_train.detach().to("cpu", torch.float32)
        y_test = y_test.detach().to("cpu", torch.float32)
        y_mean = y_train.mean(dim=1, keepdim=True)
        y_std = y_train.std(dim=1, keepdim=True) + 1e-8
        y = torch.cat([y_train, y_test], dim=1)
        normalized = ((y - y_mean) / y_std).ravel()
        if normalized.numel() == 0:
            raise ValueError("prior gave no targets")
        normalized_targets.append(normalized)
        collected += normalized.numel()

    ys = torch.cat(normalized_targets)
    ys = ys[torch.isfinite(ys)]
    if ys.numel() < num_buckets:
        raise ValueError(f"{ys.numel()} targets cannot make {num_buckets} buckets")

    return compute_bucket_borders(num_buckets, ys)


def make_global_bucket_edges(filename, n_buckets=100, device=None, max_y=5_000_000):
    if device is None:
        device = get_default_device()
    with h5py.File(filename, "r") as f:
        y = f["y"]
        num_tables, num_datapoints = y.shape

        num_tables_to_use = min(num_tables, max_y // num_datapoints)

        y_subset = np.array(y[:num_tables_to_use, :], dtype=np.float32)
        y_means = y_subset.mean(axis=1, keepdims=True)
        y_stds = y_subset.std(axis=1, keepdims=True, ddof=1) + 1e-8
        ys_concat = ((y_subset - y_means) / y_stds).ravel()

    if ys_concat.size < n_buckets:
        raise ValueError(f"Too few target samples ({ys_concat.size}) to compute {n_buckets} buckets.")

    ys_tensor = torch.tensor(ys_concat, dtype=torch.float32, device=device)
    global_bucket_edges = compute_bucket_borders(n_buckets, ys=ys_tensor).to(device)
    return global_bucket_edges


class BarDistribution(nn.Module):
    """
    bar distribution defined by borders with nan target ignoring option

    inspired by pfns.model.bar_distribution BarDistribution
    """

    def __init__(
        self,
        borders: torch.Tensor,
        *,
        ignore_nan_targets: bool = True,
    ) -> None:
        """
        makes sure borders can make bars that have width
        """
        super().__init__()
        borders = torch.as_tensor(borders)
        if borders.ndim != 1:
            raise ValueError(f"borders must have 1 dimension, not {borders.ndim}")
        if not torch.is_floating_point(borders):
            borders = borders.to(torch.get_default_dtype())
        borders = borders.contiguous()
        self.register_buffer("borders", borders)
        if not (self.bar_widths > 0).all():
            raise ValueError("borders do not increase, at least one bar has no width")
        self.ignore_nan_targets = ignore_nan_targets

    @property
    def bar_widths(self) -> torch.Tensor:
        """
        gives width of every bar
        """
        return self.borders[1:] - self.borders[:-1]

    @property
    def num_bars(self) -> int:
        """
        gives how many bars borders make
        """
        return self.borders.numel() - 1

    def _ignore_init(self, y: torch.Tensor) -> torch.Tensor:
        """
        makes ignore mask for nan targets and alters y (will be ignored later)
        """
        ignore_mask = torch.isnan(y)
        if ignore_mask.any():
            if not self.ignore_nan_targets:
                raise ValueError("targets contain nan which this distribution does not ignore")
            y[ignore_mask] = self.borders[0]
        return ignore_mask

    def _map_to_bar_indices(self, y: torch.Tensor) -> torch.Tensor:
        """
        maps each y to its corresponding bar index
        """
        indices = torch.searchsorted(self.borders, y) - 1
        indices = indices.clamp(0, self.num_bars - 1)
        return indices

    def _compute_scaled_log_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """
        log prob density
        """
        widths = self.bar_widths.to(device=logits.device, dtype=logits.dtype)
        log_probs = torch.log_softmax(logits, dim=-1)
        log_widths = torch.log(widths)
        scaled_log_probs = log_probs - log_widths
        return scaled_log_probs


class FullSupportBarDistribution(BarDistribution):
    """
    extends BarDistribution with half normal tails on both sides for full support

    inspired by pfns.model.bar_distribution FullSupportBarDistribution
    """

    def __init__(
        self,
        borders: torch.Tensor,
        *,
        ignore_nan_targets: bool = True,
    ) -> None:
        """
        builds bar distribution
        """
        super().__init__(borders, ignore_nan_targets=ignore_nan_targets)

    @staticmethod
    def _halfnormal_with_p_weight_before(
        desired_quantile_value_at_p: torch.Tensor,
        p: float = 0.5,
    ) -> torch.distributions.HalfNormal:
        """
        scales the half normal distribution so that the p weight is before the desired value
        """
        device = desired_quantile_value_at_p.device
        dtype = desired_quantile_value_at_p.dtype
        standard_halfnormal = torch.distributions.HalfNormal(torch.tensor(1.0, device=device, dtype=dtype))
        quantile_value_at_p = standard_halfnormal.icdf(torch.tensor(p, device=device, dtype=dtype))
        scale = desired_quantile_value_at_p / quantile_value_at_p
        scaled_halfnormal = torch.distributions.HalfNormal(scale)
        return scaled_halfnormal

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        negative log likelihood of y given logits
        """
        if logits.shape[-1] != self.num_bars:
            raise ValueError(f"{logits.shape[-1]} logits cannot fill {self.num_bars} bars")

        device = logits.device
        dtype = logits.dtype

        y = torch.as_tensor(y, device=device, dtype=dtype).clone().reshape(*logits.shape[:-1])
        ignore_mask = self._ignore_init(y)  # alters y
        y_bar_indices = self._map_to_bar_indices(y)
        scaled_log_probs = self._compute_scaled_log_probs(logits)
        gathered_scaled_log_probs = scaled_log_probs.gather(-1, y_bar_indices.unsqueeze(-1)).squeeze(-1)

        bar_widths = self.bar_widths.to(device=device, dtype=dtype)
        borders = self.borders.to(device=device, dtype=dtype)
        left_tail = self._halfnormal_with_p_weight_before(bar_widths[0])
        right_tail = self._halfnormal_with_p_weight_before(bar_widths[-1])

        left_mask = y_bar_indices == 0
        if left_mask.any():
            distances = (borders[1] - y[left_mask]).clamp(min=1e-8)
            gathered_scaled_log_probs[left_mask] += left_tail.log_prob(distances) + torch.log(bar_widths[0])

        right_mask = y_bar_indices == self.num_bars - 1
        if right_mask.any():
            distances = (y[right_mask] - borders[-2]).clamp(min=1e-8)
            gathered_scaled_log_probs[right_mask] += right_tail.log_prob(distances) + torch.log(bar_widths[-1])

        nll = -gathered_scaled_log_probs
        nll[ignore_mask] = 0.0
        return nll

    def mean(self, logits: torch.Tensor) -> torch.Tensor:
        """
        calculates the expected value of the distribution given logits
        """
        if logits.shape[-1] != self.num_bars:
            raise ValueError(f"{logits.shape[-1]} logits cannot fill {self.num_bars} bars")

        device = logits.device
        dtype = logits.dtype

        probs = torch.softmax(logits.to(torch.float32), dim=-1).to(dtype)

        bar_widths = self.bar_widths.to(device=device, dtype=dtype)
        borders = self.borders.to(device=device, dtype=dtype)
        left_tail = self._halfnormal_with_p_weight_before(bar_widths[0])
        right_tail = self._halfnormal_with_p_weight_before(bar_widths[-1])

        bar_means = borders[:-1] + bar_widths / 2
        bar_means = bar_means.clone()
        bar_means[0] = borders[1] - left_tail.mean
        bar_means[-1] = borders[-2] + right_tail.mean

        return probs @ bar_means


class ScalarMSELoss(nn.MSELoss):
    """
    mse loss that reports one loss per prediction, like other decoders
    """

    def __init__(self) -> None:
        """
        turns off averaging that mse loss does by default
        """
        super().__init__(reduction="none")

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        measures squared error of each prediction
        """
        scalars = logits.reshape(target.shape)
        return super().forward(scalars, target)

    def mean(self, logits: torch.Tensor) -> torch.Tensor:
        """
        gives prediction itself, since scalar head needs no mean
        """
        return logits.squeeze(-1)


class QuantileLoss(nn.Module):
    """
    pinball loss averaged over a fixed grid of quantile levels
    """

    def __init__(self, n_quantiles: int) -> None:
        """
        spreads quantile levels evenly between 0 and 1
        """
        super().__init__()
        alphas = torch.arange(1, n_quantiles + 1, dtype=torch.float) / (n_quantiles + 1)
        self.register_buffer("alphas", alphas)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        measures how far each quantile misses target
        """
        alphas = self.alphas.to(logits.device)
        error = target.unsqueeze(-1) - logits
        losses = torch.maximum(alphas * error, (alphas - 1.0) * error)
        return losses.mean(dim=-1)

    def mean(self, logits: torch.Tensor) -> torch.Tensor:
        """
        reads one prediction out of quantiles
        """
        return logits.mean(dim=-1)


def make_regression_decoder(
    model: TabularFoundationModel,
) -> ScalarMSELoss | QuantileLoss | FullSupportBarDistribution:
    """
    gives loss that fits head in model config
    """
    head = model.config.head
    if head == "scalar":
        return ScalarMSELoss()
    if head == "quantiles":
        return QuantileLoss(model.borders.numel() - 1)
    if head == "buckets":
        return FullSupportBarDistribution(model.borders)
    raise ValueError(f"{head!r} head is not in (scalar, quantiles, buckets)")


def load_model(path: str | Path) -> TabularFoundationModel:
    """
    rebuilds model from checkpoint on disk
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model_class = getattr(models, checkpoint["model_class"])
    config_class = getattr(model_configs, checkpoint["config_class"])
    model = model_class(config=config_class(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    return model


class Experiment:
    """
    keeps record of one run, so it can be found again
    """

    def __init__(self, config: ExperimentConfig) -> None:
        """
        opens directory for this run
        """
        self.started = datetime.now()
        timestamp = self.started.strftime("%y%m%d-%H%M%S")
        uid = uuid.uuid4().hex[:8]
        name = config.name.strip()
        self.id = f"{timestamp}-{uid}-{name}" if name else f"{timestamp}-{uid}"
        self.dir = Path(config.experiments_dir) / config.problem / name / self.id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.dir / f"{self.id}-log.txt"
        self.score = None
        self.best_score = None
        self.best_checkpoint_path = self.dir / f"{self.id}-ckpt-best.pth"
        self.last_checkpoint_path = self.dir / f"{self.id}-ckpt-last.pth"

    def log_configs(self, **configs) -> None:
        """
        records what run was configured with
        """
        for label, config in configs.items():
            if config is None:
                continue
            self.log(f"{label}: {type(config).__name__}")
            for name, value in asdict(config).items():
                self.log(f"  {name}: {value}")

    def save_checkpoint(self, path: Path, model: TabularFoundationModel) -> None:
        """
        saves model so that interrupted write cannot lose earlier one
        """
        checkpoint = {
            "version": version("tfmplayground"),
            "experiment_id": self.id,
            "problem": model.config.problem,
            "model_class": type(model).__name__,
            "config_class": type(model.config).__name__,
            "model_config": asdict(model.config),
            "model_state": model.state_dict(),
        }
        temporary_path = path.with_suffix(".tmp")
        torch.save(checkpoint, temporary_path)
        os.replace(temporary_path, path)

    def save_checkpoints(self, model: TabularFoundationModel) -> None:
        """
        keeps last model, and best one so far
        """
        self.save_checkpoint(self.last_checkpoint_path, model)
        if self.score is not None and math.isfinite(self.score):
            if self.best_score is None or self.score > self.best_score:
                self.best_score = self.score
                self.save_checkpoint(self.best_checkpoint_path, model)

    def log(self, s: str, console: bool = False) -> None:
        """
        records one line, and shows it when asked
        """
        with open(self.log_path, "a") as f:
            if console:
                print(s)
            print(s, file=f)
