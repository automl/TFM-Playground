import h5py
import random
import torch
import numpy as np

from torch import nn


def set_randomness_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_default_device():
    device = 'cpu'
    if torch.backends.mps.is_available(): device = 'mps'
    if torch.cuda.is_available(): device = 'cuda'
    return device


def compute_bucket_borders(num_buckets, ys):
    """
    decides equal mass bucket borders from ys
    inspired by pfns.model.bar_distribution get_bucket_borders
    """
    ys = torch.as_tensor(ys, dtype=torch.float32).flatten()
    ys = ys[torch.isfinite(ys)]

    assert ys.numel() > num_buckets

    n = (ys.numel() // num_buckets) * num_buckets
    ys = ys[:n]
    ys_per_bucket = n // num_buckets

    ys_sorted, _ = torch.sort(ys)

    chunks = ys_sorted.reshape(num_buckets, ys_per_bucket)
    interiors = (chunks[:-1, -1] + chunks[1:, 0]) / 2

    min_outer = ys_sorted[0].unsqueeze(0)
    max_outer = ys_sorted[-1].unsqueeze(0)

    borders = torch.cat([min_outer, interiors, max_outer])

    assert borders.numel() == num_buckets + 1
    assert borders.numel() == torch.unique_consecutive(borders).numel()

    return borders


def make_global_bucket_edges(filename, n_buckets=100, device=get_default_device(), max_y=5_000_000):
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

    def __init__(self, borders, *, ignore_nan_targets=True):
        super().__init__()
        borders = torch.as_tensor(borders)
        assert borders.ndim == 1, "borders != 1d"
        if not torch.is_floating_point(borders):
            borders = borders.to(torch.get_default_dtype())
        borders = borders.contiguous()
        self.register_buffer("borders", borders)
        assert (self.bar_widths > 0).all(), "borders must be strictly increasing"  # does not allow zero sized buckets
        self.ignore_nan_targets = ignore_nan_targets

    @property
    def bar_widths(self):
        return self.borders[1:] - self.borders[:-1]

    @property
    def num_bars(self):
        return self.borders.numel() - 1

    def _ignore_init(self, y):
        """
        makes ignore mask for nan targets and alters y (will be ignored later)
        """
        ignore_mask = torch.isnan(y)
        if ignore_mask.any():
            assert self.ignore_nan_targets, "nan in y while ignore_nan_targets=False"
            y[ignore_mask] = self.borders[0]  # just a default value, will be ignored later
        return ignore_mask

    def _map_to_bar_indices(self, y):
        """
        maps each y to its corresponding bar index
        """
        indices = torch.searchsorted(self.borders, y) - 1
        indices = indices.clamp(0, self.num_bars - 1)
        return indices

    def _compute_scaled_log_probs(self, logits):
        """
        log prob density
        """
        return torch.log_softmax(logits, dim=-1) - torch.log(self.bar_widths)


class FullSupportBarDistribution(BarDistribution):
    """
    extends BarDistribution with half normal tails on both sides for full support
    inspired by pfns.model.bar_distribution FullSupportBarDistribution
    """

    def __init__(self, borders: torch.Tensor, *, ignore_nan_targets: bool = True):
        super().__init__(borders, ignore_nan_targets=ignore_nan_targets)

    @staticmethod
    def _halfnormal_with_p_weight_before(desired_quantile_value_at_p, p=0.5):
        """
        scales the half normal distribution so that the p weight is before the desired value
        """
        standard_halfnormal = torch.distributions.HalfNormal(torch.tensor(1.0))
        quantile_value_at_p = standard_halfnormal.icdf(torch.tensor(p))
        scale = desired_quantile_value_at_p / quantile_value_at_p
        scaled_halfnormal = torch.distributions.HalfNormal(scale)
        return scaled_halfnormal

    def forward(self, logits, y):
        """
        negative log likelihood of y given logits
        """
        assert logits.shape[-1] == self.num_bars, f"logits last dim shape != num bars"

        y = torch.as_tensor(y).clone().reshape(*logits.shape[:-1])
        ignore_mask = self._ignore_init(y)  # alters y
        y_bar_indices = self._map_to_bar_indices(y)
        scaled_log_probs = self._compute_scaled_log_probs(logits)
        gathered_scaled_log_probs = scaled_log_probs.gather(-1, y_bar_indices.unsqueeze(-1)).squeeze(-1)

        left_tail = self._halfnormal_with_p_weight_before(self.bar_widths[0])
        right_tail = self._halfnormal_with_p_weight_before(self.bar_widths[-1])

        left_mask = y_bar_indices == 0
        if left_mask.any():
            distances = (self.borders[1] - y[left_mask]).clamp(min=1e-8)
            gathered_scaled_log_probs[left_mask] += left_tail.log_prob(distances) + torch.log(self.bar_widths[0])

        right_mask = y_bar_indices == self.num_bars - 1
        if right_mask.any():
            distances = (y[right_mask] - self.borders[-2]).clamp(min=1e-8)
            gathered_scaled_log_probs[right_mask] += right_tail.log_prob(distances) + torch.log(self.bar_widths[-1])

        nll = -gathered_scaled_log_probs
        nll[ignore_mask] = 0.0
        return nll

    def mean(self, logits):
        """
        calculates the expected value of the distribution given logits
        """
        assert logits.shape[-1] == self.num_bars, f"logits last dim shape != num bars"

        probs = torch.softmax(logits, dim=-1)

        left_tail = self._halfnormal_with_p_weight_before(self.bar_widths[0])
        right_tail = self._halfnormal_with_p_weight_before(self.bar_widths[-1])

        bar_means = self.borders[:-1] + self.bar_widths / 2
        bar_means = bar_means.clone()
        bar_means[0] = self.borders[1] - left_tail.mean
        bar_means[-1] = self.borders[-2] + right_tail.mean

        return probs @ bar_means
