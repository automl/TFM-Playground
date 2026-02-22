import h5py
import random
import torch
import numpy as np

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
