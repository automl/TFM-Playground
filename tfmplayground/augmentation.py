from __future__ import annotations

import torch


def inject_mcar_missingness(
    x: torch.Tensor,
    rate: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sets feature cells to NaN Missing-Completely-At-Random (MCAR).

    Each cell of ``x`` is independently set to NaN with probability ``rate``. This is a
    training-time augmentation: with the selective NaN filter in place, these NaNs reach
    the model, which imputes them and flags them via the indicator channel, so the indicator
    weights receive gradient and learn. Apply it to FEATURES only, never to targets.

    Args:
        x: feature tensor (any shape).
        rate: per-cell probability of being set to NaN, in [0, 1].
        generator: optional torch.Generator for reproducibility.

    Returns:
        A new tensor (the input is never modified). ``rate == 0`` returns an unchanged copy.
    """
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"rate must be in [0, 1], got {rate}.")
    x = x.clone()
    if rate == 0.0:
        return x
    mask = torch.rand(x.shape, generator=generator, device=x.device) < rate
    x[mask] = float("nan")
    return x


def add_widening_features(
    x: torch.Tensor,
    num_features_to_add: int,
    sparsity: float,
    noise_std: float,
    include_original_prob: float = 0.5,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Continuous feature widening (Algorithm 1 of TabPFN-Wide).

    Generates ``num_features_to_add`` new features as a sparse linear projection of the
    original ones plus feature-dependent Gaussian noise, mimicking the strong inter-feature
    correlations of HDLSS data. With probability ``include_original_prob`` the original
    features are appended and the whole feature order is permuted.

    Operates on the last dimension (features); all leading dimensions are treated as batch.
    Per the paper, the noise std of each new feature is proportional to that feature's own
    std over the SAMPLES dimension (the second-to-last axis).

    Args:
        x: feature tensor of shape (..., num_samples, num_features).
        num_features_to_add: how many new features to generate (>= 0).
        sparsity: probability that a projection weight is kept (Bernoulli(p)); paper uses [0, 0.05].
        noise_std: noise scale sigma; paper uses [0, 1].
        include_original_prob: probability of appending the originals and permuting.
        generator: optional torch.Generator for reproducibility.

    Returns:
        A new tensor (the input is never modified).
    """
    if num_features_to_add < 0:
        raise ValueError(f"num_features_to_add must be >= 0, got {num_features_to_add}.")
    if not 0.0 <= sparsity <= 1.0:
        raise ValueError(f"sparsity must be in [0, 1], got {sparsity}.")
    if noise_std < 0:
        raise ValueError(f"noise_std must be >= 0, got {noise_std}.")

    if num_features_to_add == 0:
        return x.clone()

    m = x.shape[-1]
    weights = torch.randn(m, num_features_to_add, generator=generator, device=x.device, dtype=x.dtype)
    mask = (torch.rand(m, num_features_to_add, generator=generator, device=x.device) < sparsity).to(x.dtype)
    x_wide = x @ (mask * weights)  # (..., num_features_to_add)

    # Feature-dependent noise: std of each new feature over the samples axis.
    stds = x_wide.std(dim=-2, keepdim=True)
    stds = torch.where(stds == 0, torch.ones_like(stds), stds)  # columnas nulas → ruido puro
    noise = torch.randn(x_wide.shape, generator=generator, device=x.device, dtype=x.dtype) * (noise_std * stds)
    x_wide = x_wide + noise

    if torch.rand(1, generator=generator, device=x.device).item() < include_original_prob:
        x_wide = torch.cat([x, x_wide], dim=-1)
        perm = torch.randperm(x_wide.shape[-1], generator=generator, device=x.device)
        x_wide = x_wide[..., perm]
    return x_wide

def detect_categorical_mask(x: torch.Tensor, max_unique: int = 20) -> torch.Tensor:
    """Boolean mask of categorical features (Algorithm 2 type detection, TabPFN-Wide).

    A feature is considered categorical if it has at most ``max_unique`` distinct
    observed (non-NaN) values, following the paper's rule (<= 20 distinct values);
    every other feature is treated as continuous. Distinct values are counted per
    feature over the SAMPLES axis, ignoring NaNs so that injected missingness never
    inflates a feature's cardinality.

    This is the WIDENING-time detection applied to the synthetic prior, and is
    deliberately separate from the inference-time categorical detection used on real
    data (which uses a different, smaller threshold calibrated for n ~ 150).

    Args:
        x: feature tensor of shape (num_samples, num_features).
        max_unique: maximum number of distinct values for a feature to count as categorical.

    Returns:
        Boolean tensor of shape (num_features,), True where the feature is categorical.
    """
    if x.ndim != 2:
        raise ValueError(f"x must be 2D (num_samples, num_features), got shape {tuple(x.shape)}.")

    # Sort each column; torch places NaNs at the end. A distinct non-NaN value starts a
    # new group when it differs from the value above and neither cell is NaN.
    x_sorted, _ = torch.sort(x, dim=0)
    nan_sorted = torch.isnan(x_sorted)
    both_valid = (~nan_sorted[:-1]) & (~nan_sorted[1:])
    new_value = both_valid & (x_sorted[:-1] != x_sorted[1:])
    has_valid = (~nan_sorted).any(dim=0)
    n_unique = new_value.sum(dim=0) + has_valid.long()
    return n_unique <= max_unique


def split_widening_budget(cat_mask: torch.Tensor, num_features_to_add: int) -> tuple[int, int]:
    """Allocate the widening budget between continuous and categorical features.

    Following the paper: with categorical ratio ``r_cat = num_categorical / num_total``,
    allocate ``d_cat = floor(r_cat * num_features_to_add)`` categorical features and the
    remaining ``d_cont = num_features_to_add - d_cat`` continuous features. When there are
    no categorical features (r_cat = 0), the whole budget goes to continuous features (the
    all-continuous omics scenario).

    Args:
        cat_mask: boolean tensor of shape (num_features,) from ``detect_categorical_mask``.
        num_features_to_add: total number of new features to generate (d - m), >= 0.

    Returns:
        (num_continuous_to_add, num_categorical_to_add), summing to num_features_to_add.
    """
    if num_features_to_add < 0:
        raise ValueError(f"num_features_to_add must be >= 0, got {num_features_to_add}.")
    num_total = cat_mask.numel()
    if num_total == 0:
        raise ValueError("cat_mask must not be empty.")

    num_categorical = int(cat_mask.sum().item())
    r_cat = num_categorical / num_total
    num_cat_to_add = int(r_cat * num_features_to_add)  # floor for non-negative values
    num_cont_to_add = num_features_to_add - num_cat_to_add
    return num_cont_to_add, num_cat_to_add


def sample_target_cardinalities(
    size: int,
    max_cats: int,
    low: int = 3,
    lam: float = 0.08,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample target cardinalities Kj from a discrete exponential distribution (Algorithm 2).

    Each Kj is drawn from {low, ..., max_cats}, with smaller values more likely (bias set by
    ``lam``). This follows the paper's prose ("sampled from a discrete exponential
    distribution, biasing towards low-cardinality"); the pseudocode's line 8 instead says
    uniform -- we follow the prose and the reference implementation.

    Args:
        size: number of cardinalities to sample (one per new categorical feature), >= 0.
        max_cats: maximum cardinality (inclusive upper bound, Kmax in the paper), >= 1.
        low: minimum cardinality (inclusive); clamped down to max_cats if larger.
        lam: exponential rate; larger biases more strongly towards ``low``.
        generator: optional torch.Generator for reproducibility.

    Returns:
        Long tensor of shape (size,) with values in [low, max_cats].
    """
    if size < 0:
        raise ValueError(f"size must be >= 0, got {size}.")
    if max_cats < 1:
        raise ValueError(f"max_cats must be >= 1, got {max_cats}.")

    device = generator.device if generator is not None else torch.device("cpu")
    if size == 0:
        return torch.empty(0, dtype=torch.long, device=device)

    low = min(low, max_cats)
    values = torch.arange(low, max_cats + 1, device=device)
    weights = torch.exp(-lam * (values - low).float())
    probs = weights / weights.sum()
    idx = torch.multinomial(probs, num_samples=size, replacement=True, generator=generator)
    return values[idx]


def reduce_cardinality(
    col: torch.Tensor,
    max_cats: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Reduce a categorical column to at most ``max_cats`` categories (Algorithm 2, lines 9-12).

    Keeps the ``max_cats`` most frequent categories and remaps every rarer category to a
    uniformly random kept one. This is a vectorised equivalent of the paper's iterative
    "merge the rarest category into one of the Kj most frequent" loop, with the same end
    state: at most ``max_cats`` categories, the rare tail absorbed into frequent ones at
    random. Assumes ``col`` has no NaN (widening runs before missingness injection).

    Args:
        col: 1D tensor of categorical values.
        max_cats: maximum number of distinct categories to keep, >= 1.
        generator: optional torch.Generator for reproducibility.

    Returns:
        A new 1D tensor, same shape as ``col``, with at most ``max_cats`` distinct values.
    """
    if max_cats < 1:
        raise ValueError(f"max_cats must be >= 1, got {max_cats}.")

    vals, inv = torch.unique(col, return_inverse=True)
    num_categories = vals.numel()
    if num_categories <= max_cats:
        return col.clone()

    counts = torch.bincount(inv, minlength=num_categories)
    order = torch.argsort(counts)                # ascending: rarest first, most frequent last
    head = order[-max_cats:]                      # the max_cats most frequent categories (kept)
    tail = order[:-max_cats]                      # the rarer categories (to be merged away)
    tgt = torch.randint(0, head.numel(), (tail.numel(),), generator=generator, device=col.device)
    mapping = torch.arange(num_categories, device=col.device)
    mapping[tail] = head[tgt]
    return vals[mapping[inv]]


def add_categorical_widening_features(
    x_cat: torch.Tensor,
    num_features_to_add: int,
    sparsity: float,
    max_cats: int = 20,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Generate new categorical features from existing ones (Algorithm 2, one dataset).

    For each new feature: pick k = max(1, floor(sparsity * m_cat)) donor columns, copy a
    value from a uniformly random donor per sample (lines 2-6), then reduce the result to a
    target cardinality Kj drawn from a discrete exponential distribution (lines 8-12).

    Operates on a SINGLE dataset; the batch loop belongs to the mixed widening wrapper,
    matching the paper's "feature widening is applied independently per dataset".

    Args:
        x_cat: categorical sub-matrix of one dataset, shape (num_samples, num_categorical).
        num_features_to_add: number of new categorical features to generate, >= 0.
        sparsity: donor sparsity p; k = max(1, floor(p * num_categorical)).
        max_cats: maximum target cardinality Kmax.
        generator: optional torch.Generator for reproducibility.

    Returns:
        A new tensor of shape (num_samples, num_features_to_add).
    """
    if x_cat.ndim != 2:
        raise ValueError(f"x_cat must be 2D (num_samples, num_categorical), got {tuple(x_cat.shape)}.")
    if num_features_to_add < 0:
        raise ValueError(f"num_features_to_add must be >= 0, got {num_features_to_add}.")
    if not 0.0 <= sparsity <= 1.0:
        raise ValueError(f"sparsity must be in [0, 1], got {sparsity}.")

    num_samples, num_categorical = x_cat.shape
    if num_features_to_add == 0:
        return x_cat.new_empty((num_samples, 0))
    if num_categorical == 0:
        raise ValueError("Cannot generate categorical features without at least one donor column.")

    k = max(1, int(sparsity * num_categorical))
    target_cards = sample_target_cardinalities(num_features_to_add, max_cats, generator=generator)
    rows = torch.arange(num_samples, device=x_cat.device)
    out = x_cat.new_empty((num_samples, num_features_to_add))
    for j in range(num_features_to_add):
        donors = torch.randperm(num_categorical, generator=generator, device=x_cat.device)[:k]
        donor_per_row = torch.randint(0, k, (num_samples,), generator=generator, device=x_cat.device)
        new_col = x_cat[rows, donors[donor_per_row]]
        out[:, j] = reduce_cardinality(new_col, int(target_cards[j].item()), generator=generator)
    return out


def add_mixed_widening_features(
    x: torch.Tensor,
    num_features_to_add: int,
    sparsity: float,
    noise_std: float,
    max_cats: int = 20,
    include_original_prob: float = 0.5,
    max_unique_for_categorical: int = 20,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Mixed continuous + categorical feature widening, applied per dataset (TabPFN-Wide).

    For each dataset in the batch: detect categorical vs continuous features, split the
    widening budget by the categorical ratio, widen continuous features with Algorithm 1 and
    categorical features with Algorithm 2, concatenate the new features, and -- once, shared
    across the batch, with probability ``include_original_prob`` -- append the original
    features and permute the column order. The per-batch (not per-dataset) append decision
    keeps every dataset the same width so the batch can be stacked back together.

    Applying widening independently per dataset follows the paper ("feature widening is
    applied independently per dataset"). The continuous branch reuses ``add_widening_features``
    with ``include_original_prob=0.0`` so the originals are appended only once, here.

    Args:
        x: feature tensor of shape (batch_size, num_samples, num_features).
        num_features_to_add: total new features per dataset (d - m), >= 0.
        sparsity: sparsity p shared by both algorithms; paper uses [0, 0.05].
        noise_std: continuous noise scale sigma; paper uses [0, 1].
        max_cats: maximum categorical cardinality Kmax for Algorithm 2.
        include_original_prob: probability of appending originals and permuting (once, batch-wide).
        max_unique_for_categorical: categorical detection threshold (paper: <= 20).
        generator: optional torch.Generator for reproducibility.

    Returns:
        A new tensor of shape (batch_size, num_samples, width), where width is
        num_features_to_add (plus num_features when the originals are appended). The input is
        never modified.
    """
    if x.ndim != 3:
        raise ValueError(f"x must be 3D (batch_size, num_samples, num_features), got {tuple(x.shape)}.")
    if num_features_to_add < 0:
        raise ValueError(f"num_features_to_add must be >= 0, got {num_features_to_add}.")
    if not 0.0 <= sparsity <= 1.0:
        raise ValueError(f"sparsity must be in [0, 1], got {sparsity}.")
    if noise_std < 0:
        raise ValueError(f"noise_std must be >= 0, got {noise_std}.")

    if num_features_to_add == 0:
        return x.clone()

    # One batch-wide decision so every dataset ends with the same width (stackable).
    include_original = (
        torch.rand(1, generator=generator, device=x.device).item() < include_original_prob
    )

    widened = []
    for b in range(x.shape[0]):
        xb = x[b]  # (num_samples, num_features)
        cat_mask = detect_categorical_mask(xb, max_unique=max_unique_for_categorical)
        num_cont_to_add, num_cat_to_add = split_widening_budget(cat_mask, num_features_to_add)

        blocks = []
        if num_cont_to_add > 0:
            x_cont = xb[:, ~cat_mask]
            blocks.append(
                add_widening_features(
                    x_cont, num_cont_to_add, sparsity, noise_std,
                    include_original_prob=0.0, generator=generator,
                )
            )
        if num_cat_to_add > 0:
            x_cat = xb[:, cat_mask]
            blocks.append(
                add_categorical_widening_features(
                    x_cat, num_cat_to_add, sparsity, max_cats=max_cats, generator=generator,
                )
            )
        new_features = torch.cat(blocks, dim=-1)  # (num_samples, num_features_to_add)

        if include_original:
            new_features = torch.cat([xb, new_features], dim=-1)
            perm = torch.randperm(new_features.shape[-1], generator=generator, device=x.device)
            new_features = new_features[:, perm]
        widened.append(new_features)

    return torch.stack(widened, dim=0)