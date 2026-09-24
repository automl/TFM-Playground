import torch
from tfmplayground.models.nanotabpfn import FeatureEncoder, TargetEncoder, normalize_features, pad_targets


def test_normalize_features_train_rows_to_zero_mean_unit_std():
    """Train rows of each feature should end up with mean ~0 and std ~1."""
    torch.manual_seed(0)

    # batch=1, 10 train rows, 3 features with deliberately different scales/offsets.
    scales = torch.tensor([1.0, 5.0, 20.0])
    offsets = torch.tensor([0.0, 3.0, -7.0])
    x = torch.randn(1, 10, 3) * scales + offsets

    normalized = normalize_features(x, train_test_split_index=10)[..., 0]  # value channel, (1, 10, 3)

    per_feature_mean = normalized.mean(dim=1)
    per_feature_std = normalized.std(dim=1)  # unbiased (ddof=1), matches normalize_features

    assert torch.allclose(per_feature_mean, torch.zeros_like(per_feature_mean), atol=1e-5)
    assert torch.allclose(per_feature_std, torch.ones_like(per_feature_std), atol=1e-4)


def test_normalize_features_uses_train_statistics_for_test_rows():
    """Test rows must be normalized using the TRAIN mean/std, not their own.

    This is the in-context-learning contract: predictions on test points may
    only depend on the training context. We build train and test with very
    different distributions and check the test output equals the train-stat
    normalization exactly (and is therefore NOT self-normalized to N(0,1)).
    """
    # 1 feature, deterministic. Train and test live on very different scales.
    train = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])  # (5, 1)
    test = torch.tensor([[10.0], [20.0]])                       # (2, 1)
    x = torch.cat([train, test], dim=0).unsqueeze(0)            # (1, 7, 1)
    n_train = train.shape[0]

    normalized = normalize_features(x, train_test_split_index=n_train)[..., 0]  # value channel, (1, 7, 1)
    normalized_test = normalized[:, n_train:].flatten()  # just the test values

    # Expected: reuse the same rule but with train stats only.
    train_mean = train.mean(dim=0)
    train_std = train.std(dim=0) + torch.finfo(torch.float32).eps  # unbiased (ddof=1), matches normalize_features
    expected_test = ((test - train_mean) / train_std).flatten()

    assert torch.allclose(normalized_test, expected_test, atol=1e-5)

    # And the point of it all: test rows are NOT self-normalized to mean 0.
    # If the rule wrongly used test's own stats, this mean would be ~0.
    assert normalized_test.mean().abs() > 1.0


def test_normalize_features_clips_extreme_values_to_plus_minus_100():
    """After normalization, values are clipped to the [-100, 100] range.

    A test point far outside the train distribution normalizes to a huge
    z-score (the train stats don't include it, so there's no self-limiting).
    It must be capped at 100.
    """
    # Train: tight cluster around 0 (mean~0, std~1). Test: one absurd value.
    train = torch.tensor([[-1.0], [0.0], [1.0], [0.0]])  # (4, 1)
    test = torch.tensor([[1e6]])                          # (1, 1)
    x = torch.cat([train, test], dim=0).unsqueeze(0)      # (1, 5, 1)

    normalized = normalize_features(x, train_test_split_index=train.shape[0])[..., 0].flatten()  # value channel

    # Nothing may exceed the clip bounds.
    assert normalized.max() <= 100.0
    assert normalized.min() >= -100.0

    # The out-of-distribution test point must have been clipped to the ceiling.
    assert torch.isclose(normalized.max(), torch.tensor(100.0), atol=1e-4)


def test_pad_targets_fills_test_positions_with_train_mean():
    """pad_targets keeps train labels intact and fills test positions with the
    train-label mean. It does NOT normalize (no division by std) — that happens
    outside the model. This pins that behavior.
    """
    # 3 train labels; the full sequence (train + test) has length 5.
    y_train = torch.tensor([[2.0], [4.0], [6.0]]).unsqueeze(0)  # (1, 3, 1)
    num_rows = 5
    n_train = y_train.shape[1]

    padded = pad_targets(y_train, num_rows).squeeze(-1).squeeze(-1)  # -> (1, 5)
    values = padded.flatten()

    train_mean = y_train.mean()  # (2 + 4 + 6) / 3 = 4.0

    # Train positions are untouched.
    assert torch.allclose(values[:n_train], torch.tensor([2.0, 4.0, 6.0]), atol=1e-6)
    # Test positions are filled with the train mean, exactly.
    assert torch.allclose(values[n_train:], torch.full((num_rows - n_train,), train_mean.item()), atol=1e-6)


def test_target_encoder_forward_embeds_padded_targets():
    """TargetEncoder.forward composes pad_targets with the linear layer: it
    returns the embedding-shaped tensor and equals embedding the padded targets.
    """
    torch.manual_seed(0)
    encoder = TargetEncoder(embedding_size=8)
    y_train = torch.tensor([[2.0], [4.0], [6.0]]).unsqueeze(0)  # (1, 3, 1)
    num_rows = 5

    out = encoder(y_train, num_rows)

    assert out.shape == (1, num_rows, 1, 8)
    assert torch.allclose(out, encoder.linear_layer(pad_targets(y_train, num_rows)))


def test_normalize_features_single_train_row_produces_finite_output():
    """A single training row makes the unbiased std NaN, and clip() does not
    rescue it (clamp of NaN is NaN), so the whole feature map used to come out
    NaN. The std must fall back so the output stays finite.
    """
    # 1 train row, 2 test rows, 2 features.
    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])  # (1, 3, 2)

    normalized = normalize_features(x, train_test_split_index=1)

    assert torch.isfinite(normalized).all()


def test_normalize_features_single_train_row_falls_back_to_unit_std():
    """With a single training row the std falls back to 1.0, so normalization
    reduces to mean-centering: the train row maps to 0 and each test row to its
    raw deviation from the train row (within the clip range here).
    """
    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])  # (1, 3, 2)

    normalized = normalize_features(x, train_test_split_index=1)[..., 0]  # value channel, (1, 3, 2)

    assert torch.allclose(normalized[0, 0], torch.tensor([0.0, 0.0]), atol=1e-6)  # train centers to 0
    assert torch.allclose(normalized[0, 1], torch.tensor([2.0, 2.0]), atol=1e-6)  # (3-1, 4-2)
    assert torch.allclose(normalized[0, 2], torch.tensor([4.0, 4.0]), atol=1e-6)  # (5-1, 6-2)


def test_feature_encoder_forward_embeds_normalized_features():
    """FeatureEncoder.forward composes normalize_features with the linear layer:
    it returns the embedding-shaped tensor and equals embedding the normalized input.
    """
    torch.manual_seed(0)
    encoder = FeatureEncoder(embedding_size=8)
    x = torch.tensor([[[1.0, 5.0], [2.0, 6.0], [3.0, 7.0]]])  # (1, 3, 2)

    out = encoder(x, train_test_split_index=2)

    assert out.shape == (1, 3, 2, 8)
    assert torch.allclose(out, encoder.linear_layer(normalize_features(x, 2)))


def test_normalize_features_emits_value_and_indicator_channels():
    """normalize_features returns a trailing axis of size 2, [value, indicator].
    With no missing entries the indicator is all zeros and the value channel equals
    the plain (mean/std) normalization. (Inverts the earlier singleton-axis pin.)
    """
    x = torch.tensor([[[1.0, 5.0], [2.0, 6.0], [3.0, 7.0]]])  # (1, 3, 2)

    out = normalize_features(x, train_test_split_index=2)

    assert out.shape == (1, 3, 2, 2)          # [value, indicator]
    assert (out[..., 1] == 0).all()           # nothing missing -> indicator all zero

    # Value channel equals the old normalization rule.
    train = x[:, :2]
    mean = train.mean(dim=1, keepdim=True)
    std = train.std(dim=1, keepdim=True) + torch.finfo(torch.float32).eps
    expected_value = torch.clip((x - mean) / std, -100, 100)
    assert torch.allclose(out[..., 0], expected_value, atol=1e-5)


def test_normalize_features_nan_in_train_no_longer_propagates():
    """A NaN in a training row no longer corrupts the column: stats ignore it, the
    hole is flagged in the indicator and imputed to 0 (the train mean after
    normalization), and every value stays finite. (Inverts the propagation quirk.)
    """
    x = torch.tensor([[[1.0, 10.0],
                       [2.0, 20.0],
                       [float("nan"), 30.0],
                       [4.0, 40.0],
                       [5.0, 50.0]]])  # (1, 5, 2); NaN in a train row, column 0

    out = normalize_features(x, train_test_split_index=3)
    value, indicator = out[..., 0], out[..., 1]

    assert torch.isfinite(value).all()        # no propagation over the column
    assert value[0, 2, 0] == 0.0              # the hole imputed to 0

    expected_indicator = torch.zeros(1, 5, 2)
    expected_indicator[0, 2, 0] = 1.0
    assert torch.equal(indicator, expected_indicator)  # flagged exactly at the hole


def test_normalize_features_nan_in_test_is_imputed_and_flagged():
    """A NaN in a test row is imputed (finite output) and flagged in the indicator,
    instead of leaving a NaN cell. (Inverts the test-local-NaN pin.)
    """
    x = torch.tensor([[[1.0], [2.0], [3.0], [float("nan")]]])  # (1, 4, 1); NaN in test row

    out = normalize_features(x, train_test_split_index=3)
    value, indicator = out[..., 0], out[..., 1]

    assert torch.isfinite(value).all()        # the test cell is imputed, not NaN
    assert indicator[0, 3, 0] == 1.0          # and flagged
    assert indicator.sum() == 1.0             # nothing else flagged


def test_feature_encoder_embeds_two_channels_and_handles_nan():
    """FeatureEncoder now embeds [value, indicator]: its linear layer takes 2 inputs,
    and a NaN in the input produces a finite embedding (value imputed, missingness
    carried by the indicator channel).
    """
    torch.manual_seed(0)
    encoder = FeatureEncoder(embedding_size=8)

    assert encoder.linear_layer.in_features == 2

    x = torch.tensor([[[1.0, 10.0], [2.0, 20.0], [float("nan"), 30.0]]])  # (1, 3, 2) with a NaN

    out = encoder(x, train_test_split_index=2)

    assert out.shape == (1, 3, 2, 8)
    assert torch.isfinite(out).all()