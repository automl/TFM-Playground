import torch
from torch import nn

from tfmplayground.interface import _migrate_feature_encoder_weights, init_model_from_state_dict_file
from tfmplayground.models.nanotabpfn import FeatureEncoder, NanoTabPFNModel, normalize_features


def test_migrate_expands_singleton_weight_with_zero_indicator():
    """A pre-indicator weight [E, 1] expands to [E, 2]: the value channel is kept, the
    new indicator channel is zero, and the bias is untouched.
    """
    weight = torch.randn(8, 1)
    bias = torch.randn(8)
    state = {
        "feature_encoder.linear_layer.weight": weight,
        "feature_encoder.linear_layer.bias": bias,
    }

    migrated = _migrate_feature_encoder_weights(state)
    w = migrated["feature_encoder.linear_layer.weight"]

    assert w.shape == (8, 2)
    assert torch.equal(w[:, 0:1], weight)   # value channel preserved
    assert torch.all(w[:, 1] == 0)          # indicator channel starts at zero
    assert torch.equal(migrated["feature_encoder.linear_layer.bias"], bias)  # bias untouched


def test_migrate_leaves_two_channel_weight_unchanged():
    """A checkpoint already in the new [E, 2] shape passes through untouched."""
    weight = torch.randn(8, 2)
    state = {"feature_encoder.linear_layer.weight": weight}

    migrated = _migrate_feature_encoder_weights(state)

    assert torch.equal(migrated["feature_encoder.linear_layer.weight"], weight)


def test_downgraded_checkpoint_loads_after_migration():
    """A state dict whose feature encoder is in the old [E, 1] shape loads into the
    current model once migrated, restoring the value channel and zeroing the indicator.
    """
    torch.manual_seed(0)
    kwargs = dict(embedding_size=8, num_attention_heads=2, mlp_hidden_size=16, num_layers=1, num_outputs=3)
    donor = NanoTabPFNModel(**kwargs)
    state = donor.state_dict()
    key = "feature_encoder.linear_layer.weight"
    old_state = {**state, key: state[key][:, :1].clone()}  # simulate an old [E, 1] checkpoint
    assert old_state[key].shape[1] == 1

    fresh = NanoTabPFNModel(**kwargs)
    fresh.load_state_dict(_migrate_feature_encoder_weights(old_state))  # must not raise

    w = fresh.feature_encoder.linear_layer.weight
    assert w.shape[1] == 2
    assert torch.equal(w[:, 0:1], old_state[key])  # value channel restored
    assert torch.all(w[:, 1] == 0)                 # indicator channel zero


def test_migrated_encoder_is_function_preserving_without_nan():
    """The crux: with the indicator channel zero-initialized, the migrated encoder
    reproduces the old value-only encoder exactly on NaN-free input.
    """
    torch.manual_seed(0)
    E = 8
    old = nn.Linear(1, E)  # the old value-only encoder linear

    enc = FeatureEncoder(E)
    migrated_w = torch.cat([old.weight, torch.zeros_like(old.weight)], dim=1)  # [E, 2]
    with torch.no_grad():
        enc.linear_layer.weight.copy_(migrated_w)
        enc.linear_layer.bias.copy_(old.bias)

    x = torch.randn(1, 6, 3)  # no NaN
    out_new = enc(x, train_test_split_index=3)

    # Old behavior: the old value-only linear applied to just the value channel.
    value = normalize_features(x, 3)[..., 0:1]  # (1, 6, 3, 1)
    out_old = old(value)

    assert torch.allclose(out_new, out_old, atol=1e-6)


def test_init_model_from_state_dict_file_migrates_old_checkpoint(tmp_path):
    """End-to-end: saving an old-format checkpoint to disk and loading it through the
    public entry point migrates it and yields a working [E, 2] model.
    """
    torch.manual_seed(0)
    arch = dict(embedding_size=8, num_attention_heads=2, mlp_hidden_size=16, num_layers=1, num_outputs=3)
    donor = NanoTabPFNModel(**arch)
    key = "feature_encoder.linear_layer.weight"
    old_model_state = {**donor.state_dict(), key: donor.state_dict()[key][:, :1].clone()}

    path = tmp_path / "old_checkpoint.pth"
    torch.save({"architecture": arch, "model": old_model_state}, path)

    model = init_model_from_state_dict_file(str(path))

    assert model.feature_encoder.linear_layer.weight.shape[1] == 2