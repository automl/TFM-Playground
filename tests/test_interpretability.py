import torch

from tfmplayground.models.nanotabpfn import NanoTabPFNModel


def _tiny_model_and_batch():
    torch.manual_seed(0)
    model = NanoTabPFNModel(
        embedding_size=16, num_attention_heads=2, mlp_hidden_size=32, num_layers=2, num_outputs=3
    )
    model.eval()
    x = torch.randn(1, 6, 3)                      # 1 batch, 6 rows, 3 features
    y = torch.randint(0, 3, (1, 4)).float()      # 4 train labels
    train_test_split_index = 4
    return model, x, y, train_test_split_index


def test_attention_not_saved_by_default():
    """Without opting in, no attention is captured: the normal path is untouched."""
    model, x, y, tts = _tiny_model_and_batch()

    with torch.inference_mode():
        model((x, y), train_test_split_index=tts)

    for block in model.transformer_blocks:
        assert block.feature_attention is None


def test_saving_attention_does_not_change_output():
    """Capturing attention is a pure side effect: predictions are identical whether or
    not the flag is on.
    """
    model, x, y, tts = _tiny_model_and_batch()

    with torch.inference_mode():
        out_off = model((x, y), train_test_split_index=tts)

    for block in model.transformer_blocks:
        block.save_feature_attention = True
    with torch.inference_mode():
        out_on = model((x, y), train_test_split_index=tts)

    assert torch.allclose(out_off, out_on, atol=1e-6)


def test_captured_attention_shape_and_normalization():
    """With the flag on, each layer stores the target column's attention to every column:
    a (num_columns,) vector that, being an averaged softmax row, sums to ~1.
    """
    model, x, y, tts = _tiny_model_and_batch()

    for block in model.transformer_blocks:
        block.save_feature_attention = True
    with torch.inference_mode():
        model((x, y), train_test_split_index=tts)

    num_columns = x.shape[2] + 1  # features + the appended target column
    for block in model.transformer_blocks:
        assert block.feature_attention is not None
        assert block.feature_attention.shape == (num_columns,)
        assert torch.allclose(block.feature_attention.sum(), torch.tensor(1.0), atol=1e-4)