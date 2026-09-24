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


def test_embeddings_not_saved_by_default():
    """Without opting in, no embeddings are captured: the normal path is untouched."""
    model, x, y, tts = _tiny_model_and_batch()
    with torch.inference_mode():
        model((x, y), train_test_split_index=tts)
    assert model.embeddings is None


def test_saving_embeddings_does_not_change_output():
    """Capturing embeddings is a pure side effect: predictions are identical either way."""
    model, x, y, tts = _tiny_model_and_batch()
    with torch.inference_mode():
        out_off = model((x, y), train_test_split_index=tts)
    model.save_embeddings = True
    with torch.inference_mode():
        out_on = model((x, y), train_test_split_index=tts)
    assert torch.allclose(out_off, out_on, atol=1e-6)


def test_captured_embeddings_shape():
    """With the flag on, the model stores the per-row target-token embedding (before the
    decoder): shape (batch, num_rows, embedding_size), covering both train and test rows.
    """
    model, x, y, tts = _tiny_model_and_batch()
    model.save_embeddings = True
    with torch.inference_mode():
        model((x, y), train_test_split_index=tts)
    num_rows = x.shape[1]
    assert model.embeddings is not None
    assert model.embeddings.shape == (1, num_rows, 16)