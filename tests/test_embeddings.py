import numpy as np
import torch

from tfmplayground.interface import NanoTabPFNClassifier, NanoTabPFNRegressor
from tfmplayground.models.nanotabpfn import NanoTabPFNModel


def _tiny_model_and_batch():
    torch.manual_seed(0)
    model = NanoTabPFNModel(
        embedding_size=16, num_attention_heads=2, mlp_hidden_size=32, num_layers=2, num_outputs=3
    )
    model.eval()
    x = torch.randn(1, 6, 3)
    y = torch.randint(0, 3, (1, 4)).float()
    train_test_split_index = 4
    return model, x, y, train_test_split_index


def test_embeddings_not_saved_by_default():
    model, x, y, tts = _tiny_model_and_batch()
    with torch.inference_mode():
        model((x, y), train_test_split_index=tts)
    assert model.embeddings is None


def test_saving_embeddings_does_not_change_output():
    model, x, y, tts = _tiny_model_and_batch()
    with torch.inference_mode():
        out_off = model((x, y), train_test_split_index=tts)
    model.save_embeddings = True
    with torch.inference_mode():
        out_on = model((x, y), train_test_split_index=tts)
    assert torch.allclose(out_off, out_on, atol=1e-6)


def test_captured_embeddings_shape():
    model, x, y, tts = _tiny_model_and_batch()
    model.save_embeddings = True
    with torch.inference_mode():
        model((x, y), train_test_split_index=tts)
    num_rows = x.shape[1]
    assert model.embeddings is not None
    assert model.embeddings.shape == (1, num_rows, 16)


def test_classifier_get_embeddings_shape():
    """The classifier returns one embedding vector per test row: (n_test, embedding_size)."""
    torch.manual_seed(0)
    model = NanoTabPFNModel(embedding_size=16, num_attention_heads=2, mlp_hidden_size=32, num_layers=2, num_outputs=3)
    clf = NanoTabPFNClassifier(model=model, device="cpu")
    X_train = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 5.0]])
    y_train = np.array([0, 1, 0, 1])
    clf.fit(X_train, y_train)

    emb = clf.get_embeddings(np.array([[5.0, 2.0], [6.0, 3.0]]))

    assert emb.shape == (2, 16)


def test_regressor_get_embeddings_shape():
    """The regressor extracts embeddings the same way (pre-decoder), so the shape matches."""
    torch.manual_seed(0)
    model = NanoTabPFNModel(embedding_size=16, num_attention_heads=2, mlp_hidden_size=32, num_layers=2, num_outputs=8)
    reg = NanoTabPFNRegressor(model=model, device="cpu")
    X_train = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 5.0]])
    y_train = np.array([1.5, 2.5, 0.5, 3.5])
    reg.fit(X_train, y_train)

    emb = reg.get_embeddings(np.array([[5.0, 2.0], [6.0, 3.0]]))

    assert emb.shape == (2, 16)


def test_get_embeddings_leaves_capture_off():
    """After extraction the flag is reset and the buffer cleared, so prediction is unaffected."""
    torch.manual_seed(0)
    model = NanoTabPFNModel(embedding_size=16, num_attention_heads=2, mlp_hidden_size=32, num_layers=2, num_outputs=3)
    clf = NanoTabPFNClassifier(model=model, device="cpu")
    X_train = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 5.0]])
    y_train = np.array([0, 1, 0, 1])
    clf.fit(X_train, y_train)

    clf.get_embeddings(np.array([[5.0, 2.0]]))

    assert clf.model.save_embeddings is False
    assert clf.model.embeddings is None