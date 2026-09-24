import numpy as np
import torch

from tfmplayground.interface import NanoTabPFNClassifier
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


def _classifier(num_layers=2):
    torch.manual_seed(0)
    model = NanoTabPFNModel(
        embedding_size=16, num_attention_heads=2, mlp_hidden_size=32, num_layers=num_layers, num_outputs=3
    )
    return model, NanoTabPFNClassifier(model=model, device="cpu")


def test_attention_not_saved_by_default():
    model, x, y, tts = _tiny_model_and_batch()
    with torch.inference_mode():
        model((x, y), train_test_split_index=tts)
    for block in model.transformer_blocks:
        assert block.feature_attention is None


def test_saving_attention_does_not_change_output():
    model, x, y, tts = _tiny_model_and_batch()
    with torch.inference_mode():
        out_off = model((x, y), train_test_split_index=tts)
    for block in model.transformer_blocks:
        block.save_feature_attention = True
    with torch.inference_mode():
        out_on = model((x, y), train_test_split_index=tts)
    assert torch.allclose(out_off, out_on, atol=1e-6)


def test_captured_attention_shape_and_normalization():
    model, x, y, tts = _tiny_model_and_batch()
    for block in model.transformer_blocks:
        block.save_feature_attention = True
    with torch.inference_mode():
        model((x, y), train_test_split_index=tts)
    num_columns = x.shape[2] + 1
    for block in model.transformer_blocks:
        assert block.feature_attention is not None
        assert block.feature_attention.shape == (num_columns,)
        assert torch.allclose(block.feature_attention.sum(), torch.tensor(1.0), atol=1e-4)


def test_feature_attention_scores_maps_to_original_features():
    """Scores come back one per ORIGINAL feature: the constant column is dropped (NaN),
    numeric and declared-categorical columns get a finite score, in the original order.
    """
    model, clf = _classifier()
    clf.categorical_features = [2]
    # cols: 0=num, 1=CONSTANT (dropped), 2=cat (declared), 3=num
    X_train = np.array([[1.0, 7.0, 0.0, 10.0],
                        [2.0, 7.0, 1.0, 20.0],
                        [3.0, 7.0, 0.0, 30.0],
                        [4.0, 7.0, 1.0, 40.0]])
    y_train = np.array([0, 1, 0, 1])
    clf.fit(X_train, y_train)

    scores = clf.feature_attention_scores(np.array([[5.0, 7.0, 0.0, 50.0]]))

    assert scores.shape == (4,)
    assert np.isnan(scores[1])          # constant column dropped
    assert not np.isnan(scores[0])      # numeric
    assert not np.isnan(scores[2])      # declared categorical
    assert not np.isnan(scores[3])      # numeric


def test_feature_attention_scores_leaves_capture_off():
    """After computing scores, capture is turned back off and buffers cleared, so normal
    prediction is unaffected.
    """
    model, clf = _classifier()
    X_train = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 5.0]])
    y_train = np.array([0, 1, 0, 1])
    clf.fit(X_train, y_train)

    clf.feature_attention_scores(np.array([[5.0, 2.0]]))

    for block in model.transformer_blocks:
        assert block.save_feature_attention is False
        assert block.feature_attention is None


def test_feature_attention_scores_in_unit_range():
    """Attention scores are softmax values, so every non-NaN score lies in [0, 1]."""
    model, clf = _classifier()
    X_train = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 1.0], [3.0, 1.0, 2.0], [4.0, 5.0, 6.0]])
    y_train = np.array([0, 1, 0, 1])
    clf.fit(X_train, y_train)

    scores = clf.feature_attention_scores(np.array([[5.0, 2.0, 4.0]]))
    valid = scores[~np.isnan(scores)]

    assert (valid >= 0).all() and (valid <= 1).all()