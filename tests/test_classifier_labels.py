"""Label-handling tests for NanoTabPFNClassifier.

The classifier advertises a scikit-learn-like interface, so it must accept
arbitrary class labels (non-contiguous integers, strings) rather than assuming
0..K-1. Labels are encoded to contiguous indices for the model and decoded back
to the originals in predict.
"""

import numpy as np
import pytest
import torch

from tfmplayground.interface import NanoTabPFNClassifier

# A small, varied feature matrix the preprocessor accepts (no constant columns).
X = np.array([[1.0, 5.0], [2.0, 6.0], [3.0, 7.0], [4.0, 8.0]])


def _make_classifier(num_outputs=10):
    # Identity model + explicit cpu so __init__ neither downloads a checkpoint
    # nor needs a GPU. The model is never called: predict_proba is patched.
    # Real models expose num_outputs, which fit() checks against the class count.
    model = torch.nn.Identity()
    model.num_outputs = num_outputs
    return NanoTabPFNClassifier(model=model, device="cpu")


def test_fit_encodes_noncontiguous_integer_labels():
    """max(set(y)) + 1 miscounts when labels are non-contiguous; the count must
    be the number of distinct classes, and labels must map to 0..K-1.
    """
    clf = _make_classifier()
    clf.fit(X, np.array([10, 20, 30, 10]))

    assert clf.num_classes == 3
    assert list(clf.classes_) == [10, 20, 30]
    assert set(clf.y_train.tolist()) == {0, 1, 2}


def test_fit_accepts_string_labels():
    """String labels used to crash (max() + 1 on strings); they must encode."""
    clf = _make_classifier()
    clf.fit(X, np.array(["cat", "dog", "cat", "bird"]))

    assert clf.num_classes == 3
    assert list(clf.classes_) == ["bird", "cat", "dog"]  # LabelEncoder sorts


def test_predict_returns_original_labels(monkeypatch):
    """predict must return the caller's labels, not internal class indices."""
    clf = _make_classifier()
    clf.fit(X[:3], np.array([10, 20, 30]))

    fake_proba = np.array([[0.1, 0.2, 0.7], [0.8, 0.1, 0.1]])  # argmax -> 2, 0
    monkeypatch.setattr(clf, "predict_proba", lambda X_test: fake_proba)

    preds = clf.predict(np.array([[9.0, 9.0], [9.0, 9.0]]))

    assert list(preds) == [30, 10]


def test_contiguous_zero_based_labels_are_unchanged(monkeypatch):
    """Regression guard: for the already-supported 0..K-1 case, both the class
    count and the predictions are exactly what they were before.
    """
    clf = _make_classifier()
    clf.fit(X[:3], np.array([0, 1, 2]))

    assert clf.num_classes == 3

    fake_proba = np.array([[0.1, 0.7, 0.2], [0.6, 0.3, 0.1]])  # argmax -> 1, 0
    monkeypatch.setattr(clf, "predict_proba", lambda X_test: fake_proba)

    preds = clf.predict(np.array([[9.0, 9.0], [9.0, 9.0]]))

    assert list(preds) == [1, 0]


def test_fit_returns_self():
    """scikit-learn convention: fit returns the estimator, so calls can chain."""
    clf = _make_classifier()
    assert clf.fit(X[:3], np.array([0, 1, 2])) is clf


def test_fit_raises_when_more_classes_than_model_outputs():
    """The model can represent at most num_outputs classes; more must error
    loudly instead of silently predicting only the first num_outputs.
    """
    clf = _make_classifier(num_outputs=2)
    with pytest.raises(ValueError):
        clf.fit(X[:3], np.array([0, 1, 2]))  # 3 classes, model supports 2