import numpy as np

from tfmplayground.configs.models import NanoTabPFNClassifierConfig
from tfmplayground.interface import TabularClassifier
from tfmplayground.models.nanotabpfn import NanoTabPFNModel


def test_classifier_is_deterministic():
    """A randomly initialized classifier should give identical outputs for identical inputs."""
    model = NanoTabPFNModel(
        config=NanoTabPFNClassifierConfig(
            embedding_size=16,
            num_attention_heads=2,
            mlp_hidden_size=32,
            num_layers=2,
            num_outputs=3,
        )
    )
    classifier = TabularClassifier(model, device="cpu")

    rng = np.random.default_rng(0)
    X_train = rng.standard_normal((20, 4))
    y_train = rng.integers(0, 3, size=20)
    X_test = rng.standard_normal((10, 4))

    classifier.fit(X_train, y_train)
    first = classifier.predict_proba(X_test)
    second = classifier.predict_proba(X_test)

    np.testing.assert_array_equal(first, second)
