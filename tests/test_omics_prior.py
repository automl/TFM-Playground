import numpy as np
import pytest

from tfmplayground.priors.omics import (
    IRRELEVANT,
    MECHANISM,
    MECHANISM_COPY,
    NOISE,
    OmicsPriorConfig,
    sample_dataset,
)


def _datasets(n, seed=0, **config):
    rng = np.random.default_rng(seed)
    return [sample_dataset(rng, OmicsPriorConfig(**config)) for _ in range(n)]


def test_shapes_and_labels_are_consistent():
    for d in _datasets(20):
        n, p = d.X.shape
        assert d.y.shape == (n,)
        assert d.layer.shape == d.is_discrete.shape == d.relevance.shape == d.is_base.shape == (p,)
        assert np.isfinite(d.X).all()
        assert set(np.unique(d.y)) == set(range(d.num_classes))  # contiguous, every class present
        assert 2 <= d.num_classes <= 10


def test_every_layer_has_its_distribution():
    for d in _datasets(30):
        for layer, allowed in (("SNP", {0, 1, 2}), ("CNV", {-2, -1, 0, 1, 2}), ("MUT", {0, 1})):
            cols = d.layer == layer
            if cols.any():
                assert set(np.unique(d.X[:, cols])) <= allowed
                assert d.is_discrete[cols].all()
        meth = d.layer == "METH"
        if meth.any():
            assert (d.X[:, meth] > 0).all() and (d.X[:, meth] < 1).all()
            assert not d.is_discrete[meth].any()


def test_label_always_has_a_mechanism():
    for d in _datasets(30):
        assert (d.relevance[d.is_base] == MECHANISM).sum() >= 1


def test_widened_features_are_copies_or_noise():
    """Widened features never count as mechanism themselves; base variables are never noise."""
    for d in _datasets(20):
        assert not (d.relevance[~d.is_base] == MECHANISM).any()
        assert not (d.relevance[d.is_base] == NOISE).any()
        assert not (d.relevance[d.is_base] == MECHANISM_COPY).any()


def test_widening_can_be_disabled():
    for d in _datasets(10, max_total_features=0):
        assert d.is_base.all()


def test_same_seed_same_dataset():
    a, b = _datasets(1, seed=7)[0], _datasets(1, seed=7)[0]
    assert np.array_equal(a.X, b.X) and np.array_equal(a.y, b.y) and np.array_equal(a.relevance, b.relevance)


def test_mechanism_carries_signal_and_noise_does_not():
    """A random forest on the mechanism variables predicts the label well above chance, and
    clearly better than on the pure-noise features (seeded, so deterministic)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score

    def auroc(d, cols):
        n = len(d.y)
        s = int(0.7 * n)
        perm = np.random.default_rng(0).permutation(n)
        X, y = d.X[perm][:, cols], d.y[perm]
        if len(np.unique(y[:s])) != d.num_classes or len(np.unique(y[s:])) != d.num_classes:
            return None
        proba = RandomForestClassifier(100, random_state=0).fit(X[:s], y[:s]).predict_proba(X[s:])
        return roc_auc_score(y[s:], proba[:, 1]) if d.num_classes == 2 else roc_auc_score(y[s:], proba, multi_class="ovr")

    mech, noise = [], []
    for d in _datasets(15, seed=1):
        is_noise = d.relevance == NOISE
        if not is_noise.any():
            continue
        a_mech, a_noise = auroc(d, d.relevance == MECHANISM), auroc(d, is_noise)
        if a_mech is not None and a_noise is not None:
            mech.append(a_mech)
            noise.append(a_noise)
    assert len(mech) >= 5
    assert np.mean(mech) > 0.7
    assert np.mean(noise) < 0.6
    assert IRRELEVANT < MECHANISM_COPY < MECHANISM  # ordering used when ranking relevance
