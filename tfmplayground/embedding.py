from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

from tfmplayground.interface import NanoTabPFNClassifier


def leave_one_fold_out_embeddings(
    estimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_folds: int = 10,
    shuffle: bool = False,
    random_state: int | None = None,
    stratified: bool | None = None,
) -> np.ndarray:
    """Extracts out-of-fold (OOF) embeddings for the training set (Ye et al., 2025).

    Embedding a training row directly is not comparable with test embeddings, because the
    training target is its true label while the test target is a dummy. This partitions the
    training set into ``n_folds``; each fold is embedded as the query using the other folds as
    in-context support, so every training row is embedded in the same (dummy-label) role as a
    test row. The result is aligned to the original row order.

    Because nano is in-context learning (``fit`` only stores the context, it does not train
    weights), the same estimator is reused across folds instead of cloning. The estimator is
    refit on the full training set before returning, so it stays usable afterwards.

    Args:
        estimator: a NanoTabPFNClassifier or NanoTabPFNRegressor.
        X_train, y_train: the training data.
        n_folds: number of folds (>= 2).
        shuffle: whether to shuffle before splitting.
        random_state: seed used only when ``shuffle`` is True.
        stratified: force stratified splitting; when None, inferred from the estimator type
                    (stratified for the classifier, plain KFold for the regressor).

    Returns:
        np.ndarray of shape (n_train, embedding_size), in the original row order.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2 for leave-one-fold-out extraction.")

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)

    if stratified is None:
        stratified = isinstance(estimator, NanoTabPFNClassifier)

    rs = random_state if shuffle else None
    if stratified:
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=shuffle, random_state=rs)
        splits = splitter.split(X_train, y_train)
    else:
        splitter = KFold(n_splits=n_folds, shuffle=shuffle, random_state=rs)
        splits = splitter.split(X_train)

    chunks: list[np.ndarray] = []
    val_indices: list[np.ndarray] = []
    for fold_train_idx, fold_val_idx in splits:
        estimator.fit(X_train[fold_train_idx], y_train[fold_train_idx])
        chunks.append(estimator.get_embeddings(X_train[fold_val_idx]))
        val_indices.append(fold_val_idx)

    oof = np.concatenate(chunks, axis=0)
    order = np.argsort(np.concatenate(val_indices))
    oof = oof[order]

    # Restore the full-data context so the estimator is usable after extraction.
    estimator.fit(X_train, y_train)
    return oof