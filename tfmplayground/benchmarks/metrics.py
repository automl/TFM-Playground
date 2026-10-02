"""Metrics, computed exactly as TabPFN-Wide does (analysis/multiomics_feature_reduction.py)."""

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score


def roc_auc(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Binary: AUROC of the probability of class 1. Multiclass: one-vs-rest, macro average."""
    if proba.shape[1] == 2:
        return float(roc_auc_score(y_true, proba[:, 1]))
    return float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=np.arange(proba.shape[1])))


def accuracy(y_true: np.ndarray, proba: np.ndarray) -> float:
    return float(accuracy_score(y_true, proba.argmax(axis=1)))
