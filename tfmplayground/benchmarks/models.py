"""Models that can be evaluated, by name. Add a model by adding an entry to make_model()."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_model(name: str, device: str = "cuda"):
    """'random_forest' and 'logreg' are baselines; any other name is taken as the path of one of our
    checkpoints and evaluated with NanoTabPFNClassifier, without ensembling (one forward pass), as
    TabPFN-Wide evaluates its own model."""
    if name == "random_forest":  # as TabPFN-Wide: default hyperparameters, NaN imputed with the mode
        return make_pipeline(SimpleImputer(strategy="most_frequent"), RandomForestClassifier(n_jobs=-1))
    if name == "logreg":
        return make_pipeline(SimpleImputer(strategy="most_frequent"), StandardScaler(), LogisticRegression(max_iter=5000))
    from tfmplayground.interface import NanoTabPFNClassifier  # needs torch, only imported here

    return NanoTabPFNClassifier(model=name, device=device)


def predict_proba(model, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    model.fit(X_train, y_train)
    return np.asarray(model.predict_proba(X_test))
