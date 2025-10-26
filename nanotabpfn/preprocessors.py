from abc import ABC, abstractmethod
from typing import Any, override
import numpy as np
from sklearn.pipeline import FunctionTransformer


class Preprocessor(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray) -> None:
        pass

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        pass


class IdentityPreprocessor(Preprocessor):
    @override
    def fit(self, X: np.ndarray) -> None:
        pass

    @override
    def transform(self, X: np.ndarray) -> np.ndarray:
        return X


class SklearnPreprocessor(Preprocessor):
    def __init__(self, transformer: Any):
        self.transformer = transformer

    @override
    def fit(self, X: np.ndarray) -> None:
        self.transformer.fit(X)

    @override
    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.nan_to_num(self.transformer.transform(X))


class LogPreprocessor(SklearnPreprocessor):
    """log1p for right-skewed, non-negative features."""
    @override
    def __init__(self):
        super().__init__(FunctionTransformer(func=np.log1p, feature_names_out="one-to-one"))


class AsinhPreprocessor(SklearnPreprocessor):
    """Signed log-like transform: linear near 0, log for large |x|; works with negatives."""
    @override
    def __init__(self):
        super().__init__(FunctionTransformer(func=np.arcsinh, feature_names_out="one-to-one"))


def sample_preprocessors(num_preprocessors: int, X_train: np.ndarray) -> list[Preprocessor]:
    """
    Return a *good* set of preprocessors for tabular transformers.
    """
    picks: list[Preprocessor] = []

    picks.append(AsinhPreprocessor())

    # For strictly non-negative, log1p is fine
    if np.nanmin(X_train) >= 0:
        picks.append(LogPreprocessor())

    return picks[:num_preprocessors]