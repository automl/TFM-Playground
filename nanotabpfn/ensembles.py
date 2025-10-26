from typing import Any
import numpy as np
import torch.nn.functional as F

from nanotabpfn.interface import get_feature_preprocessor
from nanotabpfn.preprocessors import IdentityPreprocessor, Preprocessor, sample_preprocessors

class EnsembleClassifer:
    def __init__(self, classifier: Any, num_preprocessors: int = 4, preprocess_features: bool = True):
        self.classifier = classifier
        self.model = self.classifier.model
        self.num_preprocessors = num_preprocessors
        self.preprocess_features = preprocess_features

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """ stores X_train and y_train for later use, also computes the highest class number occuring in num_classes """
        self.X_train = X_train
        if self.preprocess_features:
            self.feature_preprocessor = get_feature_preprocessor(X_train)
            self.X_train: np.ndarray = self.feature_preprocessor.fit_transform(self.X_train) # type:ignore
        self.y_train = y_train
        self.num_classes = max(set(y_train))+1
        self.preprocessors: list[Preprocessor] = [IdentityPreprocessor()] + sample_preprocessors(self.num_preprocessors, self.X_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """ calls predit_proba and picks the class with the highest probability for each datapoint """
        predicted_probabilities = self.predict_proba(X_test)
        return predicted_probabilities.argmax(axis=1)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        if self.preprocess_features:
            X_test = self.feature_preprocessor.transform(X_test) # type:ignore
        all_probabilities = []
        for preprocessor in self.preprocessors:
            preprocessor.fit(self.X_train)
            X_train_preprocessed = preprocessor.transform(self.X_train)
            X_test_preprocessed = preprocessor.transform(X_test)
            self.classifier.fit(X_train_preprocessed, self.y_train)
            all_probabilities.append(self.classifier.predict_proba(X_test_preprocessed))
        return np.average(np.stack(all_probabilities, axis=0), axis=0)