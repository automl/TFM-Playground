import os

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn.functional as F
from pfns.bar_distribution import FullSupportBarDistribution
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, OrdinalEncoder

from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.normalization import (
    compute_target_stats_numpy,
    denormalize_predictions,
    normalize_targets,
)
from tfmplayground.utils import get_default_device


def init_model_from_state_dict_file(file_path):
    """
    reads model architecture from state dict, instantiates the architecture and loads the weights
    """
    state_dict = torch.load(file_path, map_location=torch.device("cpu"))
    model = NanoTabPFNModel(
        num_attention_heads=state_dict["architecture"]["num_attention_heads"],
        embedding_size=state_dict["architecture"]["embedding_size"],
        mlp_hidden_size=state_dict["architecture"]["mlp_hidden_size"],
        num_layers=state_dict["architecture"]["num_layers"],
        num_outputs=state_dict["architecture"]["num_outputs"],
    )
    model.load_state_dict(state_dict["model"])
    return model


# doing these as lambdas would cause NanoTabPFNClassifier to not be pickle-able,
# which would cause issues if we want to run it inside the tabarena codebase
def to_pandas(x):
    return pd.DataFrame(x) if not isinstance(x, pd.DataFrame) else x


def to_numeric(x):
    return x.apply(pd.to_numeric, errors="coerce").to_numpy()


def get_feature_preprocessor(
    X: np.ndarray | pd.DataFrame,
    categorical_features: list[int] | None = None,
    infer_categorical: bool = True,
) -> ColumnTransformer:
    """
    fits a preprocessor that imputes NaNs, encodes categorical features and removes constant features.
    Columns whose positional index is in categorical_features are forced categorical (at any
    cardinality), overriding the automatic numeric/categorical detection; constant columns are still
    dropped. categorical_features=None keeps the automatic detection only.
    With infer_categorical=False, undeclared columns are treated as numeric and an undeclared
    non-numeric column raises ValueError (every categorical must be declared explicitly).
    """
    X = pd.DataFrame(X)
    num_features = X.shape[1]
    declared_categorical = set(categorical_features or [])
    for index in declared_categorical:
        if index < 0 or index >= num_features:
            raise ValueError(
                f"categorical_features index {index} is out of range for {num_features} features."
            )
    num_mask = []
    cat_mask = []
    for position, col in enumerate(X):
        unique_non_nan_entries = X[col].dropna().unique()
        if len(unique_non_nan_entries) <= 1:
            num_mask.append(False)
            cat_mask.append(False)
            continue
        if position in declared_categorical:
            num_mask.append(False)
            cat_mask.append(True)
            continue
        non_nan_entries = X[col].notna().sum()
        numeric_entries = (
            pd.to_numeric(X[col], errors="coerce").notna().sum()
        )  # in case numeric columns are stored as strings
        is_numeric = non_nan_entries == numeric_entries
        if not infer_categorical and not is_numeric:
            raise ValueError(
                f"Column at index {position} is not numeric and was not declared categorical; "
                f"with infer_categorical=False every categorical column must be listed in "
                f"categorical_features."
            )
        num_mask.append(is_numeric)
        cat_mask.append(not is_numeric)
        # num_mask.append(is_numeric_dtype(X[col]))  # Assumes pandas dtype is correct

    num_mask = np.array(num_mask)
    cat_mask = np.array(cat_mask)

    num_transformer = Pipeline(
        [
            ("to_pandas", FunctionTransformer(to_pandas)),  # to apply pd.to_numeric of pandas
            ("to_numeric", FunctionTransformer(to_numeric)),  # in case numeric columns are stored as strings
            (
                "imputer",
                SimpleImputer(strategy="mean", add_indicator=True),
            ),  # median might be better because of outliers
        ]
    )
    cat_transformer = Pipeline(
        [
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)),
            ("imputer", SimpleImputer(strategy="most_frequent", add_indicator=True)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[("num", num_transformer, num_mask), ("cat", cat_transformer, cat_mask)]
    )
    return preprocessor


class NanoTabPFNClassifier:
    """scikit-learn like interface"""

    def __init__(
        self,
        model: NanoTabPFNModel | str | None = None,
        device: None | str | torch.device = None,
        num_mem_chunks: int = 8,
        categorical_features: list[int] | None = None,
        infer_categorical: bool = True,
    ):
        if device is None:
            device = get_default_device()
        if model is None:
            model = "checkpoints/nanotabpfn.pth"
            if not os.path.isfile(model):
                os.makedirs("checkpoints", exist_ok=True)
                print("No cached model found, downloading model checkpoint.")
                response = requests.get(
                    "https://ml.informatik.uni-freiburg.de/research-artifacts/pfefferle/TFM-Playground/nanotabpfn_classifier.pth"
                )
                with open(model, "wb") as f:
                    f.write(response.content)
        if isinstance(model, str):
            model = init_model_from_state_dict_file(model)
        self.model = model.to(device)
        self.device = device
        self.num_mem_chunks = num_mem_chunks
        self.categorical_features = categorical_features
        self.infer_categorical = infer_categorical

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """stores X_train, label-encodes the targets to contiguous indices 0..num_classes-1
        (so arbitrary labels, e.g. non-contiguous integers or strings, are supported), and
        keeps the original labels in classes_ for decoding predictions"""
        self.feature_preprocessor = get_feature_preprocessor(
            X_train, categorical_features=self.categorical_features, infer_categorical=self.infer_categorical
        )
        self.X_train = self.feature_preprocessor.fit_transform(X_train)
        self.label_encoder = LabelEncoder()
        self.y_train = self.label_encoder.fit_transform(y_train)
        self.classes_ = self.label_encoder.classes_
        self.num_classes = len(self.classes_)
        if self.num_classes > self.model.num_outputs:
            raise ValueError(
                f"This model supports at most {self.model.num_outputs} classes, "
                f"but the training data has {self.num_classes}."
            )
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """calls predict_proba, picks the highest-probability class for each datapoint,
        and maps it back to the original label"""
        predicted_probabilities = self.predict_proba(X_test)
        encoded_predictions = predicted_probabilities.argmax(axis=1)
        return self.label_encoder.inverse_transform(encoded_predictions)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        creates (x,y), runs it through our PyTorch Model, cuts off the classes that didn't appear in the training data
        and applies softmax to get the probabilities
        """
        x = np.concatenate((self.X_train, self.feature_preprocessor.transform(X_test)))
        y = self.y_train
        with torch.no_grad():
            x = torch.from_numpy(x).unsqueeze(0).to(torch.float).to(self.device)  # introduce batch size 1
            y = torch.from_numpy(y).unsqueeze(0).to(torch.float).to(self.device)
            out = self.model(
                (x, y), train_test_split_index=len(self.X_train), num_mem_chunks=self.num_mem_chunks
            ).squeeze(0)  # remove batch size 1
            # our pretrained classifier supports up to num_outputs classes, if the dataset has less we cut off the rest
            out = out[:, : self.num_classes]
            # apply softmax to get a probability distribution
            probabilities = F.softmax(out, dim=1)
            return probabilities.to("cpu").numpy()


class NanoTabPFNRegressor:
    """scikit-learn like interface"""

    def __init__(
        self,
        model: NanoTabPFNModel | str | None = None,
        dist: FullSupportBarDistribution | str | None = None,
        device: str | torch.device | None = None,
        num_mem_chunks: int = 8,
        categorical_features: list[int] | None = None,
        infer_categorical: bool = True,
    ):
        if device is None:
            device = get_default_device()
        if model is None:
            os.makedirs("checkpoints", exist_ok=True)
            model = "checkpoints/nanotabpfn_regressor.pth"
            dist = "checkpoints/nanotabpfn_regressor_buckets.pth"
            if not os.path.isfile(model):
                print("No cached model found, downloading model checkpoint.")
                response = requests.get(
                    "https://ml.informatik.uni-freiburg.de/research-artifacts/pfefferle/TFM-Playground/nanotabpfn_regressor.pth"
                )
                with open(model, "wb") as f:
                    f.write(response.content)
            if not os.path.isfile(dist):
                print("No cached bucket edges found, downloading bucket edges.")
                response = requests.get(
                    "https://ml.informatik.uni-freiburg.de/research-artifacts/pfefferle/TFM-Playground/nanotabpfn_regressor_buckets.pth"
                )
                with open(dist, "wb") as f:
                    f.write(response.content)
        if isinstance(model, str):
            model = init_model_from_state_dict_file(model)

        if isinstance(dist, str):
            bucket_edges = torch.load(dist, map_location=device)
            dist = FullSupportBarDistribution(bucket_edges).float()

        self.model = model.to(device)
        self.device = device
        self.dist = dist
        self.num_mem_chunks = num_mem_chunks
        self.categorical_features = categorical_features
        self.infer_categorical = infer_categorical

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Stores X_train and y_train for later use.
        Computes target normalization.
        """
        self.feature_preprocessor = get_feature_preprocessor(
            X_train, categorical_features=self.categorical_features, infer_categorical=self.infer_categorical
        )
        self.X_train = self.feature_preprocessor.fit_transform(X_train)
        self.y_train = y_train

        self.y_train_mean, self.y_train_std = compute_target_stats_numpy(self.y_train)
        self.y_train_n = normalize_targets(self.y_train, self.y_train_mean, self.y_train_std)
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Performs in-context learning using X_train and y_train.
        Predicts the means of the output distributions for X_test.
        Renormalizes the predictions back to the original target scale.
        """
        X = np.concatenate((self.X_train, self.feature_preprocessor.transform(X_test)))
        y = self.y_train_n

        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device).unsqueeze(0)
            y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device).unsqueeze(0)

            logits = self.model(
                (X_tensor, y_tensor), train_test_split_index=len(self.X_train), num_mem_chunks=self.num_mem_chunks
            ).squeeze(0)
            preds_n = self.dist.mean(logits)
            preds = denormalize_predictions(preds_n, self.y_train_mean, self.y_train_std)

        return preds.cpu().numpy()