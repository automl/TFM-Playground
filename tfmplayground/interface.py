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
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder

from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.utils import get_default_device


def init_model_from_state_dict_file(file_path):
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


def to_pandas(x):
    return pd.DataFrame(x) if not isinstance(x, pd.DataFrame) else x


def to_numeric(x):
    return x.apply(pd.to_numeric, errors="coerce").to_numpy()


def get_feature_preprocessor(X):
    X = pd.DataFrame(X)
    num_mask = []
    cat_mask = []
    for col in X:
        unique_non_nan_entries = X[col].dropna().unique()
        if len(unique_non_nan_entries) <= 1:
            num_mask.append(False)
            cat_mask.append(False)
            continue
        non_nan_entries = X[col].notna().sum()
        numeric_entries = pd.to_numeric(X[col], errors="coerce").notna().sum()
        num_mask.append(non_nan_entries == numeric_entries)
        cat_mask.append(non_nan_entries != numeric_entries)

    num_mask = np.array(num_mask)
    cat_mask = np.array(cat_mask)

    num_transformer = Pipeline(
        [
            ("to_pandas", FunctionTransformer(to_pandas)),
            ("to_numeric", FunctionTransformer(to_numeric)),
            (
                "imputer",
                SimpleImputer(strategy="mean", add_indicator=True),
            ),
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
    def __init__(
        self,
        model=None,
        device=None,
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

    def fit(self, X_train, y_train):
        self.feature_preprocessor = get_feature_preprocessor(X_train)
        self.X_train = self.feature_preprocessor.fit_transform(X_train)
        self.y_train = y_train
        self.num_classes = max(set(y_train)) + 1

    def predict(self, X_test):
        predicted_probabilities = self.predict_proba(X_test)
        return predicted_probabilities.argmax(axis=1)

    def predict_proba(self, X_test):
        x = np.concatenate((self.X_train, self.feature_preprocessor.transform(X_test)))
        y = self.y_train
        with torch.no_grad():
            x = torch.from_numpy(x).unsqueeze(0).to(torch.float).to(self.device)
            y = torch.from_numpy(y).unsqueeze(0).to(torch.float).to(self.device)
            out = self.model((x, y), train_test_split_index=len(self.X_train)).squeeze(0)
            out = out[:, : self.num_classes]
            probabilities = F.softmax(out, dim=1)
            return probabilities.to("cpu").numpy()


class NanoTabPFNRegressor:
    def __init__(
        self,
        model=None,
        dist=None,
        device=None,
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

    def fit(self, X_train, y_train):
        self.feature_preprocessor = get_feature_preprocessor(X_train)
        self.X_train = self.feature_preprocessor.fit_transform(X_train)
        self.y_train = y_train

        self.y_train_mean = np.mean(self.y_train)
        self.y_train_std = np.std(self.y_train, ddof=1) + 1e-8
        self.y_train_n = (self.y_train - self.y_train_mean) / self.y_train_std

    def predict(self, X_test):
        X = np.concatenate((self.X_train, self.feature_preprocessor.transform(X_test)))
        y = self.y_train_n

        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device).unsqueeze(0)
            y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device).unsqueeze(0)

            logits = self.model((X_tensor, y_tensor), train_test_split_index=len(self.X_train)).squeeze(0)
            preds_n = self.dist.mean(logits)
            preds = preds_n * self.y_train_std + self.y_train_mean

        return preds.cpu().numpy()
