import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, OrdinalEncoder, StandardScaler

from tfmplayground.models import TabularFoundationModel
from tfmplayground.utils import get_default_device, make_regression_decoder


# doing these as lambdas would cause TabularClassifier to not be pickle-able,
# which would cause issues if we want to run it inside the tabarena codebase
def to_pandas(x):
    return pd.DataFrame(x) if not isinstance(x, pd.DataFrame) else x


def to_numeric(x):
    return x.apply(pd.to_numeric, errors="coerce").to_numpy()


def get_feature_preprocessor(X: np.ndarray | pd.DataFrame) -> Pipeline:
    """
    fits a preprocessor that imputes NaNs, encodes categorical features and removes constant features

    it also scales every column, because the priors give standard-scaled features
    """
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
        numeric_entries = (
            pd.to_numeric(X[col], errors="coerce").notna().sum()
        )  # in case numeric columns are stored as strings
        num_mask.append(non_nan_entries == numeric_entries)
        cat_mask.append(non_nan_entries != numeric_entries)
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

    transformers = [
        ("num", num_transformer, num_mask),
        ("cat", cat_transformer, cat_mask),
    ]
    steps = [
        ("columns", ColumnTransformer(transformers)),
        ("scale", StandardScaler()),
    ]
    return Pipeline(steps)


class TabularClassifier:
    """scikit-learn like interface"""

    def __init__(
        self,
        model: TabularFoundationModel,
        device: None | str | torch.device = None,
    ):
        if device is None:
            device = get_default_device()
        self.model = model.to(device)
        self.device = device

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """stores X_train and y_train for later use, also computes the highest class number occuring in num_classes"""
        self.feature_preprocessor = get_feature_preprocessor(X_train)
        self.X_train = self.feature_preprocessor.fit_transform(X_train)
        self.label_encoder = LabelEncoder()
        self.y_train = self.label_encoder.fit_transform(y_train)
        self.num_classes = len(self.label_encoder.classes_)
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """calls predit_proba and picks the class with the highest probability for each datapoint"""
        predicted_probabilities = self.predict_proba(X_test)
        return self.label_encoder.inverse_transform(predicted_probabilities.argmax(axis=1))

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        creates (x,y), runs it through our PyTorch Model, cuts off the classes that didn't appear in the training data
        and applies softmax to get the probabilities
        """
        X_test = self.feature_preprocessor.transform(X_test)
        self.model.eval()
        with torch.no_grad():
            # introduce batch size 1
            X_train = torch.from_numpy(self.X_train).unsqueeze(0).to(torch.float).to(self.device)
            X_test = torch.from_numpy(X_test).unsqueeze(0).to(torch.float).to(self.device)
            y_train = torch.from_numpy(self.y_train).unsqueeze(0).to(torch.float).to(self.device)
            out = self.model(X_train, y_train, X_test).squeeze(0)  # remove batch size 1
            if out.shape[-1] < self.num_classes:
                raise ValueError(f"model has {out.shape[-1]} outputs, data has {self.num_classes} classes")
            # our pretrained classifier supports up to num_outputs classes, if the dataset has less we cut off the rest
            out = out[:, : self.num_classes]
            # apply softmax to get a probability distribution
            probabilities = F.softmax(out, dim=1)
            return probabilities.to("cpu").numpy()


class TabularRegressor:
    """scikit-learn like interface"""

    def __init__(
        self,
        model: TabularFoundationModel,
        device: str | torch.device | None = None,
    ):
        if device is None:
            device = get_default_device()
        self.model = model.to(device)
        self.device = device
        self.dist = make_regression_decoder(model)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Stores X_train and y_train for later use.
        Computes target normalization.
        """
        self.feature_preprocessor = get_feature_preprocessor(X_train)
        self.X_train = self.feature_preprocessor.fit_transform(X_train)
        self.y_train = y_train

        self.y_train_mean = np.mean(self.y_train)
        self.y_train_std = np.std(self.y_train, ddof=1) + 1e-8
        self.y_train_n = (self.y_train - self.y_train_mean) / self.y_train_std
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Performs in-context learning using X_train and y_train.
        """
        X_test = self.feature_preprocessor.transform(X_test)
        self.model.eval()
        with torch.no_grad():
            X_train = torch.from_numpy(self.X_train).unsqueeze(0).to(torch.float).to(self.device)
            X_test = torch.from_numpy(X_test).unsqueeze(0).to(torch.float).to(self.device)
            y_train = torch.from_numpy(self.y_train_n).unsqueeze(0).to(torch.float).to(self.device)

            logits = self.model(X_train, y_train, X_test).squeeze(0)
            preds_n = self.dist.mean(logits)
            preds = preds_n * self.y_train_std + self.y_train_mean

        return preds.cpu().numpy()
