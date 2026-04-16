"""Synthetic data generator using decision tree forests.

Vendored from TabForestPFN (den Breejen et al.).
Generates classification data by fitting a DecisionTreeRegressor on random data,
then discretizing predictions via quantile transform and random bucket boundaries.
"""

import numpy as np
from sklearn.preprocessing import QuantileTransformer
from sklearn.tree import DecisionTreeRegressor


def synthetic_dataset_function_forest(
        min_features = 3,
        max_features = 100,
        n_samples = 10000,
        max_classes = 10,
        base_size = 1000,
        n_estimators = 1,
        min_depth = 1,
        max_depth = 25,
        categorical_x = True,
    ):

    n_classes = get_n_classes(max_classes)
    categorical_perc = get_categorical_perc(categorical_x)
    depth = get_depth(min_depth, max_depth)
    n_features = get_n_features(min_features, max_features)
    n_categorical_features = get_n_categorical_features(categorical_perc, n_features)
    n_categorical_classes = get_n_categorical_classes(n_categorical_features)

    x = np.random.normal(size=(base_size, n_features))
    y = np.random.normal(0, 1, size=(base_size,))

    clf = DecisionTreeRegressor(
        max_depth=depth,
        max_features='sqrt',
    )
    clf.fit(x, y)

    x2 = np.random.normal(size=(n_samples, n_features))
    x2 = transform_some_features_to_categorical(x2, n_categorical_features, n_categorical_classes)

    z = clf.predict(x2)
    z = quantile_transform(z)
    z = put_in_buckets(z, n_classes)

    return x2, z


def get_n_classes(max_classes: int) -> int:
    return np.random.randint(2, max_classes, size=1).item()

def get_categorical_perc(categorical_x: bool) -> float:
    if categorical_x:
        return np.random.uniform(0, 1, size=(1,)).item()
    else:
        return 0
    
def get_depth(min_depth: int, max_depth: int) -> int:
    if min_depth == max_depth:
        return min_depth
    else:
        return np.random.randint(min_depth, max_depth, size=1).item()
    
def get_n_features(min_features: int, max_features: int) -> int:
    if min_features == max_features:
        return min_features
    else:
        return np.random.randint(min_features, max_features, size=1).item()
    
def get_n_categorical_features(categorical_perc: float, n_features: int) -> int:
    return int(categorical_perc * (n_features + 1))

def get_n_categorical_classes(n_categorical_features: int) -> np.ndarray:
    return np.random.geometric(p=0.5, size=(n_categorical_features,)) + 1


def transform_some_features_to_categorical(
        x: np.ndarray, 
        n_categorical_features: int, 
        n_categorical_classes: int
    ) -> np.ndarray:

    if n_categorical_features == 0:
        return x
    
    x_index_categorical = np.random.choice(np.arange(x.shape[1]), size=(n_categorical_features,), replace=False)
    x_categorical = x[:, x_index_categorical]

    quantile_transformer = QuantileTransformer(output_distribution='uniform')
    x_categorical = quantile_transformer.fit_transform(x_categorical)

    for i in range(n_categorical_features):
        x_categorical[:, i] = put_in_buckets(x_categorical[:, i], n_categorical_classes[i])

    x[:, x_index_categorical] = x_categorical

    return x


def quantile_transform(z: np.ndarray) -> np.ndarray:
    quantile_transformer = QuantileTransformer(output_distribution='uniform')
    z = quantile_transformer.fit_transform(z.reshape(-1, 1)).flatten()
    return z


def put_in_buckets(z: np.ndarray, n_classes: int) -> np.ndarray:
    buckets = np.random.uniform(0, 1, size=(n_classes-1,))
    buckets.sort()
    buckets = np.hstack([buckets, 1])
    b = np.argmax(z <= buckets[:, None], axis=0)

    return b


def synthetic_dataset_generator_forest(**kwargs):

    while True:
        x, y = synthetic_dataset_function_forest(**kwargs)
        yield x, y
