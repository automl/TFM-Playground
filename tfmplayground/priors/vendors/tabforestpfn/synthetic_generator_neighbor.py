"""Synthetic data generator using K-nearest neighbors.

Vendored from TabForestPFN (den Breejen et al.).
Generates classification data by fitting a KNeighborsRegressor on random anchor
points, then discretizing predictions via quantile transform and random buckets.
"""

import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import QuantileTransformer
import warnings
warnings.filterwarnings('ignore', message='n_quantiles.*is set to n_samples')


def synthetic_dataset_function_neighbor(
        min_features = 3,
        max_features = 100,
        n_samples = 10000,
        max_classes = 10,
        min_neighbors = 2,
        max_neighbors = 2048,
    ):

    n_classes = np.random.randint(2, max_classes, size=1).item()
    categorical_perc = np.random.uniform(0, 1, size=(1,)).item()
    
    if min_features == max_features:
        n_features = min_features
    else:
        n_features = np.random.randint(min_features, max_features, size=1).item()

    if min_neighbors == max_neighbors:
        n_neighbors = min_neighbors
    else:
        n_neighbors = np.random.randint(min_neighbors, max_neighbors, size=1).item()

    n_categorical_features = int(categorical_perc * (n_features + 1))
    n_categorical_classes = np.random.geometric(p=0.5, size=(n_categorical_features,)) + 1

    x_neighbors = np.random.normal(size=(n_neighbors, n_features))
    y_neighbors = np.random.normal(size=(n_neighbors,))

    nn = KNeighborsRegressor(
        n_neighbors=1,
    )
    nn.fit(x_neighbors, y_neighbors)

    x = np.random.normal(size=(n_samples, n_features))

    if n_categorical_features > 0:
        x_categorical = x[:, :n_categorical_features]
        x_numerical = x[:, n_categorical_features:]

        quantile_transformer = QuantileTransformer(output_distribution='uniform')
        x_categorical = quantile_transformer.fit_transform(x_categorical)

        for i in range(n_categorical_features):
            n_categorical_class = n_categorical_classes[i]
            buckets = np.random.uniform(0, 1, size=(n_categorical_class-1,))
            buckets.sort()
            buckets = np.hstack([buckets, 1])
            b = np.argmax(x_categorical[:, i] < buckets[:, None], axis=0)
            x_categorical[:, i] = b

        x = np.hstack([x_categorical, x_numerical])

    z = nn.predict(x)

    quantile_transformer = QuantileTransformer()
    z = quantile_transformer.fit_transform(z.reshape(-1, 1)).flatten()

    buckets = np.random.uniform(0, 1, size=(n_classes-1,))
    buckets.sort()
    buckets = np.hstack([buckets, 1])
    b = np.argmax(z <= buckets[:, None], axis=0)

    return x, b


def synthetic_dataset_generator_neighbor(
        min_features = 3,
        max_features = 100,
        n_samples = 10000,
        max_classes = 10,
        min_neighbors = 2,
        max_neighbors = 2048,
    ):

    while True:
        yield synthetic_dataset_function_neighbor(
            min_features = min_features,
            max_features = max_features,
            n_samples = n_samples,
            max_classes = max_classes,
            min_neighbors = min_neighbors,
            max_neighbors = max_neighbors,
        )
