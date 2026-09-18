"""Characterization tests for get_feature_preprocessor.

These pin the CURRENT behavior of the feature preprocessor before we change it
(injectable categoricals + better detection). They document its quirks — most
importantly that integer-coded categoricals are treated as numeric — so any
behavior change becomes a conscious, visible test update.
"""

import numpy as np
import pandas as pd

from tfmplayground.interface import get_feature_preprocessor


def _fit_transform(X):
    pre = get_feature_preprocessor(X)
    return np.asarray(pre.fit_transform(X), dtype=float)


def test_mixed_input_current_output():
    """One representative mixed frame, pinning detection + encoding + mean
    imputation + missing-indicator + column order + constant-column drop at once.

    Output column order is: numeric (imputed), then numeric missing-indicators,
    then categorical (ordinal-encoded), then categorical missing-indicators.
    """
    X = pd.DataFrame(
        {
            0: [1.0, 2.0, 3.0, 4.0],          # clean numeric
            1: [10.0, np.nan, 30.0, np.nan],  # numeric with missing
            2: ["a", "b", "a", "c"],          # string categorical
            3: [7.0, 7.0, 7.0, 7.0],          # constant -> dropped
        }
    )

    out = _fit_transform(X)

    expected = np.array(
        [
            [1.0, 10.0, 0.0, 0.0],  # col0, col1 (mean-imputed), col1-missing-indicator, col2 (a=0)
            [2.0, 20.0, 1.0, 1.0],  # col1 imputed to mean 20, indicator=1, col2 (b=1)
            [3.0, 30.0, 0.0, 0.0],
            [4.0, 20.0, 1.0, 2.0],  # col2 (c=2)
        ]
    )
    assert out.shape == (4, 4)
    assert np.allclose(out, expected)


def test_constant_column_is_dropped():
    """A column with <=1 distinct non-NaN value carries no information and is
    removed entirely (neither numeric nor categorical).
    """
    X = pd.DataFrame({0: [1.0, 2.0, 3.0], 1: [5.0, 5.0, 5.0]})  # col1 constant

    out = _fit_transform(X)

    assert out.shape == (3, 1)  # only col0 survives
    assert np.allclose(out, np.array([[1.0], [2.0], [3.0]]))


def test_numeric_stored_as_string_is_treated_as_numeric():
    """Values that all parse as numbers are numeric even if stored as strings."""
    X = pd.DataFrame({0: ["1", "2", "3"]})

    out = _fit_transform(X)

    assert np.allclose(out, np.array([[1.0], [2.0], [3.0]]))


def test_integer_coded_categorical_is_currently_treated_as_numeric():
    """QUIRK we intend to change: a categorical encoded as integers (all parse as
    numbers) is detected as numeric — there is no cardinality check — so the codes
    pass through unchanged instead of being re-encoded. Pinned so the upcoming
    detection change is a conscious update.
    """
    X = np.array([[10], [20], [30], [10], [20], [30], [10]])

    out = _fit_transform(X)

    # Numeric path: values pass through unchanged (not re-encoded to 0..k-1).
    assert np.allclose(out, np.array([[10.0], [20.0], [30.0], [10.0], [20.0], [30.0], [10.0]]))