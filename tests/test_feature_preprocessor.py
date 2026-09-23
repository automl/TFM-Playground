"""Characterization tests for get_feature_preprocessor.

These pin the behavior of the feature preprocessor. Most tests characterize
existing behavior; the cardinality-detection block pins the new rule that
low-cardinality integer-coded columns are treated as categorical (with a
sample-count floor so tiny datasets are unaffected).
"""

import numpy as np
import pandas as pd
import pytest
import torch

from tfmplayground.interface import NanoTabPFNClassifier, get_feature_preprocessor


def _fit_transform(X):
    pre = get_feature_preprocessor(X)
    return np.asarray(pre.fit_transform(X), dtype=float)


def _column(values, repeats):
    """Build a single-column 2D array by tiling `values` `repeats` times.

    The resulting column has len(set(values)) distinct values and
    len(values) * repeats rows, which lets each test state cardinality and
    row count independently and read at a glance.
    """
    return np.array([[v] for v in list(values) * repeats])


def test_mixed_input_no_imputation_and_no_indicator_columns():
    """One representative mixed frame: detection + encoding + column order +
    constant-column drop, now WITHOUT imputation and WITHOUT indicator columns.
    Missing entries pass through as NaN for the model to handle (mean + indicator).

    Output column order is: numeric columns, then categorical (ordinal-encoded).
    """
    X = pd.DataFrame(
        {
            0: [1.0, 2.0, 3.0, 4.0],          # clean numeric
            1: [10.0, np.nan, 30.0, np.nan],  # numeric with missing -> NaN passes through
            2: ["a", "b", "a", "c"],          # string categorical
            3: [7.0, 7.0, 7.0, 7.0],          # constant -> dropped
        }
    )

    out = _fit_transform(X)

    assert out.shape == (4, 3)  # col0, col1 (with NaNs), col2; no indicator columns
    # numeric columns pass through; col1 keeps its holes as NaN
    assert np.allclose(out[:, 0], [1.0, 2.0, 3.0, 4.0])
    assert np.isnan(out[[1, 3], 1]).all()
    assert np.allclose(out[[0, 2], 1], [10.0, 30.0])
    # categorical ordinal-encoded: a=0, b=1, c=2
    assert np.allclose(out[:, 2], [0.0, 1.0, 0.0, 2.0])


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


def test_integer_coded_low_cardinality_is_detected_categorical():
    """Integer-coded column with few distinct values and enough rows is detected
    as categorical and re-encoded to 0..k-1, instead of passing its magnitudes
    through as numeric. n=30 clears the sample floor; 3 distinct values are under
    the cardinality threshold. (This inverts the previously-pinned quirk.)
    """
    X = _column([10, 20, 30], repeats=10)  # 3 distinct values, 30 rows

    out = _fit_transform(X)

    # Categorical path: 10->0, 20->1, 30->2 (magnitudes gone).
    assert out.shape == (30, 1)
    assert np.allclose(out[:6].ravel(), [0, 1, 2, 0, 1, 2])


def test_low_cardinality_below_sample_floor_stays_numeric():
    """The sample floor protects tiny datasets: with too few rows, low cardinality
    is not evidence enough to call a numeric column categorical, so it stays numeric
    and its magnitudes pass through. Same values as the test above, far fewer rows.
    """
    X = _column([10, 20, 30], repeats=3)  # 3 distinct values, only 9 rows (< floor)

    out = _fit_transform(X)

    assert np.allclose(out.ravel(), [10, 20, 30] * 3)


def test_high_cardinality_numeric_stays_numeric():
    """Above the cardinality threshold a numeric column keeps its magnitudes even
    with plenty of rows: genuine numeric variety must not be turned categorical.
    """
    X = _column(range(10, 160, 10), repeats=3)  # 15 distinct values, 45 rows

    out = _fit_transform(X)

    assert np.allclose(out[:15].ravel(), list(range(10, 160, 10)))


def test_cardinality_threshold_boundary():
    """Pins the exact <=10 boundary and guards against an off-by-one (< vs <=):
    exactly 10 distinct values -> categorical; exactly 11 -> numeric. Both clear
    the sample floor, so cardinality alone decides.
    """
    ten = _column(range(10, 101, 10), repeats=3)     # 10 distinct values, 30 rows
    eleven = _column(range(10, 111, 10), repeats=3)  # 11 distinct values, 33 rows

    out_ten = _fit_transform(ten)
    out_eleven = _fit_transform(eleven)

    # 10 distinct -> categorical (re-encoded 0..9).
    assert np.allclose(out_ten[:10].ravel(), list(range(10)))
    # 11 distinct -> numeric (magnitudes pass through).
    assert np.allclose(out_eleven[:11].ravel(), list(range(10, 111, 10)))


def test_declared_categorical_overrides_numeric_detection():
    """A column that would be auto-detected as numeric (integer-coded) is forced
    categorical when declared, so its codes are ordinal-encoded to 0..k-1 instead
    of passing through as magnitudes.
    """
    X = np.array([[10], [20], [30], [10], [20], [30], [10]])

    out = np.asarray(
        get_feature_preprocessor(X, categorical_features=[0]).fit_transform(X), dtype=float
    )

    assert np.allclose(out.ravel(), [0, 1, 2, 0, 1, 2, 0])


def test_declared_constant_categorical_is_still_dropped():
    """Declaring a column categorical does not save a constant column: it carries
    no information and is still dropped.
    """
    X = pd.DataFrame({0: [1.0, 2.0, 3.0], 1: [5.0, 5.0, 5.0]})  # col1 constant

    out = np.asarray(
        get_feature_preprocessor(X, categorical_features=[1]).fit_transform(X), dtype=float
    )

    assert out.shape == (3, 1)  # only col0 survives
    assert np.allclose(out, np.array([[1.0], [2.0], [3.0]]))


def test_out_of_range_categorical_index_raises():
    """An index that doesn't correspond to a column is a clear error."""
    X = pd.DataFrame({0: [1, 2, 3]})

    with pytest.raises(ValueError):
        get_feature_preprocessor(X, categorical_features=[5])


def test_categorical_features_none_matches_default():
    """categorical_features=None preserves the default behavior exactly."""
    X = np.array([[10], [20], [30], [10], [20], [30], [10]])

    with_none = np.asarray(
        get_feature_preprocessor(X, categorical_features=None).fit_transform(X), dtype=float
    )
    default = np.asarray(get_feature_preprocessor(X).fit_transform(X), dtype=float)

    assert np.allclose(with_none, default)


def test_classifier_threads_categorical_features_to_preprocessor():
    """The constructor's categorical_features reaches get_feature_preprocessor via fit."""
    model = torch.nn.Identity()
    model.num_outputs = 10
    clf = NanoTabPFNClassifier(model=model, device="cpu", categorical_features=[0])

    X = np.array([[10], [20], [30], [10], [20], [30]])  # integer-coded, declared categorical
    y = np.array([0, 1, 0, 1, 0, 1])
    clf.fit(X, y)

    # X_train reflects ordinal encoding (0, 1, 2), not the raw integers.
    assert np.allclose(np.asarray(clf.X_train, dtype=float).ravel(), [0, 1, 2, 0, 1, 2])


def test_strict_mode_undeclared_non_numeric_column_raises():
    """With infer_categorical=False, a non-numeric column that was not declared
    categorical is an error, instead of being silently inferred.
    """
    X = pd.DataFrame({0: [1.0, 2.0, 3.0, 4.0], 1: ["a", "b", "a", "c"]})  # col1 is text

    with pytest.raises(ValueError):
        get_feature_preprocessor(X, infer_categorical=False)


def test_strict_mode_declared_categorical_is_encoded():
    """In strict mode a declared categorical still works (no inference needed):
    it is ordinal-encoded normally.
    """
    X = pd.DataFrame({0: [1.0, 2.0, 3.0, 4.0], 1: ["a", "b", "a", "c"]})

    out = np.asarray(
        get_feature_preprocessor(X, categorical_features=[1], infer_categorical=False).fit_transform(X),
        dtype=float,
    )

    assert np.allclose(out, np.array([[1.0, 0.0], [2.0, 1.0], [3.0, 0.0], [4.0, 2.0]]))


def test_strict_mode_all_numeric_columns_pass_through():
    """In strict mode fully-numeric undeclared columns are treated as numeric."""
    X = pd.DataFrame({0: [1.0, 2.0, 3.0]})

    out = np.asarray(
        get_feature_preprocessor(X, infer_categorical=False).fit_transform(X), dtype=float
    )

    assert np.allclose(out.ravel(), [1.0, 2.0, 3.0])


def test_classifier_threads_infer_categorical_to_preprocessor():
    """The constructor's infer_categorical reaches get_feature_preprocessor via fit:
    strict mode + an undeclared text column raises during fit.
    """
    model = torch.nn.Identity()
    model.num_outputs = 10
    clf = NanoTabPFNClassifier(model=model, device="cpu", infer_categorical=False)

    X = np.array([["a"], ["b"], ["a"], ["b"]], dtype=object)  # text, undeclared
    y = np.array([0, 1, 0, 1])

    with pytest.raises(ValueError):
        clf.fit(X, y)


def test_classifier_threads_max_unique_for_categorical_to_preprocessor():
    """The constructor's max_unique_for_categorical reaches get_feature_preprocessor via fit:
    a stricter threshold pushes a 3-value column onto the numeric side.
    """
    model = torch.nn.Identity()
    model.num_outputs = 10
    # Default (<=10) would detect this as categorical; max_unique=2 keeps it numeric.
    clf = NanoTabPFNClassifier(model=model, device="cpu", max_unique_for_categorical=2)

    X = _column([10, 20, 30], repeats=10)  # 3 distinct values, 30 rows
    y = np.tile([0, 1, 2], 10)
    clf.fit(X, y)

    # Numeric path: magnitudes preserved (not re-encoded to 0, 1, 2).
    assert np.allclose(np.asarray(clf.X_train, dtype=float)[:3].ravel(), [10, 20, 30])


def test_classifier_threads_min_samples_for_categorical_inference_to_preprocessor():
    """The constructor's min_samples_for_categorical_inference reaches the preprocessor via fit:
    raising the floor above n keeps a low-cardinality column numeric.
    """
    model = torch.nn.Identity()
    model.num_outputs = 10
    # Default floor (30) would detect this as categorical at n=30; floor=1000 keeps it numeric.
    clf = NanoTabPFNClassifier(model=model, device="cpu", min_samples_for_categorical_inference=1000)

    X = _column([10, 20, 30], repeats=10)  # 3 distinct values, 30 rows
    y = np.tile([0, 1, 2], 10)
    clf.fit(X, y)

    assert np.allclose(np.asarray(clf.X_train, dtype=float)[:3].ravel(), [10, 20, 30])


def test_numeric_missing_is_passed_through_as_nan():
    """The preprocessor no longer imputes numeric holes: the NaN passes through (the
    model does mean-imputation + indicator) and no extra indicator column is added.
    """
    X = pd.DataFrame({0: [10.0, np.nan, 30.0, 40.0]})

    out = _fit_transform(X)

    assert out.shape == (4, 1)                       # no extra indicator column
    assert np.isnan(out[1, 0])                       # the hole is preserved as NaN
    assert np.allclose(out[[0, 2, 3], 0], [10.0, 30.0, 40.0])


def test_categorical_missing_is_passed_through_as_nan():
    """A categorical column is ordinal-encoded and a missing category stays NaN
    (no most-frequent imputation, no extra indicator column).
    """
    X = pd.DataFrame({0: ["a", "b", np.nan, "a", "c"]})

    out = _fit_transform(X)

    assert out.shape == (5, 1)
    assert np.isnan(out[2, 0])                        # hole preserved as NaN
    assert np.allclose(out[[0, 1, 3, 4], 0], [0.0, 1.0, 0.0, 2.0])  # a=0, b=1, c=2