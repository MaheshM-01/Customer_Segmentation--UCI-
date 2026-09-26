"""Unit tests for custom transformers (PercentileClipper)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from src.features.transformers import PercentileClipper


def test_percentile_clipper_initialization():
    """Test valid and invalid initialization quantiles."""
    clipper = PercentileClipper(lower_quantile=0.05, upper_quantile=0.95)
    assert clipper.lower_quantile == 0.05
    assert clipper.upper_quantile == 0.95

    with pytest.raises(ValueError):
        PercentileClipper(lower_quantile=0.9, upper_quantile=0.1)

    with pytest.raises(ValueError):
        PercentileClipper(lower_quantile=-0.1, upper_quantile=0.9)


def test_percentile_clipper_fit_and_transform_numpy():
    """Test clipping functionality on synthetic NumPy array with extreme outliers."""
    np.random.seed(42)
    raw_data = np.linspace(0, 100, 100).reshape(-1, 1)
    raw_data[0] = -1000.0  # extreme low outlier
    raw_data[-1] = 1000.0  # extreme high outlier

    clipper = PercentileClipper(lower_quantile=0.05, upper_quantile=0.95)
    clipper.fit(raw_data)

    assert hasattr(clipper, "lower_bounds_")
    assert hasattr(clipper, "upper_bounds_")
    assert clipper.lower_bounds_[0] > -1000.0
    assert clipper.upper_bounds_[0] < 1000.0

    transformed = clipper.transform(raw_data)
    assert transformed.min() >= clipper.lower_bounds_[0]
    assert transformed.max() <= clipper.upper_bounds_[0]


def test_percentile_clipper_pandas_dataframe():
    """Test clipping on pandas DataFrame preserving feature names."""
    df = pd.DataFrame(
        {
            "FeatureA": [1.0, 2.0, 3.0, 4.0, 1000.0],
            "FeatureB": [-500.0, 10.0, 20.0, 30.0, 40.0],
        }
    )

    clipper = PercentileClipper(lower_quantile=0.10, upper_quantile=0.90)
    transformed = clipper.fit_transform(df)

    assert transformed.shape == df.shape
    assert clipper.n_features_in_ == 2
    assert list(clipper.feature_names_in_) == ["FeatureA", "FeatureB"]

    names_out = clipper.get_feature_names_out()
    assert list(names_out) == ["FeatureA", "FeatureB"]


def test_percentile_clipper_unfitted_raises_error():
    """Test that transforming before fitting raises NotFittedError."""
    clipper = PercentileClipper()
    data = np.array([[1.0, 2.0], [3.0, 4.0]])

    with pytest.raises(NotFittedError):
        clipper.transform(data)


def test_percentile_clipper_dimension_mismatch():
    """Test that passing wrong feature count during transform raises ValueError."""
    train_data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    test_data = np.array([[1.0], [2.0]])

    clipper = PercentileClipper()
    clipper.fit(train_data)

    with pytest.raises(ValueError):
        clipper.transform(test_data)
