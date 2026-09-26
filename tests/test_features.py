"""Unit tests for feature engineering and transformation pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.rfm_builder import build_rfm_features
from src.features.pipeline import fit_transform_features, transform_features, get_model_features


@pytest.fixture
def sample_cleaned_transactions() -> pd.DataFrame:
    """Mock cleaned retail transactions for 3 customers."""
    return pd.DataFrame(
        {
            "InvoiceNo": ["536365", "536366", "536367", "536368", "536369"],
            "StockCode": ["85123A", "71053", "84406B", "84029G", "85123A"],
            "Description": ["Prod 1", "Prod 2", "Prod 3", "Prod 4", "Prod 1"],
            "Quantity": [5, 10, 2, 4, 6],
            "InvoiceDate": pd.to_datetime([
                "2011-01-01 10:00:00",
                "2011-01-05 12:00:00",
                "2011-02-01 14:00:00",
                "2011-03-01 16:00:00",
                "2011-03-15 11:00:00",
            ]),
            "UnitPrice": [2.0, 3.0, 5.0, 10.0, 2.0],
            "CustomerID": [101, 101, 102, 103, 103],
            "Country": ["United Kingdom"] * 5,
            "TotalPrice": [10.0, 30.0, 10.0, 40.0, 12.0],
        }
    )


def test_build_rfm_features(sample_cleaned_transactions: pd.DataFrame):
    """Test aggregation of RFM and behavioral metrics."""
    rfm_df = build_rfm_features(sample_cleaned_transactions, save_csv=False)

    assert len(rfm_df) == 3  # 3 distinct customers: 101, 102, 103
    assert "CustomerID" in rfm_df.columns

    required_features = get_model_features()
    for feat in required_features:
        assert feat in rfm_df.columns
        assert not rfm_df[feat].isna().any(), f"Feature {feat} contains NaN!"
        assert not np.isinf(rfm_df[feat]).any(), f"Feature {feat} contains Inf!"

    # Customer 102 is a single-order buyer: LifespanDays should be 0, AvgDaysBetweenPurchases 0
    c102 = rfm_df[rfm_df["CustomerID"] == 102].iloc[0]
    assert c102["Frequency"] == 1
    assert c102["LifespanDays"] == 0
    assert c102["AvgDaysBetweenPurchases"] == 0.0


def test_fit_transform_features(sample_cleaned_transactions: pd.DataFrame):
    """Test feature scaling pipeline transformation."""
    rfm_df = build_rfm_features(sample_cleaned_transactions, save_csv=False)
    X_scaled, pipeline = fit_transform_features(rfm_df, save_artifacts=False)

    assert isinstance(X_scaled, np.ndarray)
    assert X_scaled.shape == (3, 14)
    assert not np.isnan(X_scaled).any()

    # Transform new data
    X_new = transform_features(rfm_df, pipeline=pipeline)
    assert np.allclose(X_scaled, X_new)
