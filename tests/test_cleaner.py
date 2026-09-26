"""Unit tests for data cleaner module."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.cleaner import (
    clean_data,
    filter_invalid_transactions,
    handle_outliers,
    remove_cancellations,
    remove_missing_customers,
)


@pytest.fixture
def sample_raw_data() -> pd.DataFrame:
    """Provide a mock raw transactions dataframe with cancellations, nulls, and negative values."""
    return pd.DataFrame(
        {
            "InvoiceNo": ["536365", "C536366", "536367", "536368", "536369"],
            "StockCode": ["85123A", "71053", "84406B", "84029G", "84029E"],
            "Description": ["Product A", "Product B", "Product C", "Product D", "Product E"],
            "Quantity": [6, -1, 10, -5, 100000],  # normal, cancelled, normal, invalid, extreme outlier
            "InvoiceDate": pd.to_datetime(["2010-12-01 08:26:00"] * 5),
            "UnitPrice": [2.55, 3.39, 2.75, -1.0, 500.0],
            "CustomerID": [17850.0, 17850.0, None, 13047.0, 12583.0],  # 1 null
            "Country": ["United Kingdom"] * 5,
        }
    )


def test_remove_missing_customers(sample_raw_data: pd.DataFrame):
    """Test dropping rows with null CustomerID."""
    cleaned = remove_missing_customers(sample_raw_data)
    assert len(cleaned) == 4
    assert cleaned["CustomerID"].isna().sum() == 0
    assert cleaned["CustomerID"].dtype == int


def test_remove_cancellations(sample_raw_data: pd.DataFrame):
    """Test filtering out invoices starting with 'C'."""
    cleaned = remove_cancellations(sample_raw_data)
    assert len(cleaned) == 4
    assert not cleaned["InvoiceNo"].str.startswith("C").any()


def test_filter_invalid_transactions(sample_raw_data: pd.DataFrame):
    """Test filtering negative/zero quantities and unit prices."""
    cleaned = filter_invalid_transactions(sample_raw_data)
    assert (cleaned["Quantity"] >= 1).all()
    assert (cleaned["UnitPrice"] >= 0.001).all()


def test_handle_outliers(sample_raw_data: pd.DataFrame):
    """Test capping extreme quantities and unit prices."""
    capped = handle_outliers(sample_raw_data)
    assert capped["Quantity"].max() < 100000


def test_clean_data_end_to_end(sample_raw_data: pd.DataFrame):
    """Test full cleaning pipeline."""
    cleaned = clean_data(sample_raw_data)

    # Valid remaining rows should be row 0 (536365) and row 4 (536369)
    assert len(cleaned) == 2
    assert "TotalPrice" in cleaned.columns
    assert (cleaned["TotalPrice"] == cleaned["Quantity"] * cleaned["UnitPrice"]).all()
    assert cleaned["CustomerID"].dtype == int
