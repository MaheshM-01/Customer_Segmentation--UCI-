"""Unit tests for production Data Drift Monitoring (PSI, KS-Test, Cluster Shift)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift_detector import (
    calculate_psi,
    calculate_ks_test,
    DataDriftDetector,
)


def test_calculate_psi_identical_distributions():
    """Verify that identical distributions yield near-zero PSI (< 0.05)."""
    np.random.seed(42)
    expected = np.random.normal(loc=100.0, scale=15.0, size=2000)
    actual = np.random.normal(loc=100.0, scale=15.0, size=2000)

    psi = calculate_psi(expected, actual, num_bins=10)
    assert isinstance(psi, float)
    assert psi < 0.05, f"Expected near-zero PSI, got {psi}"


def test_calculate_psi_drifted_distributions():
    """Verify that significantly shifted distributions yield large PSI (> 0.25)."""
    np.random.seed(42)
    expected = np.random.normal(loc=50.0, scale=10.0, size=2000)
    actual = np.random.normal(loc=120.0, scale=30.0, size=2000)

    psi = calculate_psi(expected, actual, num_bins=10)
    assert psi > 0.25, f"Expected PSI > 0.25 indicating severe drift, got {psi}"


def test_calculate_psi_empty_or_nan():
    """Verify robust handling of empty or NaN values."""
    expected = np.array([])
    actual = np.array([1.0, 2.0, 3.0])
    assert calculate_psi(expected, actual) == 0.0

    exp_nan = np.array([np.nan, np.nan])
    act_nan = np.array([np.nan, np.nan])
    assert calculate_psi(exp_nan, act_nan) == 0.0


def test_calculate_ks_test():
    """Test two-sample KS test statistic and p-value behavior."""
    np.random.seed(42)
    dist1 = np.random.normal(0, 1, 1000)
    dist2 = np.random.normal(0, 1, 1000)
    stat_same, pval_same = calculate_ks_test(dist1, dist2)
    assert pval_same > 0.01  # Cannot reject null hypothesis that distributions are same

    dist_diff = np.random.exponential(scale=5, size=1000)
    stat_diff, pval_diff = calculate_ks_test(dist1, dist_diff)
    assert pval_diff < 0.01  # Rejects null hypothesis


def test_data_drift_detector_evaluate_drift():
    """Test DataDriftDetector end-to-end drift reporting with mock features."""
    from src.config import config

    np.random.seed(42)
    feature_cols = list(config.features.model_features)

    base_data = {col: np.random.uniform(10.0, 100.0, 500) for col in feature_cols}
    baseline_df = pd.DataFrame(base_data)

    detector = DataDriftDetector(baseline_df=baseline_df)

    # 1. Non-drifted current batch
    curr_data_nodrift = {col: np.random.uniform(10.0, 100.0, 200) for col in feature_cols}
    report_nodrift = detector.evaluate_drift(pd.DataFrame(curr_data_nodrift))
    assert not report_nodrift["overall_drift_detected"]
    assert report_nodrift["drifted_features_count"] <= 2

    # 2. Heavily drifted current batch
    curr_data_drifted = {col: np.random.uniform(500.0, 1000.0, 200) for col in feature_cols}
    report_drifted = detector.evaluate_drift(pd.DataFrame(curr_data_drifted))
    assert report_drifted["overall_drift_detected"]
    assert report_drifted["drifted_features_count"] > (len(feature_cols) // 3)
    assert report_drifted["recommended_action"] == "TRIGGER_RETRAINING"


def test_evaluate_cluster_shift():
    """Test cluster shift detection across baseline and current segment counts."""
    detector = DataDriftDetector(baseline_df=pd.DataFrame())

    # Equal proportions
    base_counts = {0: 1000, 1: 1000, 2: 1000, 3: 1000}
    curr_counts_stable = {0: 102, 1: 98, 2: 101, 3: 99}
    res_stable = detector.evaluate_cluster_shift(base_counts, curr_counts_stable)
    assert not res_stable["cluster_shift_detected"]

    # Massive shift (e.g. Cluster 0 dominates 80%)
    curr_counts_shifted = {0: 800, 1: 100, 2: 50, 3: 50}
    res_shifted = detector.evaluate_cluster_shift(base_counts, curr_counts_shifted)
    assert res_shifted["cluster_shift_detected"]
    assert res_shifted["segment_deltas"][0]["shift_flag"] is True
