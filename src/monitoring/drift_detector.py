"""Production Data and Concept Drift Monitoring Engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats

from src.config import config, PROJECT_ROOT
from src.utils.logger import logger


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    num_bins: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """Calculate the Population Stability Index (PSI) between baseline and current data.

    Interpretation:
    - PSI < 0.10: No significant population change (stable).
    - 0.10 <= PSI < 0.25: Moderate shift; warrants monitoring.
    - PSI >= 0.25: Significant population drift; model re-training recommended.

    Parameters
    ----------
    expected : np.ndarray
        Baseline / training feature values.
    actual : np.ndarray
        Production / inference feature values.
    num_bins : int, default=10
        Number of quantile bins.
    epsilon : float, default=1e-4
        Small offset to prevent zero-division in log calculation.

    Returns
    -------
    float
        Calculated PSI value.
    """
    clean_expected = expected[~np.isnan(expected)]
    clean_actual = actual[~np.isnan(actual)]

    if len(clean_expected) == 0 or len(clean_actual) == 0:
        return 0.0

    # Determine quantile bin thresholds using expected data
    quantiles = np.linspace(0, 100, num_bins + 1)
    bins = np.percentile(clean_expected, quantiles)
    bins = np.unique(bins)  # Drop duplicate bin edges

    if len(bins) < 2:
        return 0.0

    # Ensure infinite outer edges
    bins[0] = -np.inf
    bins[-1] = np.inf

    # Calculate frequencies per bin
    expected_counts, _ = np.histogram(clean_expected, bins=bins)
    actual_counts, _ = np.histogram(clean_actual, bins=bins)

    expected_pct = expected_counts / len(clean_expected)
    actual_pct = actual_counts / len(clean_actual)

    # Apply epsilon smoothing
    expected_pct = np.where(expected_pct == 0, epsilon, expected_pct)
    actual_pct = np.where(actual_pct == 0, epsilon, actual_pct)

    # Compute PSI
    psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(np.round(psi_value, 4))


def calculate_ks_test(
    expected: np.ndarray,
    actual: np.ndarray,
) -> Tuple[float, float]:
    """Perform two-sample Kolmogorov-Smirnov test to detect distribution differences."""
    clean_exp = expected[~np.isnan(expected)]
    clean_act = actual[~np.isnan(actual)]

    if len(clean_exp) < 2 or len(clean_act) < 2:
        return 0.0, 1.0

    res = stats.ks_2samp(clean_exp, clean_act)
    return float(round(res.statistic, 4)), float(round(res.pvalue, 4))


class DataDriftDetector:
    """Enterprise Data Drift and Concept Shift monitoring service."""

    def __init__(
        self,
        baseline_df: Optional[pd.DataFrame] = None,
        psi_threshold: Optional[float] = None,
        ks_pvalue_threshold: Optional[float] = None,
    ) -> None:
        self.psi_threshold = (
            psi_threshold
            if psi_threshold is not None
            else config.monitoring.drift_threshold_psi
        )
        self.ks_pvalue_threshold = (
            ks_pvalue_threshold
            if ks_pvalue_threshold is not None
            else config.monitoring.drift_threshold_ks_pvalue
        )
        self.feature_names: List[str] = list(config.features.model_features)

        # Load or set baseline dataframe
        if baseline_df is not None:
            self.baseline_df = baseline_df.copy()
        else:
            baseline_path = PROJECT_ROOT / config.monitoring.baseline_features_csv
            if baseline_path.is_file():
                self.baseline_df = pd.read_csv(baseline_path)
                logger.info(f"Loaded {len(self.baseline_df):,} baseline customer records for drift monitoring.")
            else:
                self.baseline_df = pd.DataFrame()
                logger.warning(f"Baseline features CSV not found at: {baseline_path}")

    def evaluate_drift(
        self,
        current_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Assess population drift across all 14 RFM and behavioral metrics."""
        if self.baseline_df.empty:
            raise RuntimeError("Baseline dataset is not available for drift evaluation.")

        feature_reports = {}
        drifted_features = []
        max_psi = 0.0

        for feat in self.feature_names:
            if feat in self.baseline_df.columns and feat in current_df.columns:
                exp_vals = self.baseline_df[feat].to_numpy(dtype=float)
                act_vals = current_df[feat].to_numpy(dtype=float)

                psi = calculate_psi(exp_vals, act_vals)
                ks_stat, ks_pval = calculate_ks_test(exp_vals, act_vals)

                # Determine feature-level drift condition
                has_drift = bool(psi >= self.psi_threshold or ks_pval < self.ks_pvalue_threshold)

                if has_drift:
                    drifted_features.append(feat)

                if psi > max_psi:
                    max_psi = psi

                feature_reports[feat] = {
                    "psi": psi,
                    "ks_statistic": ks_stat,
                    "ks_pvalue": ks_pval,
                    "drift_status": "SIGNIFICANT" if psi >= 0.25 else ("MODERATE" if psi >= 0.10 else "STABLE"),
                    "drift_detected": has_drift,
                }

        overall_drift = len(drifted_features) > (len(self.feature_names) // 3)

        report = {
            "overall_drift_detected": overall_drift,
            "max_psi": max_psi,
            "psi_threshold": self.psi_threshold,
            "drifted_features_count": len(drifted_features),
            "total_features_evaluated": len(feature_reports),
            "drifted_features": drifted_features,
            "feature_metrics": feature_reports,
            "recommended_action": (
                "TRIGGER_RETRAINING" if overall_drift
                else ("MONITOR_CLOSELY" if drifted_features else "NORMAL_OPERATION")
            ),
        }

        logger.info(
            f"Drift analysis complete: overall_drift={overall_drift} "
            f"({len(drifted_features)}/{len(self.feature_names)} features drifted, max PSI={max_psi:.4f})"
        )
        return report

    def evaluate_cluster_shift(
        self,
        baseline_cluster_counts: Dict[int, int],
        current_cluster_counts: Dict[int, int],
    ) -> Dict[str, Any]:
        """Detect macro behavioral shifts in customer segment proportions."""
        base_total = sum(baseline_cluster_counts.values()) or 1
        curr_total = sum(current_cluster_counts.values()) or 1

        cluster_shifts = {}
        shift_detected = False

        for cid in sorted(set(baseline_cluster_counts.keys()).union(current_cluster_counts.keys())):
            base_share = baseline_cluster_counts.get(cid, 0) / base_total
            curr_share = current_cluster_counts.get(cid, 0) / curr_total
            diff = curr_share - base_share

            if abs(diff) >= config.monitoring.cluster_shift_threshold:
                shift_detected = True

            cluster_shifts[cid] = {
                "baseline_share_pct": round(base_share * 100, 2),
                "current_share_pct": round(curr_share * 100, 2),
                "share_delta_pct": round(diff * 100, 2),
                "shift_flag": abs(diff) >= config.monitoring.cluster_shift_threshold,
            }

        return {
            "cluster_shift_detected": shift_detected,
            "threshold_pct": config.monitoring.cluster_shift_threshold * 100,
            "segment_deltas": cluster_shifts,
        }
