"""Custom scikit-learn transformers (PercentileClipper)."""

from __future__ import annotations

from typing import Any, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class PercentileClipper(BaseEstimator, TransformerMixin):
    """Clips continuous feature columns to lower and upper percentile thresholds."""

    def __init__(
        self,
        lower_quantile: float = 0.01,
        upper_quantile: float = 0.99,
    ) -> None:
        if not (0.0 <= lower_quantile < upper_quantile <= 1.0):
            raise ValueError(
                f"Quantiles must satisfy 0 <= lower < upper <= 1. "
                f"Got lower={lower_quantile}, upper={upper_quantile}"
            )
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Any] = None,
    ) -> "PercentileClipper":
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.array(X.columns, dtype=str)
            X_arr = X.to_numpy(dtype=np.float64)
        else:
            X_arr = np.asarray(X, dtype=np.float64)
            self.feature_names_in_ = None

        if X_arr.ndim != 2:
            raise ValueError(f"Expected 2d array, got shape: {X_arr.shape}")

        self.n_features_in_ = X_arr.shape[1]
        self.lower_bounds_ = np.nanpercentile(
            X_arr, self.lower_quantile * 100.0, axis=0
        )
        self.upper_bounds_ = np.nanpercentile(
            X_arr, self.upper_quantile * 100.0, axis=0
        )
        return self

    def transform(
        self,
        X: Union[np.ndarray, pd.DataFrame],
    ) -> np.ndarray:
        check_is_fitted(self, attributes=["lower_bounds_", "upper_bounds_"])

        if isinstance(X, pd.DataFrame):
            X_arr = X.to_numpy(dtype=np.float64)
        else:
            X_arr = np.asarray(X, dtype=np.float64)

        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Feature count mismatch: fitted on {self.n_features_in_} features "
                f"but received {X_arr.shape[1]} features"
            )

        X_clipped = np.clip(X_arr, self.lower_bounds_, self.upper_bounds_)
        return X_clipped

    def get_feature_names_out(
        self,
        input_features: Optional[List[str]] = None,
    ) -> np.ndarray:
        check_is_fitted(self, attributes=["n_features_in_"])
        if input_features is not None:
            if len(input_features) != self.n_features_in_:
                raise ValueError("input_features length does not match n_features_in_")
            return np.asarray(input_features, dtype=str)
        if self.feature_names_in_ is not None:
            return self.feature_names_in_
        return np.array([f"feature_{i}" for i in range(self.n_features_in_)])
