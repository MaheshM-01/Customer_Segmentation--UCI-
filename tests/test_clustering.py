"""Unit tests for clustering and anomaly detection models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.clustering import CustomerClusterer
from src.models.anomaly import CustomerAnomalyDetector
from src.models.evaluation import evaluate_clustering, explain_clusters_surrogate


@pytest.fixture
def mock_feature_matrix() -> np.ndarray:
    """Mock 100 customer records in 14-dimensional scaled space with 2 natural clusters."""
    np.random.seed(42)
    cluster_1 = np.random.normal(loc=-2.0, scale=0.5, size=(50, 14))
    cluster_2 = np.random.normal(loc=2.0, scale=0.5, size=(50, 14))
    return np.vstack([cluster_1, cluster_2])


def test_customer_clusterer_fit_and_predict(mock_feature_matrix: np.ndarray):
    """Test CustomerClusterer model training and persona prediction."""
    clusterer = CustomerClusterer(n_clusters=2, random_state=42)
    clusterer.fit(mock_feature_matrix, save_artifacts=False)

    assert clusterer.is_fitted
    labels = clusterer.predict(mock_feature_matrix)
    assert len(labels) == 100
    assert set(labels) == {0, 1}

    # PCA transformation
    pca_2d = clusterer.transform_pca(mock_feature_matrix)
    assert pca_2d.shape == (100, 2)

    # Persona prediction
    personas = clusterer.predict_persona(mock_feature_matrix[:5])
    assert len(personas) == 5
    for p in personas:
        assert "cluster_id" in p
        assert "persona" in p
        assert "strategy" in p


def test_evaluate_clustering(mock_feature_matrix: np.ndarray):
    """Test clustering evaluation metrics computation."""
    clusterer = CustomerClusterer(n_clusters=2, random_state=42)
    clusterer.fit(mock_feature_matrix, save_artifacts=False)
    labels = clusterer.predict(mock_feature_matrix)

    metrics = evaluate_clustering(mock_feature_matrix, labels)
    assert "silhouette_score" in metrics
    assert "davies_bouldin_score" in metrics
    assert "calinski_harabasz_score" in metrics
    # Well-separated synthetic clusters should have high silhouette (> 0.5)
    assert metrics["silhouette_score"] > 0.5


def test_customer_anomaly_detector(mock_feature_matrix: np.ndarray):
    """Test Isolation Forest outlier detector."""
    detector = CustomerAnomalyDetector(contamination=0.05, random_state=42)
    detector.fit(mock_feature_matrix, save_artifact=False)

    assert detector.is_fitted
    preds = detector.predict(mock_feature_matrix)
    assert len(preds) == 100
    assert set(preds).issubset({1, -1})

    df_detect = detector.detect(mock_feature_matrix)
    assert "is_anomaly" in df_detect.columns
    assert "anomaly_score" in df_detect.columns
    assert df_detect["is_anomaly"].sum() > 0  # Should detect ~5 anomalies
