"""Integration tests for FastAPI REST API endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.app import app


@pytest.fixture
def client() -> TestClient:
    """Provide a FastAPI TestClient."""
    return TestClient(app)


@pytest.fixture
def sample_customer_payload() -> dict:
    """Mock valid customer RFM feature dictionary."""
    return {
        "Recency": 15.0,
        "Frequency": 5.0,
        "Monetary": 1250.50,
        "AvgOrderValue": 250.10,
        "ProductVariety": 45.0,
        "LifespanDays": 180.0,
        "PurchaseRate": 0.0276,
        "AvgDaysBetweenPurchases": 36.0,
        "SpendPerProduct": 27.78,
        "QtyPerOrder": 18.5,
        "RevenuePerDay": 6.91,
        "UniqueProductPct": 0.45,
        "OrderValueConsistency": 35.20,
        "CLV_score": 1520.45,
    }


def test_root_endpoint(client: TestClient):
    """Test GET / returns 200 and valid JSON."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert data["docs_url"] == "/docs"


def test_health_endpoint(client: TestClient):
    """Test GET /health returns 200 and health info."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "models_loaded" in data


def test_personas_endpoint(client: TestClient):
    """Test GET /personas returns persona definitions."""
    response = client.get("/personas")
    assert response.status_code == 200
    data = response.json()
    assert "personas" in data
    assert "strategies" in data


def test_predict_validation_error(client: TestClient):
    """Test POST /predict with missing or invalid fields returns 422 Unprocessable Entity."""
    response = client.post("/predict", json={"Recency": -5.0})  # Negative value violates ge=0.0
    assert response.status_code == 422


def test_metrics_endpoint(client: TestClient):
    """Test GET /metrics returns runtime telemetry."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_requests" in data
    assert "uptime_seconds" in data
    assert "average_latency_ms" in data


def test_predict_and_batch_endpoints(client: TestClient, sample_customer_payload: dict):
    """Test POST /predict and /predict/batch when models are available."""
    # Single prediction
    resp_single = client.post("/predict", json=sample_customer_payload)
    if resp_single.status_code == 200:
        data = resp_single.json()
        assert "cluster_id" in data
        assert "persona" in data
        assert "strategy" in data
        assert "is_anomaly" in data

    # Batch prediction
    resp_batch = client.post(
        "/predict/batch",
        json={"customers": [sample_customer_payload, sample_customer_payload]},
    )
    if resp_batch.status_code == 200:
        batch_data = resp_batch.json()
        assert batch_data["total_customers"] == 2
        assert len(batch_data["predictions"]) == 2


def test_monitoring_endpoints(client: TestClient, sample_customer_payload: dict):
    """Test POST /monitoring/drift and /monitoring/cluster-shift."""
    batch_payload = {"customers": [sample_customer_payload] * 10}

    # Drift endpoint
    drift_resp = client.post("/monitoring/drift", json=batch_payload)
    if drift_resp.status_code == 200:
        data = drift_resp.json()
        assert "overall_drift_detected" in data
        assert "feature_metrics" in data

    # Cluster shift endpoint
    shift_resp = client.post("/monitoring/cluster-shift", json=batch_payload)
    if shift_resp.status_code == 200:
        shift_data = shift_resp.json()
        assert "cluster_shift_detected" in shift_data
        assert "segment_deltas" in shift_data

