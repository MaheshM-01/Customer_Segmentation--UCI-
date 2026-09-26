"""Pydantic request and response schemas for FastAPI customer segmentation endpoints."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class CustomerFeaturesInput(BaseModel):
    """Input payload representing customer-level RFM and behavioral metrics."""

    Recency: float = Field(
        ...,
        ge=0.0,
        description="Days since customer's last purchase",
        examples=[15.0],
    )
    Frequency: float = Field(
        ...,
        ge=1.0,
        description="Total distinct orders/invoices placed",
        examples=[5.0],
    )
    Monetary: float = Field(
        ...,
        ge=0.0,
        description="Total expenditure (spend in GBP)",
        examples=[1250.50],
    )
    AvgOrderValue: float = Field(
        ...,
        ge=0.0,
        description="Average spend per invoice (Monetary / Frequency)",
        examples=[250.10],
    )
    ProductVariety: float = Field(
        ...,
        ge=1.0,
        description="Count of unique StockCodes purchased",
        examples=[45.0],
    )
    LifespanDays: float = Field(
        ...,
        ge=0.0,
        description="Days between first and last purchase (tenure)",
        examples=[180.0],
    )
    PurchaseRate: float = Field(
        ...,
        ge=0.0,
        description="Frequency / (LifespanDays + 1)",
        examples=[0.0276],
    )
    AvgDaysBetweenPurchases: float = Field(
        ...,
        ge=0.0,
        description="Average gap in days between successive orders",
        examples=[36.0],
    )
    SpendPerProduct: float = Field(
        ...,
        ge=0.0,
        description="Monetary / ProductVariety",
        examples=[27.78],
    )
    QtyPerOrder: float = Field(
        ...,
        ge=0.0,
        description="Average quantity of items per order",
        examples=[18.5],
    )
    RevenuePerDay: float = Field(
        ...,
        ge=0.0,
        description="Monetary / (LifespanDays + 1)",
        examples=[6.91],
    )
    UniqueProductPct: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Ratio of unique products to total transaction lines",
        examples=[0.45],
    )
    OrderValueConsistency: float = Field(
        ...,
        ge=0.0,
        description="Standard deviation of order spending amounts",
        examples=[35.20],
    )
    CLV_score: float = Field(
        ...,
        ge=0.0,
        description="Customer Lifetime Value proxy score",
        examples=[1520.45],
    )

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class CustomerPredictionResponse(BaseModel):
    """Output prediction payload for a single customer."""

    cluster_id: int = Field(..., description="Assigned K-Means cluster index (0, 1, 2, 3)")
    persona: str = Field(..., description="Human-readable customer segment persona")
    strategy: str = Field(..., description="Actionable marketing and CRM retention strategy")
    pca_x: float = Field(..., description="PCA Dimension 1 coordinate for dashboard visualization")
    pca_y: float = Field(..., description="PCA Dimension 2 coordinate for dashboard visualization")
    is_anomaly: bool = Field(..., description="True if customer is flagged as an outlier/whale")
    anomaly_score: float = Field(..., description="Continuous Isolation Forest abnormality score")
    profile_summary: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Statistical summary of the assigned cluster persona",
    )


class BatchCustomerInput(BaseModel):
    """Batch input payload containing multiple customers."""

    customers: List[CustomerFeaturesInput]


class BatchCustomerResponse(BaseModel):
    """Batch prediction output payload."""

    total_customers: int
    predictions: List[CustomerPredictionResponse]


class RecommendationRequest(BaseModel):
    """Request payload for Market Basket product recommendations."""

    purchased_items: List[str] = Field(
        ...,
        description="List of StockCodes or product names currently in cart",
        examples=[["85123A", "WHITE HANGING HEART T-LIGHT HOLDER"]],
    )
    top_k: int = Field(default=3, ge=1, le=10, description="Max number of recommendations to return")


class RecommendedProduct(BaseModel):
    """Individual product recommendation with lift and confidence."""

    recommended_product: str
    product_code: str
    confidence: float
    lift: float


class RecommendationResponse(BaseModel):
    """Response payload containing suggested cross-sell products."""

    query_items: List[str]
    recommendations: List[RecommendedProduct]


class HealthResponse(BaseModel):
    """Service health and model status response."""

    status: str
    project: str
    version: str
    models_loaded: bool


class FeatureDriftMetric(BaseModel):
    """Statistical drift metrics for a single feature."""

    psi: float
    ks_statistic: float
    ks_pvalue: float
    drift_status: str
    drift_detected: bool


class DriftMonitoringResponse(BaseModel):
    """Comprehensive data drift assessment response across all RFM features."""

    overall_drift_detected: bool
    max_psi: float
    psi_threshold: float
    drifted_features_count: int
    total_features_evaluated: int
    drifted_features: List[str]
    feature_metrics: Dict[str, FeatureDriftMetric]
    recommended_action: str


class ClusterShiftResponse(BaseModel):
    """Macro cluster shift assessment response."""

    cluster_shift_detected: bool
    threshold_pct: float
    segment_deltas: Dict[str, Any]

