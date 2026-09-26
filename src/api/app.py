"""FastAPI REST API application for Customer Segmentation, Anomaly Detection & Market Basket."""

from __future__ import annotations

from contextlib import asynccontextmanager
import time
from typing import Any, Dict, List
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware

from src.config import config
from src.utils.logger import logger
from src.pipeline.inference_pipeline import (
    CustomerSegmentationService,
    get_inference_service,
)
from src.monitoring.drift_detector import DataDriftDetector
from src.api.schemas import (
    BatchCustomerInput,
    BatchCustomerResponse,
    ClusterShiftResponse,
    CustomerFeaturesInput,
    CustomerPredictionResponse,
    DriftMonitoringResponse,
    HealthResponse,
    RecommendationRequest,
    RecommendationResponse,
    RecommendedProduct,
)

# API metrics tracking accumulator
API_METRICS: Dict[str, Any] = {
    "total_requests": 0,
    "total_predictions": 0,
    "total_errors": 0,
    "total_latency_seconds": 0.0,
    "start_time": time.time(),
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler to pre-load ML models on startup."""
    logger.info("Initializing Customer Segmentation API...")
    try:
        service = get_inference_service()
        logger.info(
            f"Successfully warmed up ML models: KMeans (K={service.kmeans.n_clusters if service.kmeans else 'N/A'})"
        )
    except Exception as exc:
        logger.warning(
            f"Models not loaded during startup (run train_pipeline first if not trained): {exc}"
        )
    yield
    logger.info("Shutting down Customer Segmentation API.")


app = FastAPI(
    title=config.api.title,
    version=config.project.version,
    description=(
        "Production Machine Learning API for UCI Customer Segmentation, "
        "Behavioral RFM Persona Profiling, Whale Anomaly Detection, "
        "and Market Basket Cross-Sell Recommendations."
    ),
    lifespan=lifespan,
)

# Enable CORS for external dashboard frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def track_metrics_middleware(request: Request, call_next):
    """Middleware to track API latency, request counts, and error rates."""
    start_time = time.time()
    API_METRICS["total_requests"] += 1
    try:
        response: Response = await call_next(request)
        latency = time.time() - start_time
        API_METRICS["total_latency_seconds"] += latency
        response.headers["X-Process-Time-Ms"] = f"{latency * 1000:.2f}"
        if response.status_code >= 400:
            API_METRICS["total_errors"] += 1
        return response
    except Exception:
        API_METRICS["total_errors"] += 1
        raise


@app.get(
    "/",
    tags=["Root"],
    summary="Root welcome message",
)
def root() -> Dict[str, str]:
    """Root endpoint providing service information and documentation link."""
    return {
        "service": config.api.title,
        "version": config.project.version,
        "docs_url": "/docs",
        "health_url": "/health",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check and model status",
)
def health_check() -> HealthResponse:
    """Check API operational health and model readiness."""
    try:
        service = get_inference_service()
        models_ready = bool(service.is_loaded and service.kmeans is not None)
    except Exception:
        models_ready = False

    return HealthResponse(
        status="healthy" if models_ready else "degraded (models not trained)",
        project=config.project.name,
        version=config.project.version,
        models_loaded=models_ready,
    )


@app.get(
    "/metrics",
    tags=["Monitoring"],
    summary="Get API operational performance metrics",
)
def get_metrics() -> Dict[str, Any]:
    """Retrieve runtime operational metrics: request counts, latencies, errors, and uptime."""
    reqs = API_METRICS["total_requests"]
    avg_latency_ms = (
        (API_METRICS["total_latency_seconds"] / reqs * 1000) if reqs > 0 else 0.0
    )
    uptime_sec = time.time() - API_METRICS["start_time"]
    return {
        "service": config.api.title,
        "version": config.project.version,
        "uptime_seconds": round(uptime_sec, 2),
        "total_requests": reqs,
        "total_predictions": API_METRICS["total_predictions"],
        "total_errors": API_METRICS["total_errors"],
        "average_latency_ms": round(avg_latency_ms, 2),
    }


@app.post(
    "/predict",
    response_model=CustomerPredictionResponse,
    tags=["Segmentation"],
    summary="Score a single customer",
)
def predict_single(customer: CustomerFeaturesInput) -> CustomerPredictionResponse:
    """Predict customer cluster, persona, retention strategy, PCA coordinates, and anomaly status."""
    try:
        service = get_inference_service()
        payload = customer.model_dump()
        result = service.predict_customer(payload)
        API_METRICS["total_predictions"] += 1
        return CustomerPredictionResponse(**result)
    except FileNotFoundError as fnf:
        logger.error(f"Inference error - missing model: {fnf}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifacts not found. Please train models first using: python -m src.pipeline.train_pipeline",
        )
    except Exception as exc:
        logger.error(f"Inference error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Customer scoring failed: {str(exc)}",
        )


@app.post(
    "/predict/batch",
    response_model=BatchCustomerResponse,
    tags=["Segmentation"],
    summary="Score multiple customers in batch",
)
def predict_batch(batch_input: BatchCustomerInput) -> BatchCustomerResponse:
    """Score a batch list of customer feature vectors."""
    try:
        service = get_inference_service()
        customer_records = [c.model_dump() for c in batch_input.customers]
        df_batch = pd.DataFrame(customer_records)

        predictions_df = service.predict_batch(df_batch)

        preds = []
        for _, row in predictions_df.iterrows():
            cid = int(row["Cluster"])
            preds.append(
                CustomerPredictionResponse(
                    cluster_id=cid,
                    persona=str(row["Persona"]),
                    strategy=str(row["Strategy"]),
                    pca_x=float(row["PCA_1"]),
                    pca_y=float(row["PCA_2"]),
                    is_anomaly=bool(row["Is_Anomaly"]),
                    anomaly_score=float(row["Anomaly_Score"]),
                    profile_summary=service.cluster_profiles.get(str(cid), {}),
                )
            )

        API_METRICS["total_predictions"] += len(batch_input.customers)
        return BatchCustomerResponse(
            total_customers=len(preds),
            predictions=preds,
        )
    except Exception as exc:
        logger.error(f"Batch prediction error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch customer scoring failed: {str(exc)}",
        )


@app.post(
    "/recommend",
    response_model=RecommendationResponse,
    tags=["Market Basket"],
    summary="Get cross-sell product recommendations",
)
def recommend(request: RecommendationRequest) -> RecommendationResponse:
    """Recommend complementary cross-sell products for items currently in cart."""
    try:
        service = get_inference_service()
        recs_data = service.get_recommendations(
            purchased_items=request.purchased_items,
            top_k=request.top_k,
        )
        recommendations = [RecommendedProduct(**r) for r in recs_data]

        return RecommendationResponse(
            query_items=request.purchased_items,
            recommendations=recommendations,
        )
    except Exception as exc:
        logger.error(f"Recommendation error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Product recommendation failed: {str(exc)}",
        )


@app.get(
    "/personas",
    tags=["Segmentation"],
    summary="Get all cluster persona definitions and profiles",
)
def get_personas() -> Dict[str, Any]:
    """Retrieve metadata, profiles, and marketing strategies for all customer clusters."""
    try:
        service = get_inference_service()
        return {
            "n_clusters": service.kmeans.n_clusters if service.kmeans else config.clustering.n_clusters,
            "personas": service.personas,
            "strategies": service.strategies,
            "profiles": service.cluster_profiles,
        }
    except Exception as exc:
        return {
            "personas": dict(config.clustering.personas),
            "strategies": dict(config.clustering.strategies),
            "profiles": {},
            "notice": f"Fitted profiles not yet loaded: {str(exc)}",
        }


@app.post(
    "/monitoring/drift",
    response_model=DriftMonitoringResponse,
    tags=["Monitoring"],
    summary="Evaluate feature data drift against baseline dataset",
)
def evaluate_drift_endpoint(batch_input: BatchCustomerInput) -> DriftMonitoringResponse:
    """Evaluate Population Stability Index (PSI) and Kolmogorov-Smirnov test drift on incoming batch."""
    try:
        if not batch_input.customers:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Batch must contain at least 1 customer record for drift analysis.",
            )
        records = [c.model_dump() for c in batch_input.customers]
        df_batch = pd.DataFrame(records)
        detector = DataDriftDetector()
        report = detector.evaluate_drift(df_batch)
        return DriftMonitoringResponse(**report)
    except RuntimeError as r_err:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(r_err),
        )
    except Exception as exc:
        logger.error(f"Drift monitoring error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Drift evaluation failed: {str(exc)}",
        )


@app.post(
    "/monitoring/cluster-shift",
    response_model=ClusterShiftResponse,
    tags=["Monitoring"],
    summary="Evaluate macro cluster distribution shift",
)
def evaluate_cluster_shift_endpoint(batch_input: BatchCustomerInput) -> ClusterShiftResponse:
    """Evaluate macro behavioral shift in customer cluster proportions."""
    try:
        if not batch_input.customers:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Batch must contain at least 1 customer record for cluster shift analysis.",
            )
        service = get_inference_service()
        records = [c.model_dump() for c in batch_input.customers]
        df_batch = pd.DataFrame(records)
        pred_df = service.predict_batch(df_batch)

        current_counts = {int(k): int(v) for k, v in pred_df["Cluster"].value_counts().items()}
        baseline_counts = {
            int(k): int(v.get("customer_count", 0))
            for k, v in service.cluster_profiles.items()
        }

        detector = DataDriftDetector()
        shift_result = detector.evaluate_cluster_shift(baseline_counts, current_counts)

        # Ensure keys in segment_deltas are strings for JSON schema
        formatted_deltas = {str(k): v for k, v in shift_result["segment_deltas"].items()}
        shift_result["segment_deltas"] = formatted_deltas

        return ClusterShiftResponse(**shift_result)
    except Exception as exc:
        logger.error(f"Cluster shift evaluation error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cluster shift evaluation failed: {str(exc)}",
        )


if __name__ == "__main__":
    uvicorn.run(
        "src.api.app:app",
        host=config.api.host,
        port=config.api.port,
        reload=True,
    )
