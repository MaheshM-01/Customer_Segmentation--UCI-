"""Clustering, anomaly detection, market basket analysis, and evaluation modules."""

from src.models.clustering import CustomerClusterer
from src.models.evaluation import evaluate_clustering, evaluate_k_range, explain_clusters_surrogate
from src.models.anomaly import CustomerAnomalyDetector
from src.models.market_basket import (
    run_market_basket_analysis,
    recommend_products,
    build_basket_matrix,
    build_product_lookup,
)

__all__ = [
    "CustomerClusterer",
    "evaluate_clustering",
    "evaluate_k_range",
    "explain_clusters_surrogate",
    "CustomerAnomalyDetector",
    "run_market_basket_analysis",
    "recommend_products",
    "build_basket_matrix",
    "build_product_lookup",
]
