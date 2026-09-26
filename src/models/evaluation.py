"""Clustering evaluation metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz) and explainability."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.tree import DecisionTreeClassifier, export_text

from src.config import config
from src.utils.logger import logger
from src.utils.helpers import timeit


@timeit
def evaluate_clustering(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    sample_size: Optional[int] = 10000,
) -> Dict[str, float]:
    """Calculate standard unsupervised clustering quality metrics."""
    n_labels = len(np.unique(labels))
    if n_labels < 2:
        logger.warning("Clustering produced < 2 clusters. Cannot compute evaluation metric.")
        return {"silhouette_score": -1.0, "davies_bouldin_score": 999.0, "calinski_harabasz_score": 0.0}

    logger.info(f"Evaluating clustering performance for {n_labels} clusters across {len(X_scaled):,} samples...")

    sil_sample = sample_size if len(X_scaled) > (sample_size or 0) else None
    sil = float(silhouette_score(X_scaled, labels, sample_size=sil_sample, random_state=42))
    db = float(davies_bouldin_score(X_scaled, labels))
    ch = float(calinski_harabasz_score(X_scaled, labels))

    metrics = {
        "silhouette_score": round(sil, 4),
        "davies_bouldin_score": round(db, 4),
        "davies_bouldin_index": round(db, 4),
        "calinski_harabasz_score": round(ch, 2),
        "calinski_harabasz_index": round(ch, 2),
    }

    logger.info(
        f"Evaluation Metrics: Silhouette = {metrics['silhouette_score']} | "
        f"Davies-Bouldin = {metrics['davies_bouldin_score']} | "
        f"Calinski-Harabasz = {metrics['calinski_harabasz_score']}"
    )
    return metrics


def evaluate_k_range(
    X_scaled: np.ndarray,
    k_min: int = 2,
    k_max: int = 8,
    random_state: int = 42,
) -> pd.DataFrame:
    """Evaluate K-Means across a range of K to compute Elbow curve and Silhouette scores."""
    logger.info(f"Scanning optimal K from {k_min} to {k_max}...")
    records = []

    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=random_state)
        km.fit(X_scaled)
        labels = km.labels_

        sil = float(silhouette_score(X_scaled, labels, sample_size=5000, random_state=random_state))
        db = float(davies_bouldin_score(X_scaled, labels))
        ch = float(calinski_harabasz_score(X_scaled, labels))

        records.append(
            {
                "k": k,
                "inertia": round(float(km.inertia_), 2),
                "silhouette": round(sil, 4),
                "davies_bouldin": round(db, 4),
                "calinski_harabasz": round(ch, 2),
            }
        )
        logger.info(f"  K={k} -> Silhouette={sil:.4f}, Inertia={km.inertia_:.1f}, DB={db:.3f}")

    return pd.DataFrame(records)


def explain_clusters_surrogate(
    X_df: pd.DataFrame,
    cluster_labels: np.ndarray,
    max_depth: int = 3,
) -> Dict[str, Any]:
    """Train a shallow surrogate Decision Tree to extract human-interpretable segmentation rules."""
    logger.info(f"Fitting surrogate Decision Tree (depth={max_depth}) for clustering explainability...")
    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    tree.fit(X_df, cluster_labels)

    fidelity_acc = float(tree.score(X_df, cluster_labels))
    logger.info(f"Surrogate Decision Tree fidelity accuracy: {fidelity_acc * 100:.2f}%")

    feature_names = list(X_df.columns)
    importances = dict(
        sorted(
            zip(feature_names, [round(float(v), 4) for v in tree.feature_importances_]),
            key=lambda item: item[1],
            reverse=True,
        )
    )

    tree_rules = export_text(tree, feature_names=feature_names)

    return {
        "fidelity_accuracy": round(fidelity_acc, 4),
        "tree_accuracy": round(fidelity_acc, 4),
        "top_features": importances,
        "feature_importances": importances,
        "decision_rules": tree_rules,
    }


# Backwards compatibility alias
explain_cluster_surrogate = explain_clusters_surrogate