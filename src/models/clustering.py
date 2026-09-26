"""Clustering models (K-Means, PCA) and business persona mapping."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from src.config import config
from src.utils.logger import logger
from src.utils.helpers import save_joblib, load_joblib, save_json, timeit


class CustomerClusterer:
    """Production K-Means customer segmentation engine."""

    def __init__(
        self,
        n_clusters: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_clusters = (
            n_clusters if n_clusters is not None else config.clustering.n_clusters
        )
        self.random_state = (
            random_state if random_state is not None else config.clustering.random_state
        )

        self.model = KMeans(
            n_clusters=self.n_clusters,
            init=config.clustering.init_method,
            n_init=config.clustering.n_init,
            max_iter=config.clustering.max_iter,
            random_state=self.random_state,
        )
        self.pca = PCA(n_components=2, random_state=self.random_state)
        self.personas: Dict[int, str] = dict(config.clustering.personas)
        self.strategies: Dict[int, str] = dict(config.clustering.strategies)
        self.cluster_profiles: Dict[int, Dict[str, Any]] = {}
        self.is_fitted = False

    @timeit
    def fit(
        self,
        X_scaled: np.ndarray,
        customer_df: Optional[pd.DataFrame] = None,
        save_artifacts: bool = True,
    ) -> "CustomerClusterer":
        """Fit K-Means clustering and PCA on scaled customer features."""
        logger.info(
            f"Fitting K-Means (K={self.n_clusters}, init='k-means++', n_init={self.model.n_init}) "
            f"on {len(X_scaled):,} customer records..."
        )
        self.model.fit(X_scaled)
        self.pca.fit(X_scaled)
        self.is_fitted = True

        labels = self.model.labels_
        logger.info(f"K-Means fitting complete. Inertia: {self.model.inertia_:.2f}")

        unique, counts = np.unique(labels, return_counts=True)
        for cluster_id, count in zip(unique, counts):
            pct = (count / len(X_scaled)) * 100
            persona_name = self.personas.get(cluster_id, f"Cluster {cluster_id}")
            logger.info(f"  - Cluster {cluster_id} ({persona_name}): {count:,} customers ({pct:.1f}%)")

        if customer_df is not None:
            self.generate_profiles(customer_df, labels, save_json_file=save_artifacts)

        if save_artifacts:
            self.save()

        return self

    def predict(self, X_scaled: np.ndarray) -> np.ndarray:
        """Predict cluster indices for given feature matrix."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() or load() first.")
        return self.model.predict(X_scaled)

    def transform_pca(self, X_scaled: np.ndarray) -> np.ndarray:
        """Project scaled features into 2D PCA space for visualization."""
        if not self.is_fitted:
            raise RuntimeError("PCA is not fitted yet.")
        return self.pca.transform(X_scaled)

    def predict_persona(self, X_scaled: np.ndarray) -> List[Dict[str, Any]]:
        """Predict cluster, persona label, and recommended marketing strategy."""
        cluster_ids = self.predict(X_scaled)
        results = []
        for cid in cluster_ids:
            cid_int = int(cid)
            results.append(
                {
                    "cluster_id": cid_int,
                    "persona": self.personas.get(cid_int, f"Segment {cid_int}"),
                    "strategy": self.strategies.get(cid_int, "Standard customer retention"),
                }
            )
        return results

    def generate_profiles(
        self,
        customer_df: pd.DataFrame,
        labels: np.ndarray,
        save_json_file: bool = True,
    ) -> Dict[int, Dict[str, Any]]:
        """Calculate summary statistics per cluster and export to JSON."""
        logger.info("Generating business persona profiles per cluster...")
        df_copy = customer_df.copy()
        df_copy["Cluster"] = labels

        total_customers = len(df_copy)
        profiles = {}

        numeric_cols = [
            "Recency",
            "Frequency",
            "Monetary",
            "AvgOrderValue",
            "ProductVariety",
            "LifespanDays",
            "CLV_score",
        ]
        available_cols = [c for c in numeric_cols if c in df_copy.columns]

        for cid in range(self.n_clusters):
            c_data = df_copy[df_copy["Cluster"] == cid]
            c_size = len(c_data)
            c_pct = (c_size / total_customers) * 100 if total_customers > 0 else 0.0

            stats: Dict[str, Any] = {
                "persona_name": self.personas.get(cid, f"Cluster {cid}"),
                "strategy": self.strategies.get(cid, "N/A"),
                "customer_count": c_size,
                "percentage_of_base": round(c_pct, 2),
                "metrics": {},
            }

            for col in available_cols:
                stats["metrics"][col] = {
                    "mean": round(float(c_data[col].mean()), 2),
                    "median": round(float(c_data[col].median()), 2),
                    "min": round(float(c_data[col].min()), 2),
                    "max": round(float(c_data[col].max()), 2),
                }

            profiles[cid] = stats

        self.cluster_profiles = profiles

        if save_json_file:
            target_path = Path(config.paths.cluster_profiles_json).resolve()
            save_json(profiles, target_path)
            logger.info(f"Saved cluster profiles JSON to: {target_path}")

        return profiles

    def save(self) -> None:
        """Save fitted KMeans model and PCA transformer to disk."""
        save_joblib(self.model, config.paths.kmeans_model_path)
        save_joblib(self.pca, config.paths.pca_transformer_path)
        logger.info("Saved clustering artifacts (KMeans & PCA) to models/")

    def load(self) -> "CustomerClusterer":
        """Load fitted KMeans model, PCA, and profiles from disk."""
        self.model = load_joblib(config.paths.kmeans_model_path)
        self.pca = load_joblib(config.paths.pca_transformer_path)
        self.n_clusters = self.model.n_clusters
        self.is_fitted = True

        prof_path = Path(config.paths.cluster_profiles_json).resolve()
        if prof_path.is_file():
            from src.utils.helpers import load_json
            self.cluster_profiles = load_json(prof_path)

        logger.info(f"Loaded CustomerClusterer (K={self.n_clusters}) from models/")
        return self