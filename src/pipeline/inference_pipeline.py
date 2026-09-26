"""Batch and single customer inference pipeline for production deployment."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from src.config import config
from src.utils.logger import logger
from src.utils.helpers import load_joblib, load_json
from src.models.market_basket import recommend_products


class CustomerSegmentationService:
    """Production scoring and inference engine for Customer Segmentation & Analytics."""

    def __init__(self) -> None:
        self.preprocessor = None
        self.kmeans = None
        self.pca = None
        self.anomaly_detector = None
        self.cluster_profiles: Dict[str, Any] = {}
        self.association_rules_df: pd.DataFrame = pd.DataFrame()
        self.personas: Dict[int, str] = dict(config.clustering.personas)
        self.strategies: Dict[int, str] = dict(config.clustering.strategies)
        self.feature_names: List[str] = list(config.features.model_features)
        self.is_loaded = False

    def load_artifacts(self) -> "CustomerSegmentationService":
        """Load all serialized production model artifacts from disk."""
        if self.is_loaded:
            return self

        logger.info("Loading inference artifacts into CustomerSegmentationService...")

        self.preprocessor = load_joblib(config.paths.preprocessor_path)
        self.kmeans = load_joblib(config.paths.kmeans_model_path)
        self.pca = load_joblib(config.paths.pca_transformer_path)
        self.anomaly_detector = load_joblib(config.paths.isolation_forest_model_path)

        prof_path = Path(config.paths.cluster_profiles_json).resolve()
        if prof_path.is_file():
            self.cluster_profiles = load_json(prof_path)

        rules_path = Path(config.paths.association_rules_csv).resolve()
        if rules_path.is_file():
            self.association_rules_df = pd.read_csv(rules_path)
            logger.info(f"Loaded {len(self.association_rules_df):,} association rules for recommendations.")

        self.is_loaded = True
        logger.info("All inference artifacts loaded successfully.")
        return self

    def _prepare_feature_dataframe(
        self,
        features: Union[Dict[str, Any], pd.DataFrame],
    ) -> pd.DataFrame:
        """Validate and format input features into standard DataFrame."""
        if isinstance(features, dict):
            df = pd.DataFrame([features])
        else:
            df = features.copy()

        missing = [f for f in self.feature_names if f not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required customer features for inference: {missing}\n"
                f"Expected features: {self.feature_names}"
            )

        return df[self.feature_names]

    def predict_customer(self, customer_features: Dict[str, Any]) -> Dict[str, Any]:
        """Score a single customer and return segment, persona, strategy, and anomaly status."""
        self.load_artifacts()
        df_feat = self._prepare_feature_dataframe(customer_features)

        X_scaled = self.preprocessor.transform(df_feat)

        cluster_id = int(self.kmeans.predict(X_scaled)[0])
        persona_name = self.personas.get(cluster_id, f"Cluster {cluster_id}")
        strategy = self.strategies.get(cluster_id, "Standard customer retention")

        pca_coords = self.pca.transform(X_scaled)[0]
        pca_x = round(float(pca_coords[0]), 4)
        pca_y = round(float(pca_coords[1]), 4)

        is_anomaly = bool(self.anomaly_detector.predict(X_scaled)[0] == -1)
        anomaly_score = round(float(self.anomaly_detector.decision_function(X_scaled)[0]), 4)

        return {
            "cluster_id": cluster_id,
            "persona": persona_name,
            "strategy": strategy,
            "pca_x": pca_x,
            "pca_y": pca_y,
            "is_anomaly": is_anomaly,
            "anomaly_score": anomaly_score,
            "profile_summary": self.cluster_profiles.get(str(cluster_id), {}),
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Batch score an entire DataFrame of customers."""
        self.load_artifacts()
        df_feat = self._prepare_feature_dataframe(df)

        X_scaled = self.preprocessor.transform(df_feat)

        clusters = self.kmeans.predict(X_scaled)
        pca_2d = self.pca.transform(X_scaled)
        anomaly_preds = self.anomaly_detector.predict(X_scaled)
        anomaly_scores = self.anomaly_detector.decision_function(X_scaled)

        result_df = df.copy()
        result_df["Cluster"] = clusters
        result_df["Persona"] = [self.personas.get(int(c), f"Cluster {c}") for c in clusters]
        result_df["Strategy"] = [self.strategies.get(int(c), "Standard retention") for c in clusters]
        result_df["PCA_1"] = np.round(pca_2d[:, 0], 4)
        result_df["PCA_2"] = np.round(pca_2d[:, 1], 4)
        result_df["Is_Anomaly"] = anomaly_preds == -1
        result_df["Anomaly_Score"] = np.round(anomaly_scores, 4)

        return result_df

    def get_recommendations(
        self,
        purchased_items: List[str],
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Get product recommendations for items currently in cart using Market Basket Analysis."""
        self.load_artifacts()
        return recommend_products(purchased_items, self.association_rules_df, top_k=top_k)


# Cached singleton service instance
_inference_service: Optional[CustomerSegmentationService] = None


def get_inference_service() -> CustomerSegmentationService:
    """Retrieve or initialize the global CustomerSegmentationService singleton."""
    global _inference_service
    if _inference_service is None:
        _inference_service = CustomerSegmentationService().load_artifacts()
    return _inference_service
