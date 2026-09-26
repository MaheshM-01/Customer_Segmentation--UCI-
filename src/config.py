"""Configuration loader and management module for Customer Segmentation.

Loads project parameters from config/config.yaml, resolves absolute filesystem paths,
and provides dot-accessible and dictionary-accessible settings across all pipelines.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml

# Project root directory: parent of src/
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "config" / "config.yaml"

# Default feature set in exact order used during modeling
DEFAULT_MODEL_FEATURES: List[str] = [
    "Recency",
    "Frequency",
    "Monetary",
    "AvgOrderValue",
    "ProductVariety",
    "LifespanDays",
    "PurchaseRate",
    "AvgDaysBetweenPurchases",
    "SpendPerProduct",
    "QtyPerOrder",
    "RevenuePerDay",
    "UniqueProductPct",
    "OrderValueConsistency",
    "CLV_score",
]


class DotDict(dict):
    """Dictionary subclass enabling attribute dot notation access (dict.key)."""

    def __getattr__(self, key: str) -> Any:
        try:
            val = self[key]
            if isinstance(val, dict) and not isinstance(val, DotDict):
                val = DotDict(val)
                self[key] = val
            return val
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")


class PathsConfig(DotDict):
    """Container for project file and directory paths resolved as Path objects."""

    raw_data_dir: Path
    raw_data_file: Path
    processed_dir: Path
    models_dir: Path
    logs_dir: Path
    rfm_features_csv: Path
    processed_features_npy: Path
    model_features_txt: Path
    association_rules_csv: Path
    preprocessor_path: Path
    kmeans_model_path: Path
    isolation_forest_model_path: Path
    pca_transformer_path: Path
    cluster_profiles_json: Path

    def ensure_directories(self) -> None:
        """Create required project directories if they do not exist."""
        for path_obj in (self.raw_data_dir, self.processed_dir, self.models_dir, self.logs_dir):
            path_obj.mkdir(parents=True, exist_ok=True)


class Config(DotDict):
    """Main Application Configuration wrapper.

    Provides dot-access and dictionary-access to configuration sections:
    project, paths, cleaning, features, clustering, anomaly_detection,
    market_basket, and api.
    """

    @classmethod
    def load(cls, config_path: Optional[Path | str] = None) -> "Config":
        """Load and parse YAML configuration, resolve paths and normalize settings."""
        cfg_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Configuration file not found at: {cfg_path}")

        with open(cfg_path, "r", encoding="utf-8") as f:
            raw_cfg: Dict[str, Any] = yaml.safe_load(f) or {}

        # 1. Normalize Paths section (resolve all to absolute Path objects)
        paths_raw = raw_cfg.get("paths", {})
        resolved_paths: Dict[str, Path] = {}
        for key, rel_path in paths_raw.items():
            if rel_path is not None:
                clean_key = "association_rules_csv" if key == "association_rules_cvs" else key
                resolved_paths[clean_key] = PROJECT_ROOT / str(rel_path)

        default_path_mappings = {
            "raw_data_dir": PROJECT_ROOT / "data" / "raw" / "online+retail",
            "raw_data_file": PROJECT_ROOT / "data" / "raw" / "online+retail" / "Online Retail.xlsx",
            "processed_dir": PROJECT_ROOT / "data" / "processed",
            "models_dir": PROJECT_ROOT / "models",
            "logs_dir": PROJECT_ROOT / "logs",
            "rfm_features_csv": PROJECT_ROOT / "data" / "processed" / "rfm_features.csv",
            "processed_features_npy": PROJECT_ROOT / "data" / "processed" / "x_processed.npy",
            "model_features_txt": PROJECT_ROOT / "data" / "processed" / "model_features.txt",
            "association_rules_csv": PROJECT_ROOT / "data" / "processed" / "association_rules.csv",
            "preprocessor_path": PROJECT_ROOT / "models" / "preprocessor.joblib",
            "kmeans_model_path": PROJECT_ROOT / "models" / "kmeans_cluster.joblib",
            "cluster_model_path": PROJECT_ROOT / "models" / "kmeans_cluster.joblib",
            "isolation_forest_model_path": PROJECT_ROOT / "models" / "isolation_forest_model.joblib",
            "anomaly_model_path": PROJECT_ROOT / "models" / "isolation_forest_model.joblib",
            "pca_transformer_path": PROJECT_ROOT / "models" / "pca_transformer.joblib",
            "pca_model_path": PROJECT_ROOT / "models" / "pca_transformer.joblib",
            "cluster_profiles_json": PROJECT_ROOT / "models" / "cluster_profiles.json",
            "cluster_profiles_path": PROJECT_ROOT / "models" / "cluster_profiles.json",
            "features_data_path": PROJECT_ROOT / "data" / "processed" / "rfm_features.csv",
        }
        for k, v in default_path_mappings.items():
            resolved_paths.setdefault(k, v)

        paths_obj = PathsConfig(resolved_paths)
        paths_obj.ensure_directories()

        # 2. Normalize Features section
        features_raw = raw_cfg.get("features", {})
        raw_mf = features_raw.get("model_features")
        if isinstance(raw_mf, str):
            tokens = re.findall(r'["\']?([A-Za-z0-9_]+)["\']?', raw_mf)
            extracted = [t for t in tokens if t]
            feature_list = extracted if extracted else DEFAULT_MODEL_FEATURES
        elif isinstance(raw_mf, list):
            feature_list = [str(x).strip() for x in raw_mf]
        else:
            feature_list = DEFAULT_MODEL_FEATURES

        canonical_features = []
        for feat in feature_list:
            if feat.lower() == "productvariety":
                canonical_features.append("ProductVariety")
            else:
                canonical_features.append(feat)

        features_obj = DotDict(features_raw)
        features_obj["model_features"] = canonical_features
        features_obj.setdefault("n_rfm_bins", 5)
        features_obj.setdefault("clipping_lower_quantile", 0.01)
        features_obj.setdefault("clipping_upper_quantile", 0.99)

        # 3. Normalize Clustering section
        clustering_raw = raw_cfg.get("clustering", {})
        clustering_obj = DotDict(clustering_raw)
        personas = clustering_raw.get("personas", {})
        strategies = clustering_raw.get("strategies", {})

        if isinstance(personas, dict) and "strategies" in personas:
            strategies = personas.pop("strategies")

        clean_personas = {int(k): str(v) for k, v in personas.items() if str(k).isdigit()}
        clean_strategies = {int(k): str(v) for k, v in strategies.items() if str(k).isdigit()}

        clustering_obj["personas"] = clean_personas
        clustering_obj["strategies"] = clean_strategies
        clustering_obj.setdefault("n_clusters", clustering_raw.get("n_cluster", 4))
        clustering_obj.setdefault("init_method", "k-means++")
        clustering_obj.setdefault("n_init", 20)
        clustering_obj.setdefault("max_iter", 300)
        clustering_obj.setdefault("random_state", 42)

        # 4. Normalize Anomaly Detection section
        anomaly_raw = raw_cfg.get("anomaly_detection") or raw_cfg.get("anomaly") or {}
        anomaly_obj = DotDict(anomaly_raw)
        anomaly_obj.setdefault("contamination", 0.03)
        anomaly_obj.setdefault("n_estimators", 100)
        anomaly_obj.setdefault("random_state", 42)
        anomaly_obj.setdefault("n_jobs", -1)

        # 5. Cleaning, Market Basket, API, and Project sections
        cleaning_raw = raw_cfg.get("cleaning", {})
        cleaning_obj = DotDict(cleaning_raw)
        cleaning_obj.setdefault("drop_missing_customer_id", cleaning_raw.get("drop_missing_cluster_id", True))
        cleaning_obj.setdefault("min_quantity", cleaning_raw.get("min_quality", 1))
        cleaning_obj.setdefault("min_unit_price", 0.001)
        cleaning_obj.setdefault("outlier_quantile", 0.999)
        cleaning_obj.setdefault("cancellation_prefix", cleaning_raw.get("cancellation_prefiz", "C"))

        market_basket_raw = raw_cfg.get("market_basket", {})
        market_basket_obj = DotDict(market_basket_raw)
        market_basket_obj.setdefault("top_n_products", 100)
        market_basket_obj.setdefault("min_support", 0.01)
        market_basket_obj.setdefault("min_confidence", 0.3)
        market_basket_obj.setdefault("min_lift", 1.0)

        api_raw = raw_cfg.get("api", {})
        api_obj = DotDict(api_raw)
        api_obj.setdefault("host", "0.0.0.0")
        api_obj.setdefault("port", 8000)
        api_obj.setdefault("title", "Customer Segmentation & Analytics API")

        project_raw = raw_cfg.get("project", {})
        project_obj = DotDict(project_raw)
        project_obj.setdefault("name", "customer-segmentation")
        project_obj.setdefault("version", "0.1.0")
        project_obj.setdefault("random_seed", 42)

        mlflow_raw = raw_cfg.get("mlflow", {})
        mlflow_obj = DotDict(mlflow_raw)
        mlflow_obj.setdefault("enabled", True)
        mlflow_obj.setdefault("tracking_uri", "mlruns")
        mlflow_obj.setdefault("experiment_name", "Customer_Segmentation_UCI")
        mlflow_obj.setdefault("registered_model_name", "Customer-Segmentation-Production")
        mlflow_obj.setdefault("log_models", True)

        monitoring_raw = raw_cfg.get("monitoring", {})
        monitoring_obj = DotDict(monitoring_raw)
        monitoring_obj.setdefault("baseline_features_csv", "data/processed/rfm_features.csv")
        monitoring_obj.setdefault("drift_threshold_psi", 0.25)
        monitoring_obj.setdefault("drift_threshold_ks_pvalue", 0.05)
        monitoring_obj.setdefault("cluster_shift_threshold", 0.10)

        config_dict = {
            "project": project_obj,
            "paths": paths_obj,
            "cleaning": cleaning_obj,
            "features": features_obj,
            "clustering": clustering_obj,
            "anomaly_detection": anomaly_obj,
            "anomaly": anomaly_obj,
            "market_basket": market_basket_obj,
            "api": api_obj,
            "mlflow": mlflow_obj,
            "monitoring": monitoring_obj,
        }
        return cls(config_dict)


# Global singleton instance loaded once on import
config: Config = Config.load()


def get_config(reload: bool = False) -> Config:
    """Return the global configuration instance, optionally reloading from disk."""
    global config
    if reload:
        config = Config.load()
    return config