"""MLflow experiment tracking, artifact lineage, and model registry manager."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import mlflow
import mlflow.sklearn

from src.config import config, PROJECT_ROOT
from src.utils.logger import logger

# Configure MLflow runtime environment
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"


class MLflowTracker:
    """Wrapper around MLflow for experiment tracking, visualization, and model registry."""

    def __init__(
        self,
        experiment_name: Optional[str] = None,
        tracking_uri: Optional[str] = None,
    ) -> None:
        self.experiment_name = (
            experiment_name
            if experiment_name is not None
            else config.mlflow.experiment_name
        )
        raw_uri = (
            tracking_uri
            if tracking_uri is not None
            else config.mlflow.tracking_uri
        )

        # Resolve tracking URI to absolute path
        if raw_uri.startswith(("http://", "https://", "databricks://")):
            self.resolved_uri = raw_uri
        elif raw_uri.startswith("sqlite:"):
            db_rel = raw_uri.replace("sqlite:///", "").replace("sqlite://", "")
            db_path = (PROJECT_ROOT / db_rel).resolve().as_posix()
            self.resolved_uri = f"sqlite:///{db_path}"
        elif raw_uri.endswith(".db"):
            db_path = (PROJECT_ROOT / raw_uri).resolve().as_posix()
            self.resolved_uri = f"sqlite:///{db_path}"
        else:
            db_path = (PROJECT_ROOT / "mlflow.db").resolve().as_posix()
            self.resolved_uri = f"sqlite:///{db_path}"

        mlflow.set_tracking_uri(self.resolved_uri)
        mlflow.set_experiment(self.experiment_name)
        self.active_run = None
        logger.info(
            f"MLflowTracker initialized: experiment='{self.experiment_name}', "
            f"tracking_uri='{self.resolved_uri}'"
        )

    def start_run(self, run_name: Optional[str] = None) -> mlflow.ActiveRun:
        """Start a new MLflow tracking run."""
        self.active_run = mlflow.start_run(run_name=run_name)
        logger.info(f"Started MLflow run: ID={self.active_run.info.run_id}, Name='{run_name}'")
        return self.active_run

    def end_run(self) -> None:
        """End the active MLflow run."""
        if mlflow.active_run():
            mlflow.end_run()
            logger.info("Ended active MLflow run.")
            self.active_run = None

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log key-value hyperparameters to active MLflow run."""
        flat_params = {}
        for k, v in params.items():
            if isinstance(v, (list, tuple, dict)):
                flat_params[k] = str(v)[:250]  # truncate long structures
            else:
                flat_params[k] = v
        mlflow.log_params(flat_params)
        logger.info(f"Logged {len(flat_params)} parameters to MLflow.")

    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None) -> None:
        """Log evaluation metrics to MLflow."""
        clean_metrics = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float, np.floating, np.integer))}
        mlflow.log_metrics(clean_metrics, step=step)
        logger.info(f"Logged {len(clean_metrics)} metrics to MLflow: {clean_metrics}")

    def log_artifact(self, file_path: Union[Path, str], artifact_path: Optional[str] = None) -> None:
        """Log a local file artifact (JSON, CSV, model) to MLflow."""
        path = Path(file_path).resolve()
        if path.is_file():
            mlflow.log_artifact(str(path), artifact_path=artifact_path)
            logger.info(f"Logged artifact to MLflow: {path.name}")
        else:
            logger.warning(f"Artifact not found for MLflow logging: {path}")

    def log_sklearn_model(
        self,
        model: Any,
        artifact_path: str = "model",
        registered_model_name: Optional[str] = None,
    ) -> None:
        """Log a scikit-learn model and optionally register it to Model Registry."""
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name,
        )
        logger.info(f"Logged scikit-learn model under artifact path: '{artifact_path}'")

    def plot_and_log_clusters_2d(
        self,
        pca_coords: np.ndarray,
        cluster_labels: np.ndarray,
        personas: Dict[int, str],
        filename: str = "cluster_pca_scatter.png",
    ) -> Path:
        """Generate and log 2D PCA cluster visualization plot."""
        temp_dir = PROJECT_ROOT / "logs" / "plots"
        temp_dir.mkdir(parents=True, exist_ok=True)
        save_path = temp_dir / filename

        plt.figure(figsize=(10, 7))
        palette = sns.color_palette("tab10", len(personas))

        for cid in sorted(personas.keys()):
            mask = cluster_labels == cid
            label_text = f"Cluster {cid}: {personas.get(cid, f'Segment {cid}')} ({mask.sum():,} pts)"
            plt.scatter(
                pca_coords[mask, 0],
                pca_coords[mask, 1],
                c=[palette[cid % len(palette)]],
                label=label_text,
                alpha=0.6,
                edgecolors="none",
                s=25,
            )

        plt.title("Customer Segments in 2D Latent Space (PCA)", fontsize=14, weight="bold")
        plt.xlabel("Principal Component 1 (Spend & Frequency Scale)", fontsize=11)
        plt.ylabel("Principal Component 2 (Tenure & Velocity Scale)", fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", frameon=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

        self.log_artifact(save_path, artifact_path="visualizations")
        return save_path

    def plot_and_log_feature_importances(
        self,
        feature_importances: Dict[str, float],
        filename: str = "surrogate_feature_importance.png",
    ) -> Path:
        """Generate and log surrogate Decision Tree feature importance bar chart."""
        temp_dir = PROJECT_ROOT / "logs" / "plots"
        temp_dir.mkdir(parents=True, exist_ok=True)
        save_path = temp_dir / filename

        sorted_feats = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
        names = [x[0] for x in sorted_feats if x[1] > 0]
        values = [x[1] for x in sorted_feats if x[1] > 0]

        plt.figure(figsize=(9, 5))
        sns.barplot(x=values, y=names, hue=names, palette="viridis", legend=False)
        plt.title("Feature Importance in Defining Customer Segments", fontsize=13, weight="bold")
        plt.xlabel("Importance Ratio (Gini Reduction)", fontsize=10)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

        self.log_artifact(save_path, artifact_path="visualizations")
        return save_path
