from __future__ import annotations
import argparse
from typing import Any, Dict
import pandas as pd
from src.config import config
from src.utils.logger import logger
from src.utils.helpers import timeit
from src.utils.mlflow_tracker import MLflowTracker
from src.data.loader import load_raw_data
from src.data.cleaner import clean_data
from src.features.rfm_builder import build_rfm_features
from src.features.pipeline import fit_transform_features
from src.models.clustering import CustomerClusterer
from src.models.evaluation import evaluate_clustering, explain_clusters_surrogate
from src.models.anomaly import CustomerAnomalyDetector
from src.models.market_basket import run_market_basket_analysis


@timeit
def run_train_pipeline(skip_market_basket: bool = False) -> Dict[str, Any]:
    logger.info("=" * 70)
    logger.info("STARTING END-TO-END CUSTOMER SEGMENTATION TRAINING PIPELINE")
    logger.info("=" * 70)

    tracker: MLflowTracker | None = None
    if config.mlflow.enabled:
        try:
            tracker = MLflowTracker()
            tracker.start_run(run_name="UCI_Customer_Segmentation_Production")
            tracker.log_params({
                "clustering_algorithm": "KMeans",
                "n_clusters": config.clustering.n_clusters,
                "init_method": config.clustering.init_method,
                "n_init": config.clustering.n_init,
                "max_iter": config.clustering.max_iter,
                "random_state": config.clustering.random_state,
                "pca_n_components": getattr(config.clustering, "pca_n_components", 2),
                "anomaly_contamination": config.anomaly.contamination,
                "features_n_rfm_bins": config.features.n_rfm_bins,
                "features_list": list(config.features.model_features),
                "mba_min_support": config.market_basket.min_support,
                "mba_min_confidence": config.market_basket.min_confidence,
                "mba_min_lift": config.market_basket.min_lift,
            })
        except Exception as e:
            logger.warning(f"Could not initialize or start MLflow run: {e}. Continuing without tracking.")
            tracker = None

    try:
        logger.info("Step 1: Loading raw transactions...")
        raw_df = load_raw_data()

        logger.info("Step 2: Cleaning transactions and filtering cancellations...")
        clean_df = clean_data(raw_df)

        logger.info("Step 3: Building customer-level RFM and behavioral metrics...")
        rfm_df = build_rfm_features(clean_df, save_csv=True)

        logger.info("Step 4: Fitting feature preprocessing and transformations...")
        x_scaled, preprocessor = fit_transform_features(rfm_df, save_artifacts=True)

        logger.info("Step 5: Training customer clustering engine (K-Means + PCA)...")
        clusterer = CustomerClusterer()
        clusterer.fit(x_scaled, customer_df=rfm_df, save_artifacts=True)
        cluster_labels = clusterer.predict(x_scaled)

        eval_metrics = evaluate_clustering(x_scaled, cluster_labels)
        features_cols = list(config.features.model_features)
        explainability = explain_clusters_surrogate(rfm_df[features_cols], cluster_labels)

        logger.info("Step 6: Training Isolation Forest anomaly detector...")
        anomaly_detector = CustomerAnomalyDetector()
        anomaly_detector.fit(x_scaled, save_artifact=True)
        is_anomaly_mask = anomaly_detector.predict(x_scaled) == -1
        anomaly_summary = anomaly_detector.profile_anomalies(rfm_df, is_anomaly_mask)

        # Step 7: Market Basket Analysis (Association Rules)
        mba_rules_count = 0
        if not skip_market_basket:
            logger.info("Step 7: Running Market Basket Analysis (FP-Growth)...")
            rules_df = run_market_basket_analysis(clean_df, save_csv=True)
            mba_rules_count = len(rules_df)
        else:
            logger.info("Step 7: Skipping Market Basket Analysis (--skip-market-basket set).")

        # Step 8: MLflow Logging (Metrics, Visualizations, Artifacts, Model Registry)
        if tracker:
            logger.info("Step 8: Logging metrics and artifacts to MLflow...")
            tracker.log_metrics({
                "n_customers": len(rfm_df),
                "silhouette_score": eval_metrics["silhouette_score"],
                "davies_bouldin_index": eval_metrics["davies_bouldin_index"],
                "calinski_harabasz_index": eval_metrics["calinski_harabasz_index"],
                "kmeans_inertia": float(clusterer.model.inertia_),
                "surrogate_fidelity_accuracy": explainability.get("tree_accuracy", 0.0),
                "anomaly_count": anomaly_summary.get("anomaly_count", 0),
                "anomaly_revenue_share_pct": anomaly_summary.get("anomaly_revenue_share_pct", 0.0),
                "association_rules_count": mba_rules_count,
            })

            # 2D PCA cluster visualization
            pca_2d = clusterer.transform_pca(x_scaled)
            tracker.plot_and_log_clusters_2d(
                pca_coords=pca_2d,
                cluster_labels=cluster_labels,
                personas=clusterer.personas,
            )

            # Feature importance plot
            if "feature_importances" in explainability:
                tracker.plot_and_log_feature_importances(explainability["feature_importances"])

            # Log saved model artifacts
            tracker.log_artifact(config.paths.preprocessor_path, artifact_path="models")
            tracker.log_artifact(config.paths.cluster_model_path, artifact_path="models")
            tracker.log_artifact(config.paths.pca_model_path, artifact_path="models")
            tracker.log_artifact(config.paths.anomaly_model_path, artifact_path="models")
            tracker.log_artifact(config.paths.cluster_profiles_path, artifact_path="models")
            if config.paths.features_data_path.exists():
                tracker.log_artifact(config.paths.features_data_path, artifact_path="data")

            # Register production KMeans model
            tracker.log_sklearn_model(
                model=clusterer.model,
                artifact_path="kmeans_model",
                registered_model_name=config.mlflow.registered_model_name,
            )

        logger.info("=" * 70)
        logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info(f"  - Total customers processed : {len(rfm_df):,}")
        logger.info(f"  - Silhouette Score          : {eval_metrics['silhouette_score']:.4f}")
        logger.info(f"  - Davies-Bouldin Index      : {eval_metrics['davies_bouldin_index']:.4f}")
        logger.info(f"  - Calinski-Harabasz Index   : {eval_metrics['calinski_harabasz_index']:.1f}")
        logger.info(f"  - Surrogate Fidelity        : {explainability.get('tree_accuracy', 0.0) * 100:.2f}%")
        logger.info(f"  - Anomalies Flagged         : {anomaly_summary['anomaly_count']:,} ({anomaly_summary['anomaly_revenue_share_pct']}% revenue)")
        logger.info(f"  - Association Rules Created : {mba_rules_count:,}")
        logger.info("=" * 70)

        return {
            "n_customers": len(rfm_df),
            "evaluation_metrics": eval_metrics,
            "explainability": explainability,
            "anomalies": anomaly_summary,
            "association_rules_count": mba_rules_count,
        }
    finally:
        if tracker:
            tracker.end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Customer Segmentation Training Pipeline")
    parser.add_argument(
        "--skip-market-basket",
        action="store_true",
        help="Skip FP-Growth Market Basket Analysis for faster training run",
    )
    args = parser.parse_args()
    run_train_pipeline(skip_market_basket=args.skip_market_basket)


