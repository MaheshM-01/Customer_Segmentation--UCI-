"""End-to-end training and inference pipeline orchestration."""

from src.pipeline.train_pipeline import run_train_pipeline
from src.pipeline.inference_pipeline import (
    CustomerSegmentationService,
    get_inference_service,
)

__all__ = [
    "run_train_pipeline",
    "CustomerSegmentationService",
    "get_inference_service",
]
