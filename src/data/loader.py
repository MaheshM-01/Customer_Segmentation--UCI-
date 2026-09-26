"""Raw dataset loading and schema validation."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union
import pandas as pd

from src.config import config
from src.utils.logger import logger
from src.utils.helpers import timeit

REQUIRED_COLUMNS: List[str] = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
]


def validate_schema(df: pd.DataFrame) -> bool:
    """Validate that the DataFrame contains all mandatory UCI Online Retail columns."""
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        err_msg = f"Schema validation failed! Missing required columns: {missing_cols}"
        logger.error(err_msg)
        raise ValueError(err_msg)

    logger.info(f"Schema validation passed ({len(df):,} rows x {len(df.columns)} columns).")
    return True


@timeit
def load_raw_data(filepath: Optional[Union[Path, str]] = None) -> pd.DataFrame:
    """Load the raw UCI Online Retail dataset and validate its schema."""
    path = Path(filepath).resolve() if filepath else config.paths.raw_data_file

    if not path.is_file():
        err_msg = (
            f"Raw dataset file not found at: {path}\n"
            "Please ensure 'Online Retail.xlsx' is placed in 'data/raw/online+retail/'."
        )
        logger.error(err_msg)
        raise FileNotFoundError(err_msg)

    logger.info(f"Loading raw dataset from: {path}")

    suffix = path.suffix.lower()
    if suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(path, engine="openpyxl")
    elif suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Expected .xlsx, .xls, or .csv")

    validate_schema(df)

    memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    logger.info(
        f"Successfully loaded dataset: {len(df):,} rows, {len(df.columns)} cols, {memory_mb:.2f} MB"
    )

    return df