"""Utility and helper functions (file I/O, serialization, timing)."""

from __future__ import annotations

import functools
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Union

import joblib
import numpy as np

from src.utils.logger import logger


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to safely serialize NumPy scalar types and arrays."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super().default(obj)


def timeit(func: Callable) -> Callable:
    """Decorator to measure and log function execution duration."""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        func_name = func.__name__
        logger.info(f"Starting [{func_name}]...")
        try:
            result = func(*args, **kwargs)
            duration = time.perf_counter() - start_time
            logger.info(f"Completed [{func_name}] in {duration:.2f}s")
            return result
        except Exception as exc:
            duration = time.perf_counter() - start_time
            logger.error(f"Failed [{func_name}] after {duration:.2f}s: {exc}")
            raise
    return wrapper


def save_joblib(obj: Any, filepath: Union[Path, str]) -> Path:
    """Serialize an object (model, transformer) to disk using joblib.

    Parameters
    ----------
    obj : Any
        Python object (e.g. Scikit-learn estimator / pipeline) to serialize.
    filepath : Path or str
        Destination path.

    Returns
    -------
    Path
        Resolved saved file path.
    """
    path = Path(filepath).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    size_mb = path.stat().st_size / (1024 * 1024)
    logger.info(f"Saved joblib model to: {path} ({size_mb:.2f} MB)")
    return path


def load_joblib(filepath: Union[Path, str]) -> Any:
    """Deserialize an object from disk using joblib.

    Parameters
    ----------
    filepath : Path or str
        Source file path.

    Returns
    -------
    Any
        Loaded Python object.
    """
    path = Path(filepath).resolve()
    if not path.is_file():
        err_msg = f"Joblib file not found at: {path}"
        logger.error(err_msg)
        raise FileNotFoundError(err_msg)
    obj = joblib.load(path)
    logger.info(f"Loaded joblib model from: {path}")
    return obj


def save_json(data: Union[Dict[str, Any], List[Any]], filepath: Union[Path, str], indent: int = 4) -> Path:
    """Save dictionary or list data to a JSON file, converting NumPy types safely."""
    path = Path(filepath).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, cls=NumpyEncoder, ensure_ascii=False)
    logger.info(f"Saved JSON data to: {path}")
    return path


def load_json(filepath: Union[Path, str]) -> Any:
    """Load JSON file from disk into a Python dict or list."""
    path = Path(filepath).resolve()
    if not path.is_file():
        err_msg = f"JSON file not found at: {path}"
        logger.error(err_msg)
        raise FileNotFoundError(err_msg)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Loaded JSON data from: {path}")
    return data


def save_numpy(arr: np.ndarray, filepath: Union[Path, str]) -> Path:
    """Save NumPy array to disk in .npy binary format."""
    path = Path(filepath).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    size_mb = path.stat().st_size / (1024 * 1024)
    logger.info(f"Saved NumPy array ({arr.shape}) to: {path} ({size_mb:.2f} MB)")
    return path


def load_numpy(filepath: Union[Path, str]) -> np.ndarray:
    """Load NumPy array from .npy file."""
    path = Path(filepath).resolve()
    if not path.is_file():
        err_msg = f"NumPy file not found at: {path}"
        logger.error(err_msg)
        raise FileNotFoundError(err_msg)
    arr = np.load(path)
    logger.info(f"Loaded NumPy array {arr.shape} from: {path}")
    return arr
