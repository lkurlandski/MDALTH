"""Useful functions and classes.
"""

from collections.abc import Collection
from pathlib import Path
import pickle
from typing import Any

import numpy as np


def save_with_pickle(path: Path, obj: Any, **kwds) -> None:
    with open(path, "wb") as handle:
        pickle.dump(obj, handle, **kwds)


def load_with_pickle(path: Path, **kwds) -> Any:
    with open(path, "rb") as handle:
        return pickle.load(handle, **kwds)


def is_directory_empty(path: Path) -> bool:
    for _ in path.iterdir():
        return False
    return True


def get_highest_path(path_or_files: Collection[Path] | Path, lstrip: str = "", rstrip: str = "") -> Path:
    if isinstance(path_or_files, (Path, str)):
        files = Path(path_or_files).iterdir()
    else:
        files = path_or_files
    return list(sorted(files, key=lambda p: int(p.stem.lstrip(lstrip).rstrip(rstrip))))[-1]


def shuffled_argmax(values: np.ndarray, n_instances: int = 1) -> np.ndarray:
    shuffled_idx = np.random.permutation(len(values))
    shuffled_values = values[shuffled_idx]
    sorted_query_idx = np.argsort(shuffled_values, kind="mergesort")[len(shuffled_values) - n_instances :]
    query_idx = shuffled_idx[sorted_query_idx]
    return query_idx, values[query_idx]


def shuffled_argmin(values: np.ndarray, n_instances: int = 1) -> np.ndarray:
    indexes, index_values = shuffled_argmax(-values, n_instances)
    return indexes, -index_values


def multi_argmax(values: np.ndarray, n_instances: int = 1) -> np.ndarray:
    max_idx = np.argpartition(-values, n_instances - 1, axis=0)[:n_instances]
    return max_idx, values[max_idx]


def multi_argmin(values: np.ndarray, n_instances: int = 1) -> np.ndarray:
    indexes, index_values = multi_argmax(-values, n_instances)
    return indexes, -index_values
