"""
Utility functions.
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


def get_highest_path(
    path_or_files: Collection[Path] | Path, lstrip: str = "", rstrip: str = ""
) -> Path:
    if isinstance(path_or_files, (Path, str)):
        files = Path(path_or_files).iterdir()
    else:
        files = path_or_files
    return list(sorted(files, key=lambda p: int(p.stem.lstrip(lstrip).rstrip(rstrip))))[-1]


def probs_to_confs(probs: np.ndarray) -> np.ndarray:
    return np.max(probs, axis=1)


def probs_to_preds(probs: np.ndarray) -> np.ndarray:
    return np.argmax(probs, axis=1)
