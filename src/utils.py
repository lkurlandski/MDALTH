"""
"""

from typing import Any, Optional
from pathlib import Path
import pickle


def is_directory_empty(path: Path) -> bool:
    for _ in path.iterdir():
        return False
    return True


def pickler(path: Path, obj: Optional[Any] = None, **kwds) -> Any:
    if obj is None:
        with open(path, "rb") as handle:
            return pickle.load(handle, **kwds)
    with open(path, "wb") as handle:
        pickle.dump(obj, handle, **kwds)
    return None
