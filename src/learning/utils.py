"""Useful functions and classes.
"""

from typing import Any, Optional
from pathlib import Path
import pickle


def is_directory_empty(path: Path) -> bool:
    for _ in path.iterdir():
        return False
    return True


def save_with_pickle(path: Path, obj: Any, **kwds) -> None:
    with open(path, "wb") as handle:
        pickle.dump(obj, handle, **kwds)


def load_with_pickle(path: Path, **kwds) -> Any:
    with open(path, "rb") as handle:
        return pickle.load(handle, **kwds)


class SaveLoadPickleMixin:
    @classmethod
    def load(cls, path: Path, **kwds) -> Any:
        return load_with_pickle(path, **kwds)

    def save(self, path: Path, **kwds) -> None:
        save_with_pickle(path, self, **kwds)
