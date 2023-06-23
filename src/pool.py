"""
Manage the labeled and unlabeled pools.
"""

import warnings

import numpy as np
from numpy import ma


class Pool:
    def __init__(self, n: int) -> None:
        self.idx = ma.array(np.arange(n, dtype=int), mask=True)

    def __len__(self) -> int:
        return len(self.idx)

    def __str__(self) -> str:
        return f"labeled={self.labeled}\nunlabeled={self.unlabeled}"

    @property
    def labeled(self) -> np.ndarray:
        return self.idx.compressed()

    @property
    def unlabeled(self) -> np.ndarray:
        return ma.getdata(self.idx[self.idx.mask])

    def label(self, idx: np.ndarray) -> None:
        if not np.issubdtype(idx.dtype, np.integer):
            raise TypeError(f"Indexing with non-int dtype: {idx.dtype}." "")
        self.idx.mask[idx] = ma.nomask

    def unlabel(self, idx: np.ndarray) -> None:
        if not np.issubdtype(idx.dtype, np.integer):
            raise TypeError(f"Indexing with non-int dtype: {idx.dtype}." "")
        self.idx.mask[idx] = True
