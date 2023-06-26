"""
Manage the labeled and unlabeled pools.
"""

from __future__ import annotations

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

    @classmethod  # TODO: TEST
    def from_ma(cls, idx: ma.MaskedArray) -> Pool:
        pool = cls(len(idx))
        pool.idx.mask = idx.mask

    @classmethod  # TODO: TEST
    def from_pools(cls, labeled: np.ndarray, unlabeled: np.ndarray) -> Pool:
        pool = cls(len(labeled) + len(unlabeled))
        pool.idx.mask = unlabeled

    def label(self, idx: np.ndarray) -> None:
        if not np.issubdtype(idx.dtype, np.integer):
            raise TypeError(f"Indexing with non-int dtype: {idx.dtype}." "")
        self.idx.mask[idx] = ma.nomask

    def unlabel(self, idx: np.ndarray) -> None:
        if not np.issubdtype(idx.dtype, np.integer):
            raise TypeError(f"Indexing with non-int dtype: {idx.dtype}." "")
        self.idx.mask[idx] = True
