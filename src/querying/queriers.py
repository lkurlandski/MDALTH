"""
"""

from abc import abstractmethod, ABC
from typing import Literal, Optional, TypeAlias

import numpy as np
from numpy import ma
from numpy import random
from scipy.stats import entropy

from querying.utils import multi_argmax


__all__ = ["Wrapper", "Random", "Uncertainty", "BALD"]


ClassWiseProbas: TypeAlias = np.ndarray


class Querier(ABC):
    def __init__(self, length: int, *, initial: Optional[np.ndarray] = None) -> None:
        self._idx = ma.array(np.arange(length, dtype=np.int64), mask=True)
        if initial is not None:
            self._update(initial)

    def __call__(self, n: int, **kwds) -> np.ndarray:
        batch = self._query(n, **kwds)
        self._update(batch)
        return batch

    def _update(self, batch: np.ndarray) -> None:
        self._idx.mask[batch] = ma.nomask

    @property
    def idx(self) -> ma.array:
        return self._idx

    @property
    def labeled(self) -> np.ndarray:
        return self.idx[~self.idx.mask]

    @property
    def unlabeled(self) -> np.ndarray:
        return self.idx[self.idx.mask]

    @abstractmethod
    def _query(self, n: int, *args, **kwds) -> np.ndarray:
        ...


class RandomQuerier(Querier):
    def __init__(self, length: int, **kwds) -> None:
        super().__init__(length, **kwds)

    def _query(self, n: int, *args, **kwds) -> np.ndarray:
        return random.choice(self.idx, size=n, replace=False)


class UncertaintyQuerier(Querier):
    def __init__(self, length: int, mode: Literal["entropy", "margin", "uncertainty"], **kwds) -> None:
        super().__init__(length, **kwds)
        self.mode = mode

    @staticmethod
    def uncertainty(probs: ClassWiseProbas) -> np.ndarray:
        return 1 - np.max(probs, axis=1)

    @staticmethod
    def margin(probs: ClassWiseProbas) -> np.ndarray:
        if probs.shape[1] == 1:
            return np.zeros(shape=(probs.shape[0],))
        part = np.partition(-probs, 1, axis=1)
        margin = -part[:, 0] + part[:, 1]
        return margin

    @staticmethod
    def entropy(probs: ClassWiseProbas) -> np.ndarray:
        return np.transpose(entropy(np.transpose(probs)))

    def _query(self, n: int, probs: ClassWiseProbas) -> np.ndarray:
        if self.mode == "entropy":
            scores = self.entropy(probs)
        elif self.mode == "margin":
            scores = self.margin(probs)
        elif self.mode == "uncertainty":
            scores = self.uncertainty(probs)

        return multi_argmax(scores, n_instances=n)


class BALDQuerier(Querier):
    def __init__(self, length: int, n_drop: int, **kwds) -> None:
        super().__init__(length, **kwds)
        self.n_drop = n_drop

    def _query(self, n: int, probs: np.ndarray, **kwds) -> np.ndarray:
        p = probs.mean(0)
        entropy_1 = (-p * np.log(p)).sum(1)
        entropy_2 = (-probs * np.log(probs)).sum(2).mean(0)
        u = entropy_2 - entropy_1
        return self.unlabeled[u.sort()[1][:n]]
