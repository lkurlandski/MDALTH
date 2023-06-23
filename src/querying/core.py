"""Query algorithms for active learning.
"""

from abc import abstractmethod, ABC
from typing import Literal, Optional, Protocol, TypeAlias

import numpy as np
from numpy import ma
from numpy import random
from scipy.stats import entropy

from src.querying.utils import multi_argmax


__all__ = [
    "Querier",
    "RandomQuerier",
    "UncertaintyQuerier",
]


ClassWiseProbas: TypeAlias = np.ndarray


class Querier(Protocol):
    def __call__(self, *args, **kwds) -> np.ndarray:
        ...


class RandomQuerier:
    def __call__(self, n: int, unlabeled_idx: np.ndarray) -> np.ndarray:
        return random.choice(unlabeled_idx, size=n, replace=False)


class UncertaintyQuerier:
    def __init__(self, mode: Literal["E", "M", "U"]) -> None:
        self.mode = mode

    def __call__(self, n: int, probs: ClassWiseProbas) -> np.ndarray:
        if self.mode == "E":
            scores = self.entropy(probs)
        elif self.mode == "M":
            scores = self.margin(probs)
        elif self.mode == "U":
            scores = self.uncertainty(probs)

        return multi_argmax(scores, n_instances=n)

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


# class BALDQuerier(Querier):
#     def __init__(self, n_drop: int) -> None:
#         self.n_drop = n_drop

#     def __call__(self, n: int, probs: np.ndarray) -> np.ndarray:
#         p = probs.mean(0)
#         entropy_1 = (-p * np.log(p)).sum(1)
#         entropy_2 = (-probs * np.log(probs)).sum(2).mean(0)
#         u = entropy_2 - entropy_1
#         return self.unlabeled[u.sort()[1][:n]]
