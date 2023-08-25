"""
Query algorithms for active learning.
"""

from typing import Literal, Protocol

import numpy as np
from numpy import random
from scipy.stats import entropy


class Querier(Protocol):
    def __call__(self, *args, **kwds) -> np.ndarray:
        ...


class RandomQuerier:
    def __call__(self, n_query: int, unlabeled_idx: np.ndarray) -> np.ndarray:
        return random.choice(unlabeled_idx, size=n_query, replace=False)


class UncertaintyQuerier:
    def __init__(self, mode: Literal["E", "M", "U"]) -> None:
        self.mode = mode

    def __call__(self, n_query: int, classwise_probs: np.ndarray) -> np.ndarray:
        if self.mode == "E":
            scores = self.entropy(classwise_probs)
        elif self.mode == "M":
            scores = self.margin(classwise_probs)
        elif self.mode == "U":
            scores = self.uncertainty(classwise_probs)
        return np.flip(scores.argsort())[:n_query]

    @staticmethod
    def uncertainty(probs: np.ndarray) -> np.ndarray:
        return 1 - np.max(probs, axis=1)

    @staticmethod
    def margin(probs: np.ndarray) -> np.ndarray:
        if probs.shape[1] == 1:
            return np.zeros(shape=(probs.shape[0],))
        part = np.partition(-probs, 1, axis=1)
        margin = -part[:, 0] + part[:, 1]
        return margin

    @staticmethod
    def entropy(probs: np.ndarray) -> np.ndarray:
        return np.transpose(entropy(np.transpose(probs)))
