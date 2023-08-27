"""
Core logic for query algorithms.

Notes
-----
  - should be framework agnostic, i.e., rely only on mumpy and scipy, not torch etc.
"""

from typing import Protocol

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
    def __call__(self, n_query: int, classwise_probs: np.ndarray) -> np.ndarray:
        scores = 1 - np.max(classwise_probs, axis=1)
        return np.flip(scores.argsort())[:n_query]


class MarginQuerier:
    def __call__(self, n_query: int, classwise_probs: np.ndarray) -> np.ndarray:
        if classwise_probs.shape[1] == 1:
            return np.zeros(shape=(classwise_probs.shape[0],))
        part = np.partition(-1 * classwise_probs, 1, axis=1)
        scores = -part[:, 0] + part[:, 1]
        return np.flip(scores.argsort())[:n_query]


class EntropyQuerier:
    def __call__(self, n_query: int, classwise_probs: np.ndarray) -> np.ndarray:
        scores = np.transpose(entropy(np.transpose(classwise_probs)))
        return np.flip(scores.argsort())[:n_query]
