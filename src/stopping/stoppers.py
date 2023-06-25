"""Stopping algorithms for active learning.
"""

from collections import deque
from typing import Literal, Protocol

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score


__all__ = [
    "Stopper",
    "ContinuousStopper",
    "StabilizingPredictionsStopper",
    "ChangingCondifenceStopper",
    "ClassificationChangeStopper",
]


class Stopper(Protocol):
    def __call__(self, *args, **kwds) -> bool:
        ...


class ContinuousStopper:
    def __call__(self) -> bool:
        return False


class StabilizingPredictionsStopper:
    def __init__(self, windows: int, threshold: float) -> None:
        self.windows = windows
        self.threshold = threshold
        self.agreement_scores = deque(maxlen=windows)
        self.prv_stop_set_preds = None
        self.stop_set_indices = None

    def __call__(self, stop_set_preds: np.ndarray) -> bool:
        if self.prv_stop_set_preds is None:
            agreement = np.NaN
        elif np.array_equal(self.prv_stop_set_preds, stop_set_preds):
            agreement = 1.0
        else:
            agreement = cohen_kappa_score(self.prv_stop_set_preds, stop_set_preds)

        self.agreement_scores.append(agreement)
        self.prv_stop_set_preds = stop_set_preds

        if len(self.agreement_scores) < self.windows:
            return False
        if np.mean(self.agreement_scores) > self.threshold:
            return True
        return False


class ChangingCondifenceStopper:
    def __init__(self, windows: int, mode: Literal["D", "N"]) -> None:
        self.windows = windows
        self.mode = mode
        self.conf_scores = deque(maxlen=windows)

    def __call__(self, stop_set_confs: np.ndarray) -> bool:
        c = np.mean(stop_set_confs)
        if len(self.conf_scores) < self.windows:
            self.conf_scores.append(c)
            return False
        prv = self.conf_scores.popleft()
        self.conf_scores.append(c)
        if self.mode == "D" and all(c < prv for c in self.conf_scores):
            return True
        if self.mode == "N" and all(c <= prv for c in self.conf_scores):
            return True
        return False


class ClassificationChangeStopper:
    def __init__(self, threshold: float, increment: float) -> None:
        self.threshold = threshold
        self.threshold_increment = increment
        self.prev_unlabeled_preds = None

    def __call__(self, unlabeled_preds: np.ndarray, batch_preds: np.ndarray, batch_labels: np.ndarray) -> bool:
        if self.prev_unlabeled_preds is None:
            return False

        if accuracy_score(batch_preds, batch_labels) > self.threshold:
            if np.array_equal(self.prev_unlabeled_preds, unlabeled_preds):
                return True
            self.threshold += self.threshold_increment

        self.prev_unlabeled_preds = unlabeled_preds
        return False
