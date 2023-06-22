"""
"""

from abc import abstractmethod, ABC
from typing import Literal

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score


class Stopper(ABC):
    def __init__(self) -> None:
        ...

    @abstractmethod
    def __call__(self) -> bool:
        ...


class NoOpStopper(Stopper):
    def __call__(self) -> bool:
        return False


class StabilizingPredictionsStopper(Stopper):
    def __init__(self, windows: int, threshold: float, **kwds) -> None:
        super().__init__(**kwds)
        self.windows = windows
        self.threshold = threshold
        self.agreement_scores = []
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

        if len(self.agreement_scores[1:]) < self.windows:
            return False
        if np.mean(self.agreement_scores[-self.windows :]) > self.threshold:
            return True
        return False


class ChangingCondifenceStopper(Stopper):
    def __init__(self, windows: int, mode: Literal["<", "<="], **kwds) -> None:
        super().__init__(**kwds)
        self.windows = windows
        self.mode = mode
        self.conf_scores = []
        self.stop_set_indices = None

    def __call__(self, stop_set_confs: np.ndarray) -> bool:
        confidence = np.mean(stop_set_confs)
        self.conf_scores.append(confidence)

        if len(self.conf_scores) + 1 < self.windows:
            return False

        if self.mode == "<":
            prv_conf = self.conf_scores[-self.windows - 1]
            for conf in self.conf_scores[-self.windows :]:
                if conf >= prv_conf:
                    return False
            return True

        if self.mode == "<=":
            prv_conf = self.conf_scores[-self.windows - 1]
            for conf in self.conf_scores[-self.windows :]:
                if conf >= prv_conf:
                    return False
                prv_conf = conf
            return True

        raise ValueError(f"{self.mode=} is not supported.")


class ClassificationChangeStopper(Stopper):
    def __init__(self, threshold: float, increment: float, **kwds) -> None:
        super().__init__(**kwds)
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
