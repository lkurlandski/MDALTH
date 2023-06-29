"""High-level wrappers around stopping algorithms.
"""

from __future__ import annotations
from abc import abstractmethod, ABC
from pathlib import Path
import pickle

from datasets import Dataset
import numpy as np
from numpy import random
from transformers import Trainer

from src.stopping.stoppers import (
    Stopper,
    ContinuousStopper,
    StabilizingPredictionsStopper,
    ChangingConfidenceStopper,
    ClassificationChangeStopper,
)

__all__ = [
    "StopperWrapper",
    "ContinuousStopperWrapper",
    "StabilizingPredictionsStopperWrapper",
    "ChangingCondifenceStopperWrapper",
    "ClassificationChangeStopperWrapper",
]


class StopperWrapper(ABC):
    def __init__(self, stopper: Stopper, interrupt: bool) -> None:
        self.stopper = stopper
        self.interrupt = interrupt

    @abstractmethod
    def __call__(self, trainer: Trainer, dataset: Dataset) -> bool:
        ...

    @classmethod
    def from_stopper(cls, stopper: Stopper, interrupt: bool, **kwds) -> StopperWrapper:
        if isinstance(stopper, ContinuousStopper):
            return ContinuousStopperWrapper(stopper, interrupt, **kwds)
        if isinstance(stopper, StabilizingPredictionsStopper):
            return StabilizingPredictionsStopperWrapper(stopper, interrupt, **kwds)
        if isinstance(stopper, ChangingConfidenceStopper):
            return ChangingCondifenceStopperWrapper(stopper, interrupt, **kwds)
        if isinstance(stopper, ClassificationChangeStopper):
            return ClassificationChangeStopperWrapper(stopper, interrupt, **kwds)
        raise TypeError()


class ContinuousStopperWrapper(StopperWrapper):
    def __call__(self) -> bool:
        return False


class StabilizingPredictionsStopperWrapper(StopperWrapper):
    def __init__(
        self,
        stopper: StabilizingPredictionsStopper,
        interrupt: bool,
        stop_set_idx: np.ndarray,
    ) -> None:
        super().__init__(stopper, interrupt)
        self.stop_set_idx = stop_set_idx

    def __call__(self, trainer: Trainer) -> bool:
        stop_set_preds = trainer
        return self.stopper()


class ChangingCondifenceStopperWrapper:
    def __call__(self, model: PreTrainedModel, dataset: Dataset) -> bool:
        ...


class ClassificationChangeStopperWrapper:
    def __call__(self, model: PreTrainedModel, dataset: Dataset) -> bool:
        ...
