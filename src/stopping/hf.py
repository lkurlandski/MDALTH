"""High-level wrappers around stopping algorithms.
"""

from __future__ import annotations
from abc import abstractmethod, ABC
from pathlib import Path
import pickle

from datasets import Dataset
from transformers import PreTrainedModel

from src.stopping.stoppers import (
    Stopper,
    ContinuousStopper,
    StabilizingPredictionsStopper,
    ChangingCondifenceStopper,
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
    def __init__(self, stopper: Stopper, interrupt: bool = False) -> None:
        self.stopper = stopper
        self.interrupt = interrupt

    @abstractmethod
    def __call__(self, model: PreTrainedModel, dataset: Dataset) -> bool:
        ...

    @classmethod
    def from_stopper(cls, stopper: Stopper, interrupt: bool = False) -> StopperWrapper:
        if isinstance(stopper, ContinuousStopper):
            return ContinuousStopperWrapper(stopper, interrupt)
        if isinstance(stopper, StabilizingPredictionsStopper):
            return StabilizingPredictionsStopperWrapper(stopper, interrupt)
        if isinstance(stopper, ChangingCondifenceStopper):
            return ChangingCondifenceStopperWrapper(stopper, interrupt)
        if isinstance(stopper, ClassificationChangeStopper):
            return ClassificationChangeStopperWrapper(stopper, interrupt)
        raise TypeError()


class ContinuousStopperWrapper(StopperWrapper):
    def __call__(self) -> bool:
        return False


class StabilizingPredictionsStopperWrapper(StopperWrapper):
    def __call__(self, model: PreTrainedModel, dataset: Dataset) -> bool:
        ...


class ChangingCondifenceStopperWrapper:
    def __call__(self, model: PreTrainedModel, dataset: Dataset) -> bool:
        ...


class ClassificationChangeStopperWrapper:
    def __call__(self, model: PreTrainedModel, dataset: Dataset) -> bool:
        ...
