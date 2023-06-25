"""High-level wrappers around stopping algorithms.
"""

from __future__ import annotations
from abc import abstractmethod, ABC
from pathlib import Path
import pickle

from datasets import Dataset
from transformers import PreTrainedModel

from src.stopping.stoppers import Stopper

__all__ = [
    "StopperWrapper",
    "ContinuousStopperWrapper",
    "StabilizingPredictionsStopperWrapper",
    "ChangingCondifenceStopperWrapper",
    "ClassificationChangeStopperWrapper",
]


class StopperWrapper(ABC):
    def __init__(self, stopper: Stopper, *, interrupt: bool = False) -> None:
        self._stopper = stopper
        self._interrupt = interrupt

    def __call__(self, model: PreTrainedModel, dataset: Dataset) -> bool:
        return self.stopper(*self._get_args(model, dataset))

    @property
    def stopper(self) -> Stopper:
        return self._stopper

    @property
    def interrupt(self) -> bool:
        return self._interrupt

    @abstractmethod
    def _get_args(self, model: PreTrainedModel, dataset: Dataset, n: int = 1) -> tuple:
        ...


class ContinuousStopperWrapper(StopperWrapper):
    def _get_args(self, *args, **kwds) -> tuple:  # pylint: disable=unused-argument
        return tuple()


class StabilizingPredictionsStopperWrapper(StopperWrapper):
    def __init__(self, stopper: Stopper, arg: int, **kwds) -> None:
        super().__init__(stopper, **kwds)
        self.arg = arg

    def _get_args(self, model: PreTrainedModel, dataset: Dataset, n: int = 1) -> tuple:
        return (self.arg,)


class ChangingCondifenceStopperWrapper:
    ...


class ClassificationChangeStopperWrapper:
    ...
