"""
"""

from __future__ import annotations
from abc import abstractmethod, ABC
from pathlib import Path
import pickle

from datasets import Dataset
from transformers import PreTrainedModel

from stopping.stoppers import Stopper
from utils import pickler


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

    @classmethod
    def load(cls, path) -> StopperWrapper:
        return pickler(path)

    def save(self, path: Path) -> None:
        pickler(path, self)

    @abstractmethod
    def _get_args(self, model: PreTrainedModel, dataset: Dataset, n: int = 1) -> tuple:
        ...


class NoOpWrapper(StopperWrapper):
    def _get_args(self, *args, **kwds) -> tuple:  # pylint: disable=unused-argument
        return tuple()


class StabizingPredictionsWrapper(StopperWrapper):
    def __init__(self, stopper: Stopper, arg: int, **kwds) -> None:
        super().__init__(stopper, **kwds)
        self.arg = arg

    def _get_args(self, model: PreTrainedModel, dataset: Dataset, n: int = 1) -> tuple:
        return (self.arg,)
