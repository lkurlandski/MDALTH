"""
"""

from __future__ import annotations
from abc import abstractmethod, ABC
from pathlib import Path

from datasets import Dataset
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
from transformers import PreTrainedModel

from querying.queriers import Querier
from utils import pickler


__all__ = ["Wrapper", "Random", "Uncertainty", "BALD"]


# TODO: should the QuerierWrapper contain a reference to the dataset?
# TODO: should take a Trainer not a PretrainedModel
class QuerierWrapper(ABC):
    def __init__(self, querier: Querier, *, device: torch.device = None) -> None:
        self._querier = querier
        self._device = torch.device("cpu") if device is None else device

    def __call__(self, model: PreTrainedModel, dataset: Dataset, n: int = 1) -> np.ndarray:
        return self.querier(*self._get_args(model, dataset, n=n))

    @property
    def querier(self) -> Querier:
        return self._querier

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def labeled(self) -> np.ndarray:
        return self.querier.labeled

    @property
    def unlabeled(self) -> np.ndarray:
        return self.querier.labeled

    @classmethod
    def load(cls, path, **kwds) -> QuerierWrapper:
        return pickler(path, **kwds)

    def save(self, path: Path, **kwds) -> None:
        pickler(path, self, **kwds)

    @abstractmethod
    def _get_args(self, model: PreTrainedModel, dataset: Dataset, n: int) -> tuple:
        ...


class ExampleWrapper(QuerierWrapper):
    def __init__(self, querier: Querier, arg: int, **kwds) -> None:
        super().__init__(querier, **kwds)
        self.arg = arg

    def _get_args(self, model: PreTrainedModel, dataset: Dataset, n: int) -> tuple:
        return (42,)


class RandomWrapper(QuerierWrapper):
    def _get_args(self, model: PreTrainedModel, dataset: Dataset, n: int) -> tuple:  # pylint: disable=unused-argument
        return (n,)


class UncertaintyWrapper(QuerierWrapper):
    def _get_args(self, model: PreTrainedModel, dataset: Dataset, n: int) -> tuple:
        probs = model.predict(dataset)
        return n, probs


class BALDWrapper(QuerierWrapper):
    def __init__(self, querier: Querier, batch_size: int, n_drop: int, n_classes: int, **kwds) -> None:
        super().__init__(querier, **kwds)
        self.batch_size = batch_size  # batch_size may have to be 1
        self.n_drop = n_drop
        self.n_classes = n_classes

    def __call__(self, model: PreTrainedModel, unlabeled: Dataset, n: int, *args, **kwds) -> np.ndarray:
        loader = DataLoader(unlabeled, shuffle=False, batch_size=self.batch_size)
        probs = torch.zeros([self.n_drop, len(unlabeled), self.n_classes])
        for i in range(self.n_drop):
            for j, (x, y) in enumerate(loader):
                x = x.to(self.device)
                y = y.to(self.device)
                out = model(x)
                idx = list(range(j * self.batch_size, (j + 1) * self.batch_size))  # TODO: may have to be scaler
                probs[i][idx] += softmax(out, dim=1).cpu().data

        return self.sampler(n, probs.numpy(force=True))
