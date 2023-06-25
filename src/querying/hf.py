"""High-level wrappers around query algorithms.
"""

from __future__ import annotations
from abc import abstractmethod, ABC
from pathlib import Path
from typing import Any, Optional, Protocol

from datasets import Dataset
import numpy as np

from transformers import (
    Trainer,
)

from src.querying.queriers import (
    Querier,
    RandomQuerier,
    UncertaintyQuerier,
)
from src.pool import Pool


__all__ = [
    "QuerierWrapper",
    "RandomQuerierWrapper",
    "UncertaintyQuerierWrapper",
]


class QuerierWrapper(ABC):
    def __init__(self, querier: Querier, dataset: Dataset) -> None:
        self.querier = querier
        self.dataset = dataset
        self.pool = Pool(len(dataset))

    @abstractmethod
    def __call__(self, trainer: Trainer, n_query: int) -> np.ndarray:
        ...

    @classmethod
    def from_querier(cls, querier: Querier, dataset: Dataset) -> QuerierWrapper:
        if isinstance(querier, RandomQuerier):
            return RandomQuerierWrapper(querier, dataset)
        if isinstance(querier, UncertaintyQuerier):
            return UncertaintyQuerierWrapper(querier, dataset)
        raise TypeError()

    @property
    def labeled(self) -> Optional[np.ndarray]:
        return self.pool.labeled

    @property
    def unlabeled(self) -> Optional[np.ndarray]:
        return self.pool.unlabeled

    def label(self, idx: np.ndarray) -> None:
        return self.pool.label(idx)

    def unlabel(self, idx: np.ndarray) -> None:
        return self.pool.unlabel(idx)


class RandomQuerierWrapper(QuerierWrapper):
    def __call__(self, trainer: Trainer, n_query: int) -> np.ndarray:  # pylint: disable=unused-argument
        return self.querier(n_query, self.pool.unlabeled)


class UncertaintyQuerierWrapper(QuerierWrapper):
    def __call__(self, trainer: Trainer, n_query: int) -> np.ndarray:
        dataset = self.dataset.select(self.pool.unlabeled)
        prediction_output = trainer.predict(dataset)
        classwise_probabilities = prediction_output.predictions
        return self.querier(n_query, classwise_probabilities)


# class BALDQuerierWrapper:

#     def get_args(self, trainer: Trainer, n: int, *args, **kwds) -> tuple:
#         n_classes = self.dataset.ClassLabel.num_classes
#         dataset = self.dataset.select(self.querier.pool.unlabeled)
#         loader = DataLoader(dataset, shuffle=False, batch_size=1)
#         probs = torch.zeros([self.n_drop, len(dataset), n_classes])

#         for i in range(self.querier.n_drop):
#             for j, (x, y) in enumerate(loader):
#                 x = x.to(self.device)
#                 y = y.to(self.device)
#                 out = trainer.model(x)
#                 idx = list(range(j * self.batch_size, (j + 1) * self.batch_size))  # T_ODo may have to be scaler
#                 probs[i][idx] += softmax(out, dim=1).cpu().data

#         return self.sampler(n, probs.numpy(force=True))
