"""High-level wrappers around query algorithms.
"""

from __future__ import annotations
from abc import abstractmethod, ABC
from pathlib import Path
from typing import Any, Optional, Protocol

from datasets import Dataset
import numpy as np

# import torch
# from torch.utils.data import DataLoader
# from torch.nn.functional import softmax
from transformers import (
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from src.querying.core import (
    Querier,
    RandomQuerier,
    UncertaintyQuerier,
)
from src.pool import Pool


__all__ = [
    "QuerierWrapper",
    "RandomQuerier",
    "UncertaintyQuerier",
]


class QuerierWrapper(Protocol):
    def __init__(self, querier: Querier, pool: Pool, *args, **kwds) -> None:
        ...

    def __call__(self, *args, **kwds) -> np.ndarray:
        ...


class RandomQuerierWrapper:
    def __init__(self, querier: RandomQuerier, pool: Pool) -> None:
        self.querier = querier
        self.pool = pool

    def __call__(self, n: int) -> np.ndarray:
        return self.querier(n, self.pool.unlabeled)


class UncertaintyQuerierWrapper:
    def __init__(
        self,
        querier: UncertaintyQuerier,
        pool: Pool,
        training_args: Optional[TrainingArguments] = None,
        data_collator: Optional[DataCollator] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ) -> None:
        self.querier = querier
        self.pool = pool
        self.training_args = training_args
        self.data_collator = data_collator
        self.tokenizer = tokenizer

    def __call__(self, dataset: Dataset, n: int, model: PreTrainedModel) -> np.ndarray:
        trainer = self.construct_trainer(model)
        probas = trainer.predict(dataset.select[self.pool.unlabeled])
        return self.querier(probas, n)

    def construct_trainer(self, model) -> Trainer:
        return Trainer(
            model=model,
            args=self.training_args,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
        )


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
