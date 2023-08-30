"""
huggingface wrappers around the core querying algorithms.
"""

from __future__ import annotations
from abc import abstractmethod, ABC
from typing import Optional

import numpy as np
from transformers import PreTrainedModel
from torch import tensor
from torch.nn.functional import softmax

from mdalth.helpers import Pool, TrainerFactory
from mdalth.querying import (
    Querier,
    RandomQuerier,
    UncertaintyQuerier,
    MarginQuerier,
    EntropyQuerier,
)


class QuerierWrapper(ABC):
    """Interface for querying algorithms to the Learner."""

    def __init__(self, querier: Querier, pool: Pool, *_) -> None:
        self.querier = querier
        self.pool = pool

    @abstractmethod
    def __call__(self, n_query: int, model: PreTrainedModel) -> np.ndarray:
        assert n_query <= len(self.pool.unlabeled_idx), "Requested to query more than available."


class RandomQuerierWrapper(QuerierWrapper):
    """Interface between the random querier and the Learner."""

    def __call__(self, n_query: int, *_) -> np.ndarray:
        super().__call__(n_query, None)
        return self.querier(n_query, self.pool.unlabeled_idx)


class ClasswiseProbsQuerierWrapper(QuerierWrapper):
    """Interface for querying algorithms that require classwise probabilities."""

    def __init__(
        self,
        querier: UncertaintyQuerier | MarginQuerier | EntropyQuerier,
        pool: Pool,
        trainer_fact: TrainerFactory,
        probabalistic: bool = True,
    ) -> None:
        super().__init__(querier, pool)
        self.pool = pool
        self.trainer_fact = trainer_fact
        self.probabalistic = probabalistic

    def __call__(self, n_query: int, model: PreTrainedModel) -> np.ndarray:
        super().__call__(n_query, model)
        trainer = self.trainer_fact(model=model)
        prediction_output = trainer.predict(self.pool.dataset.select(self.pool.unlabeled_idx))
        classwise_probs = prediction_output.predictions
        if self.probabalistic:
            classwise_probs = softmax(tensor(classwise_probs), dim=1).numpy()
        idx_map = {i: self.pool.unlabeled_idx[i] for i in range(len(self.pool.unlabeled_idx))}
        idx_ = self.querier(n_query, classwise_probs)
        idx = np.array([idx_map[i] for i in idx_])
        return idx


def querier_wrapper_factory(
    querier: Querier,
    pool: Optional[Pool] = None,
    trainer_fact: Optional[TrainerFactory] = None,
) -> QuerierWrapper:
    """Get a wrapper for a given querier."""

    if isinstance(querier, RandomQuerier):
        return RandomQuerierWrapper(querier, pool)

    if isinstance(querier, (UncertaintyQuerier, MarginQuerier, EntropyQuerier)):
        return ClasswiseProbsQuerierWrapper(querier, pool, trainer_fact)

    raise TypeError(f"Querier of type {type(querier)} not supported.")
