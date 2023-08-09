import sys

sys.path.insert(0, ".")

import numpy as np
from numpy import random
from transformers import Trainer

from mdalt.querying.hf import *
from mdalt.querying.queriers import *

from tests.utils import hf


def get_wrapper_and_trainer(querier: Querier) -> tuple[QuerierWrapper, Trainer]:
    dataset = hf.DATASET["train"]
    querier_wrapper = QuerierWrapper.from_querier(querier, dataset)
    querier_wrapper.label(random.choice(len(dataset), size=42, replace=False))
    datasets = dataset.select(querier_wrapper.labeled).train_test_split()
    trainer = hf.get_trainer(datasets["train"], datasets["test"])
    return querier_wrapper, trainer


def test_RandomQuerierWrapper():
    n_query = 5
    querier_wrapper, trainer = get_wrapper_and_trainer(RandomQuerier())
    result = querier_wrapper(trainer, n_query)
    assert isinstance(result, np.ndarray)
    assert len(result) == n_query


def test_UncertaintyQuerierWrapper():
    n_query = 5
    querier_wrapper, trainer = get_wrapper_and_trainer(UncertaintyQuerier("U"))
    result = querier_wrapper(trainer, n_query)
    assert isinstance(result, np.ndarray)
    assert len(result) == n_query
