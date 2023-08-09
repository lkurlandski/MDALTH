import sys

sys.path.insert(0, ".")

import numpy as np

from mdalt.querying.queriers import *


def test_RandomQuerier():
    querier = RandomQuerier()
    n_query = 5
    unlabeled_idx = np.arange(10)
    result = querier(n_query, unlabeled_idx)
    assert isinstance(result, np.ndarray)
    assert len(result) == n_query
    assert np.all(np.isin(result, unlabeled_idx))


def test_UncertaintyQuerier():
    mode = "E"
    querier = UncertaintyQuerier(mode)
    n_query = 5
    probs = np.random.rand(10, 3)
    result = querier(n_query, probs)
    assert isinstance(result, np.ndarray)
    assert len(result) == n_query


def test_UncertaintyQuerier_uncertainty():
    probs = np.random.rand(10, 3)
    result = UncertaintyQuerier.uncertainty(probs)
    assert isinstance(result, np.ndarray)
    assert len(result) == 10


def test_UncertaintyQuerier_margin():
    probs = np.random.rand(10, 2)
    result = UncertaintyQuerier.margin(probs)
    assert isinstance(result, np.ndarray)
    assert len(result) == 10


def test_UncertaintyQuerier_entropy():
    probs = np.random.rand(10, 3)
    result = UncertaintyQuerier.entropy(probs)
    assert isinstance(result, np.ndarray)
    assert len(result) == 10
