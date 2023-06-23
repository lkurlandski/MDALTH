import sys

sys.path.insert(0, ".")

import numpy as np

from src.stopping.core import *


def test_stabilizing_predictions():

    stop_set_preds = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    s = StabilizingPredictionsStopper(3, 0.9)
    for i in range(10):
        if s(stop_set_preds):
            break
    assert i == 3


def test_changing_confidence():

    stop_set_confs = np.array(
        [
            [0.1, 0.2],
            [0.1, 0.2],
            [0.1, 0.2],
            [0.1, 0.2],
            [0.1, 0.2],
            [0.1, 0.2],
        ]
    )
    s = ChangingCondifenceStopper(3, "N")
    for i in range(10):
        if s(stop_set_confs[i]):
            break
    assert i == 3


def test_classification_change():

    unlabeled_preds = np.array(
        [
            [1, 1, 0],
            [1, 1, 0],
        ]
    )
    batch_labels = np.array([1, 1])
    batch_preds = np.array([1, 1])
    s = ClassificationChangeStopper(0.9, 0.002)

    for i in range(10):
        if s(unlabeled_preds[1], batch_preds, batch_labels):
            break
    assert i == 1


test_stabilizing_predictions()
test_changing_confidence()
test_classification_change()
