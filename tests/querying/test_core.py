import sys

sys.path.insert(0, ".")

import numpy as np

from src.querying.core import *


def test_random():

    q = RandomQuerier()
    b = q(10, np.arange(100))
    print(b)


def test_uncertainty():

    q = UncertaintyQuerier("U")
    p = np.random.rand(100, 10)
    b = q(10, p)
    print(p)
    print(b)


test_uncertainty()
test_random()
