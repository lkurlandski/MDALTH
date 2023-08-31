"""
Custom type aliases.

TODO
----
    - use the newest type hinting features from numpy to annotate dtypes
"""

import sys

if sys.version_info.major >= 3 and sys.version_info.minor >= 11:
    from typing import TypeAlias
else:
    from typing import Any as TypeAlias

import numpy as np


# Used to indcate that floating point numbers on (0, 1) are interpreted as
# proportions, while those on [1, float('inf')] are interpreted as integers.
ProportionOrInteger: TypeAlias = float

# (B, C) array of class-wise predictions
ClassWisePreds: TypeAlias = np.ndarray

# (B,) array of predictions
Preds: TypeAlias = np.ndarray

# (B, C) array of class-wise prediction probabilities
ClassWiseProbs: TypeAlias = np.ndarray

# (B,) array of probabilities of the most likely class
Probs: TypeAlias = np.ndarray

# (B, C) array of class-wise non-probabalistic predictions
ClassWiseLogits: TypeAlias = np.ndarray

# (B,) array of non-probabalistic logits of the most likely class
Logits: TypeAlias = np.ndarray
