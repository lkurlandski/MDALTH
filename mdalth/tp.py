"""
Custom type aliases.
"""

from typing import TypeAlias

import numpy as np


# Used to indcate that floating point numbers on (0, 1) are interpreted as
# proportions, while those on [1, float('inf')] are interpreted as integers.
ProportionOrInteger = TypeAlias("ProportionOrInteger", float)

# (B, C) array of class-wise predictions
ClassWisePreds = TypeAlias("ClassWisePreds", np.ndarray)

# (B,) array of predictions
Preds = TypeAlias("Preds", np.ndarray)

# (B, C) array of class-wise prediction probabilities
ClassWiseProbs = TypeAlias("ClassWiseProbs", np.ndarray)

# (B,) array of probabilities of the most likely class
Probs = TypeAlias("Probs", np.ndarray)

# (B, C) array of class-wise non-probabalistic predictions
ClassWiseLogits = TypeAlias("ClassWiseLogits", np.ndarray)

# (B,) array of non-probabalistic logits of the most likely class
Logits = TypeAlias("Logits", np.ndarray)
