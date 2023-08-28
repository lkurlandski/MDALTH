"""
huggingface wrappers around the core stopping algorithms.
"""

from __future__ import annotations
from abc import abstractmethod, ABC


from mdalth.stopping import (
    Stopper,
    NullStopper,
    StabilizingPredictions,
    ChangingConfidence,
)


class StoppingWrapper:
    """Interface for stopping algorithms to the Learner.
    
    Notes
    -----
        - if in stop mode (`stop == True`), the wrapper will return True
        when the stopping condition is met.
        - if in `dump` mode (`dump == True`), the wrapper will dump all neccessary information
        required to analyze the `stopper` at a later point.
    """

    def __init__(self, stopper: Stopper, stop: bool = False, dump: bool = False, *_) -> None:
        assert dump or stop, "StoppingWrapper must either stop or dump or both."
        self.stopper = stopper
        self.stop = stop
        self.dump = dump

    @abstractmethod
    def __call__(self) -> bool:
        pass


def stopper_wrapper_factory(stopper: Stopper) -> StoppingWrapper:
    """Get a wrapper for a given stopper."""
    raise NotImplementedError()
