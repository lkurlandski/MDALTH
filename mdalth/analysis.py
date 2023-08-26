"""
Vanilla analyze of experiments.
"""

import json

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from mdalth.helpers import IOHelper


class Analyzer:
    """Simplistic analysis of active learning curves.

    Notes
    -----
        - This is not meant to be beautiful, merely functional and reliable.
    """

    def __init__(self, io_helper: IOHelper) -> None:
        self.io_helper = io_helper

    def __call__(self, keys: list[str] = ("eval_loss", "eval_accuracy")) -> tuple[Figure, Axes]:
        n_iterations = sum(1 for _ in self.io_helper.iterations_path.iterdir())
        test_metrics = []
        for i in range(n_iterations):
            with open(self.io_helper.test_metrics_path(i), "r") as fp:
                d = json.load(fp)
            if keys:
                d = {k: v for k, v in d.items() if k in keys}
            test_metrics.append(d)

        fig, ax = plt.subplots()
        x = list(range(n_iterations))
        for k in test_metrics[0].keys():
            ax.plot(x, [d[k] for d in test_metrics], label=k)
        ax.legend()
        return fig, ax
