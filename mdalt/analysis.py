"""
Vanilla analyze of experiments.
"""

import json
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np
import pandas as pd

from mdalt.helpers import IOHelper


class Analyzer:
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
