"""
Helper classes.
"""

from __future__ import annotations
from abc import abstractmethod, ABC
from collections.abc import Callable, Collection
from dataclasses import dataclass
import json
from pathlib import Path
import pickle
from pprint import pformat, pprint
import tempfile
from typing import Any, ClassVar, Optional

from datasets import Dataset, DatasetDict
import numpy as np
from numpy import ma
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers import (
    DataCollator,
    EvalPrediction,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from transformers.trainer_utils import TrainOutput

from mdalt.utils import is_directory_empty


class IOHelper:
    """Manage the various paths for saving and loading.

    Output Structure
    ----------------
    | - {root}
        | - iterations
            | - 0
                | - batch.txt
                | - test_metrics.json
                | - trainer_output.pickle
                | - model
                    ...
                | - checkpoints
                    | - checkpoint-1
                    ...
                    | - checkpoint-{n_epochs}
                    ...
            ...
            | - {n_iterations}
                ...
        | - meta
            | - dataset/
                ...
            | - querier.pickle
            | - stopper.pickle
            | - config.pickle
            | - io_helper.pickle
    """

    _meta: ClassVar[str] = "meta/"
    _dataset: ClassVar[str] = "dataset/"
    _tr_dataset: ClassVar[str] = "tr/"
    _ts_dataset: ClassVar[str] = "ts/"
    _querier: ClassVar[str] = "querier.pickle"
    _stopper: ClassVar[str] = "stopper.pickle"
    _config: ClassVar[str] = "config.pickle"
    _io_helper: ClassVar[str] = "io_helper.pickle"
    _iterations: ClassVar[str] = "iterations/"
    _model: ClassVar[str] = "model/"
    _checkpoints: ClassVar[str] = "checkpoints/"
    _batch: ClassVar[str] = "batch.txt"
    _test_metrics: ClassVar[str] = "test_metrics.json"
    _trainer_output: ClassVar[str] = "trainer_output.pickle"

    def __init__(self, root_path: Path = None, overwrite: bool = False) -> None:
        if root_path is None:
            root_path = tempfile.mkdtemp()
        self._root_path = Path(root_path)
        self._overwrite = overwrite
        self._valid = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(\n{pformat(vars(self))}\n)"

    @property
    def root_path(self) -> Path:
        return self._root_path

    @root_path.setter
    def root_path(self, root_path: Path) -> None:
        self._root_path = root_path

    @property
    def overwrite(self) -> bool:
        return self._overwrite

    @property
    def meta_path(self) -> Path:
        return self.root_path / self._meta

    @property
    def iterations_path(self) -> Path:
        return self.root_path / self._iterations

    @property
    def dataset_path(self) -> Path:
        return self.meta_path / self._dataset

    @property
    def tr_dataset_path(self) -> Path:
        return self.dataset_path / self._tr_dataset

    @property
    def ts_dataset_path(self) -> Path:
        return self.dataset_path / self._ts_dataset

    @property
    def config_path(self) -> Path:
        return self.meta_path / self._config

    @property
    def io_helper_path(self) -> Path:
        return self.meta_path / self._io_helper

    @property
    def querier_path(self) -> Path:
        return self.meta_path / self._querier

    @property
    def stopper_path(self) -> Path:
        return self.meta_path / self._stopper

    @property
    def valid(self) -> bool:
        v = not self.root_path.exists() or is_directory_empty(self.root_path) or self.overwrite
        if self._valid is None:
            self._valid = v
        return self._valid

    def checkpoints_path(self, iteration: int) -> Path:
        return self.iterations_path / str(iteration) / self._checkpoints

    def model_path(self, iteration: int) -> Path:
        return self.iterations_path / str(iteration) / self._model

    def batch_path(self, iteration: int) -> Path:
        return self.iterations_path / str(iteration) / self._batch

    def test_metrics_path(self, iteration: int) -> Path:
        return self.iterations_path / str(iteration) / self._test_metrics

    def trainer_output_path(self, iteration: int) -> Path:
        return self.iterations_path / str(iteration) / self._trainer_output

    def mkdir(self, *, parents: bool = False, exist_ok: bool = False) -> None:
        self.root_path.mkdir(parents=parents, exist_ok=exist_ok)
        self.iterations_path.mkdir(exist_ok=exist_ok)
        self.meta_path.mkdir(exist_ok=exist_ok)
        self.dataset_path.mkdir(exist_ok=exist_ok)
        self.tr_dataset_path.mkdir(exist_ok=exist_ok)
        self.ts_dataset_path.mkdir(exist_ok=exist_ok)

    def exists(
        self,
        *,
        root: bool = True,
        iterations: bool = True,
        meta: bool = True,
        dataset: bool = True,
        tr_dataset: bool = True,
        ts_dataset: bool = True,
    ) -> bool:
        return (
            (not root or self.root_path.exists())
            and (not iterations or self.iterations_path.exists())
            and (not meta or self.meta_path.exists())
            and (not dataset or self.dataset_path.exists())
            and (not tr_dataset or self.tr_dataset_path.exists())
            and (not ts_dataset or self.ts_dataset_path.exists())
        )


class BasePool:
    def __init__(self, n: int) -> None:
        self._idx = ma.array(np.arange(n, dtype=int), mask=True)

    def __len__(self) -> int:
        return len(self.idx)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(\n{pformat(vars(self))}\n)"

    @property
    def idx(self) -> ma.MaskedArray:
        return self._idx

    @idx.setter
    def idx(self, idx: ma.MaskedArray) -> None:
        self._idx = idx

    @property
    def labeled_idx(self) -> np.ndarray:
        return self.idx.compressed()

    @property
    def unlabeled_idx(self) -> np.ndarray:
        return ma.getdata(self.idx[self.idx.mask])

    @classmethod
    def from_ma(cls, idx: ma.MaskedArray) -> Pool:
        pool = cls(len(idx))
        pool.idx.mask = idx.mask
        return pool

    @classmethod
    def from_pools(cls, labeled_idx: np.ndarray, unlabeled_idx: np.ndarray) -> BasePool:
        pool = cls(len(labeled_idx) + len(unlabeled_idx))
        pool.idx.mask = unlabeled_idx
        return pool

    def label(self, idx: np.ndarray) -> None:
        if not np.issubdtype(idx.dtype, np.integer):
            raise TypeError(f"Indexing with non-int dtype: {idx.dtype}." "")
        self.idx.mask[idx] = ma.nomask

    def unlabel(self, idx: np.ndarray) -> None:
        if not np.issubdtype(idx.dtype, np.integer):
            raise TypeError(f"Indexing with non-int dtype: {idx.dtype}." "")
        self.idx.mask[idx] = True


class Pool(BasePool):
    def __init__(self, dataset: Dataset) -> None:
        super().__init__(len(dataset))
        self._dataset = dataset

    @property
    def dataset(self) -> Dataset:
        return self._dataset

    @dataset.setter
    def dataset(self, dataset: Dataset) -> None:
        self._dataset = dataset

    @property
    def labeled(self) -> Dataset:
        return self.dataset.select(super().labeled_idx)

    @property
    def unlabeled(self) -> Dataset:
        return self.dataset.select(super().unlabeled_idx)

    @classmethod
    def from_ma(cls, idx: ma.MaskedArray, dataset: Dataset) -> Pool:
        pool = cls(dataset)
        pool.idx.mask = idx.mask
        return pool

    @classmethod
    def from_pools(
        cls, labeled_idx: np.ndarray, unlabeled_idx: np.ndarray, dataset: Dataset
    ) -> Pool:
        pool = cls(dataset)
        pool.label(labeled_idx)
        pool.unlabel(unlabeled_idx)
        return pool
