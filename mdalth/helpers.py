"""
Helper classes.
"""

from __future__ import annotations
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from pprint import pformat
import tempfile
from typing import ClassVar, Optional

from datasets import Dataset
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

from mdalth.utils import is_directory_empty


class IOHelper:
    """Manage the various paths for saving and loading.

    Output Structure
    ----------------
    | - {root}
        | - iterations
            | - 0
                | - batch.txt
                | - log_history.json
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
    _log_history: ClassVar[str] = "log_history.json"
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

    def log_history_path(self, iteration: int) -> Path:
        return self.iterations_path / str(iteration) / self._log_history

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


class PoolIdx:
    """Manages the indices of the active learning pools."""

    def __init__(self, n: int) -> None:
        self.idx: ma.MaskedArray = ma.array(np.arange(n, dtype=int), mask=True)

    def __len__(self) -> int:
        return len(self.idx)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(\n{pformat(vars(self))}\n)"

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
    def from_pools(cls, labeled_idx: np.ndarray, unlabeled_idx: np.ndarray) -> PoolIdx:
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


class Pool:
    """Provide PoolIdx support around a Dataset object."""

    def __init__(self, dataset: Dataset) -> None:
        self.pool = PoolIdx(len(dataset))
        self.dataset = dataset

    @property
    def idx(self) -> ma.MaskedArray:
        return self.pool.idx

    @property
    def labeled_idx(self) -> np.ndarray:
        return self.pool.labeled_idx

    @property
    def unlabeled_idx(self) -> np.ndarray:
        return self.pool.unlabeled_idx

    @property
    def labeled(self) -> Dataset:
        return self.dataset.select(self.labeled_idx)

    @property
    def unlabeled(self) -> Dataset:
        return self.dataset.select(self.unlabeled_idx)

    @classmethod
    def from_ma(cls, dataset: Dataset, idx: ma.MaskedArray) -> Pool:
        pool = cls(dataset)
        pool.idx.mask = idx.mask
        return pool

    @classmethod
    def from_pools(
        cls,
        dataset: Dataset,
        labeled_idx: Optional[np.ndarray] = None,
        unlabeled_idx: Optional[np.ndarray] = None,
    ) -> Pool:
        if labeled_idx is None and unlabeled_idx is None:
            raise ValueError("At least one of labeled_idx and unlabeled_idx must be provided.")
        pool = cls(dataset)
        if labeled_idx is not None:
            pool.label(labeled_idx)
        if unlabeled_idx is not None:
            pool.unlabel(unlabeled_idx)
        return pool

    def label(self, idx: np.ndarray) -> None:
        self.pool.label(idx)

    def unlabel(self, idx: np.ndarray) -> None:
        self.pool.unlabel(idx)


class TrainerFactory:
    """Create Trainer objects with a consistent configuration."""

    def __init__(
        self,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        args: Optional[TrainingArguments] = None,
        data_collator: Optional[DataCollator] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], dict]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optimizer, LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
    ) -> None:
        self.model_init = model_init
        args = TrainingArguments() if args is None else args
        self.args_fact = TrainingArgumentsFactory(args)
        self.data_collator = data_collator
        self.tokenizer = tokenizer
        self.compute_metrics = compute_metrics
        self.callbacks = callbacks
        self.optimizers = optimizers
        self.preprocess_logits_for_metrics = preprocess_logits_for_metrics

    def __call__(
        self,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        model: Optional[PreTrainedModel] = None,
    ) -> Trainer:
        model_init = None if model else self.model_init
        return Trainer(
            model,
            self.args_fact(train_dataset, eval_dataset, model),
            deepcopy(self.data_collator),
            train_dataset,
            eval_dataset,
            deepcopy(self.tokenizer),
            model_init,
            deepcopy(self.compute_metrics),
            deepcopy(self.callbacks),
            deepcopy(self.optimizers),
            deepcopy(self.preprocess_logits_for_metrics),
        )


# FIXME: remove, as this is not nececcary in later versions of transformers. 
class TrainingArgumentsFactory:
    """Create TrainingArguments objects with a consistent configuration.

    Notes
    -----
        - The motivation for this is that we need to ensure there are enough samples
        for the train and evaluation batches at each iteration, else huggingface's
        Trainer will raise an error.
        - However, this may not work as expected because
        TrainingArguments.__init__ will not be called when
        TrainingArgumentsFactory.__call__ is called.
        - Therefore, we may have to create a custom TrainingArguments class that
        inherits from TrainingArguments or pass in keyword arguments for the
        TrainingArguments constructor instead of a TrainingArguments object,
        then create a new TrainingArguments within __call__.
    """

    def __init__(self, args: TrainingArguments) -> None:
        self.args = args
        assert not self.args.resume_from_checkpoint, "resume_from_checkpoint not supported."
        assert self.args.load_best_model_at_end, "load_best_model_at_end must be True."

    def __call__(
        self,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        model: Optional[PreTrainedModel] = None,  # pylint: disable=unused-argument
    ) -> TrainingArguments:
        args = deepcopy(self.args)
        if train_dataset is not None:
            object.__setattr__(  # FIXME: replace with dataclasses.replace
                args,
                "per_device_train_batch_size",
                min(self.args.per_device_train_batch_size, len(train_dataset)),
            )
        if eval_dataset is not None:
            object.__setattr__(  # FIXME: replace with dataclasses.replace
                args,
                "per_device_eval_batch_size",
                min(self.args.per_device_eval_batch_size, len(eval_dataset)),
            )
        return args
