"""
Active learning loop.
"""

from __future__ import annotations
from abc import abstractmethod, ABC
from collections.abc import Callable
from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
from typing import ClassVar, Optional

import numpy as np
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from datasets import Dataset, DatasetDict
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

from src.pool import Pool
from src.querying.queriers import Querier, RandomQuerier
from src.querying.hf import QuerierWrapper, RandomQuerierWrapper
from src.stopping.stoppers import Stopper, ContinuousStopper
from src.stopping.hf import StopperWrapper, ContinuousStopperWrapper
from src.utils import (
    get_highest_path,
    is_directory_empty,
    load_with_pickle,
    save_with_pickle,
)


class ALIOHelper:
    """Manage the various paths for saving and loading.

    Output Structure
    ----------------
    | - {root}
        | - iterations
            | - 0
                | - batch.txt
                | - test_metrics.json
                | - trainer_output.pickle
                | - models
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
            | - querier_wrapper.pickle
            | - stopper_wrapper.pickle
            | - config.pickle
            | - io_helper.pickle
    """

    _meta: ClassVar[str] = "meta/"
    _dataset: ClassVar[str] = "dataset/"
    _querier_wrapper: ClassVar[str] = "querier_wrapper.pickle"
    _stopper_wrapper: ClassVar[str] = "stopper_wrapper.pickle"
    _config: ClassVar[str] = "config.pickle"
    _io_helper: ClassVar[str] = "io_helper.pickle"
    _iterations: ClassVar[str] = "iterations/"
    _models: ClassVar[str] = "models/"
    _batch: ClassVar[str] = "batch.txt"
    _test_metrics: ClassVar[str] = "test_metrics.json"
    _trainer_output: ClassVar[str] = "trainer_output.pickle"

    def __init__(self, root_path: Path = None, overwrite: bool = False) -> None:
        if root_path is None:
            root_path = tempfile.mkdtemp()
        self._root_path = Path(root_path)
        self._overwrite = overwrite
        self._valid = None

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
    def config_path(self) -> Path:
        return self.meta_path / self._config

    @property
    def io_helper_path(self) -> Path:
        return self.meta_path / self._io_helper

    @property
    def querier_wrapper_path(self) -> Path:
        return self.meta_path / self._querier_wrapper

    @property
    def stopper_wrapper_path(self) -> Path:
        return self.meta_path / self._stopper_wrapper

    @property
    def valid(self) -> bool:
        v = not self.root_path.exists() or is_directory_empty(self.root_path) or self.overwrite
        if self._valid is None:
            self._valid = v
        return self._valid

    def models_path(self, iteration: int) -> Path:
        return self.iterations_path / str(iteration) / self._models

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


class ALConfig:
    """Arguments for the generalized active learning loop."""

    def __init__(
        self,
        n_rows: int,
        n_start: int | float = 0.1,
        n_query: int | float = 0.1,
        validation_set_size: int | float = 0.1,
        do_train: bool = True,
        do_test: bool = True,
    ) -> None:
        self.n_rows = n_rows
        self._n_start = n_start
        self._n_query = n_query
        self._validation_set_size = validation_set_size
        self.do_train = do_train
        self.do_test = do_test

    @property
    def n_start(self) -> int:
        if isinstance(self._n_start, float):
            return int(self.n_start * self.n_rows)
        return self._n_start

    @property
    def n_query(self) -> int:
        if isinstance(self._n_query, float):
            return int(self.n_query * self.n_rows)
        return self._n_query

    @property
    def validation_set_size(self) -> int:
        if isinstance(self._validation_set_size, float):
            return int(self.validation_set_size * self.n_rows)
        return self._validation_set_size


class ALTrainerBuilder(ABC):
    """Construct a huggingface Trainer object at every iteration."""

    def __init__(
        self,
        model_init: Callable[[], PreTrainedModel],
        args: Optional[TrainingArguments] = None,
        data_collator: Optional[DataCollator] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], dict]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optimizer, LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
    ) -> None:
        self.model_init = model_init
        self.args = args
        self.data_collator = data_collator
        self.tokenizer = tokenizer
        self.compute_metrics = compute_metrics
        self.callbacks = callbacks
        self.optimizers = optimizers
        self.preprocess_logits_for_metrics = preprocess_logits_for_metrics

    @abstractmethod
    def __call__(
        self,
        iteration: int,
        *,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        model: Optional[PreTrainedModel] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
    ) -> Trainer:
        if model is not None and model_init is not None:
            raise ValueError("Cannot provide both `model` and `model_init`.")
        return Trainer(
            model,
            self.args,
            self.data_collator,
            train_dataset,
            eval_dataset,
            self.tokenizer,
            self.model_init if model is None else None,
            self.compute_metrics,
            self.callbacks,
            self.optimizers,
            self.preprocess_logits_for_metrics,
        )


class ALLearner(ABC):
    """Generalized active learning loop."""

    def __init__(
        self,
        dataset: DatasetDict,
        querier_wrapper: Optional[QuerierWrapper],
        stopper_wrapper: Optional[StopperWrapper],
        config: ALConfig,
        io_helper: ALIOHelper,
        trainer_init: ALTrainerBuilder,
        _iteration: int = 0,
    ) -> None:
        self._dataset = dataset
        self._querier_wrapper = querier_wrapper
        self._stopper_wrapper = stopper_wrapper
        self.config = config
        self.io_helper = io_helper
        self.trainer_init = trainer_init
        self.iteration = _iteration
        self.batch = np.array([])
        if not self.io_helper.valid:
            raise FileExistsError(self.io_helper.root_path)

    def __call__(self) -> None:
        if self.iteration != 0:
            raise RuntimeError()
        self.batch = RandomQuerierWrapper(RandomQuerier(), self.pool)(self.config.n_start)
        self._train_one_iteration()

    def __iter__(self) -> ALLearner:
        return self

    def __len__(self) -> int:
        q, r = divmod(self.n_rows - self.config.n_start, self.config.n_query)
        if r == 0:
            return q
        return q + 1

    def __next__(self) -> None:
        if self.iteration == 0:
            raise RuntimeError()
        if self.iteration > len(self):
            raise StopIteration()

        self.stopper_wrapper()
        self.iteration += 1
        self.batch = self.querier_wrapper()
        self._train_one_iteration()

    def __str__(self) -> str:
        raise NotImplementedError()

    @property
    def dataset(self) -> DatasetDict:
        return self._dataset

    @dataset.setter  # TODO: add validation
    def dataset(self, dataset: DatasetDict) -> None:
        self._dataset = dataset

    @property
    def querier_wrapper(self) -> QuerierWrapper:
        return self._querier_wrapper

    @querier_wrapper.setter  # TODO: add validation
    def querier_wrapper(self, querier_wrapper: Optional[QuerierWrapper]) -> None:
        if querier_wrapper is None:
            self._querier_wrapper = RandomQuerierWrapper(RandomQuerier(), Pool(self.n_rows))
        else:
            self._querier_wrapper = querier_wrapper

    @property
    def stopper_wrapper(self) -> StopperWrapper:
        return self._stopper_wrapper

    @stopper_wrapper.setter  # TODO: add validation
    def stopper_wrapper(self, stopper_wrapper: Optional[StopperWrapper]) -> None:
        if stopper_wrapper is None:
            self._stopper_wrapper = ContinuousStopperWrapper(ContinuousStopper())
        else:
            self._stopper_wrapper = stopper_wrapper

    @property
    def n_rows(self) -> int:
        return len(self.pool)

    @property
    def pool(self) -> Pool:
        return self.querier_wrapper.pool

    @property
    def labeled(self) -> np.ndarray:
        return self.querier_wrapper.labeled

    @property
    def unlabeled(self) -> np.ndarray:
        return self.querier_wrapper.unlabeled

    @property
    def validation_set_size(self) -> int:
        return self.config.validation_set_size

    @classmethod
    def load_from_disk(cls, output_dir: Path, iteration: Optional[int] = None, overwrite: bool = False) -> ALLearner:
        io_helper = ALIOHelper(output_dir, overwrite)
        iteration = get_highest_path(io_helper.iterations_path.glob("*")) if iteration is None else iteration
        dataset = Dataset.load_from_disk(io_helper.dataset_path)
        querier_wrapper = load_with_pickle(io_helper.querier_wrapper_path)
        stopper_wrapper = load_with_pickle(io_helper.stopper_wrapper_path)
        config = load_with_pickle(io_helper.config_path)
        trainer_init = load_with_pickle(io_helper.io_helper_path)
        return cls(dataset, querier_wrapper, stopper_wrapper, config, io_helper, trainer_init, iteration)

    def save_to_disk(self, train_output: Optional[TrainOutput], test_metrics: Optional[dict[str, float]]) -> None:
        if self.iteration == 0:
            self.dataset.save_to_disk(self.io_helper.dataset_path)
            save_with_pickle(self.io_helper.querier_wrapper_path, self.querier_wrapper)
            save_with_pickle(self.io_helper.stopper_wrapper_path, self.stopper_wrapper)
            save_with_pickle(self.io_helper.config_path, self.config)

        np.savetxt(self.io_helper.batch_path(self.iteration), self.batch)
        if train_output:
            save_with_pickle(self.io_helper.trainer_output_path(self.iteration), train_output)
        if test_metrics:
            with open(self.io_helper.test_metrics_path(self.iteration), "w") as handle:
                json.dump(test_metrics, handle)

    def _train_one_iteration(self) -> TrainOutput:
        dataset = self.dataset["train"].select(self.labeled).train_test_split(self.config.validation_set_size)
        trainer = self.trainer_init(dataset["train"], dataset["test"])
        trainer.training_args.output_dir = self.io_helper.models_path(self.iteration)
        train_output = trainer.train() if self.config.do_train else None
        test_metrics = trainer.evaluate(self.dataset["test"]) if self.config.do_test else None
        self.save_to_disk(train_output, test_metrics)
