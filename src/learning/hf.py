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
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

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

from src.learning.utils import is_directory_empty, SaveLoadPickleMixin
from src.pool import Pool
from src.querying.core import Querier, RandomQuerier
from src.querying.hf import QuerierWrapper, RandomQuerierWrapper
from src.stopping.core import Stopper, ContinuousStopper
from src.stopping.hf import StopperWrapper, ContinuousStopperWrapper


@dataclass
class IOArgs(SaveLoadPickleMixin):

    dataset: ClassVar[str] = "dataset/"
    al_args: ClassVar[str] = "al_args.pickle"
    io_args: ClassVar[str] = "io_args.pickle"
    models: ClassVar[str] = "models/"
    batch: ClassVar[str] = "batch.txt"
    results: ClassVar[str] = "results.json"
    querier_wrapper: ClassVar[str] = "querier_wrapper.pickle"
    stopper_wrapper: ClassVar[str] = "stopper_wrapper.pickle"
    args: ClassVar[str] = "args.pickle"
    root: Optional[Path] = None
    overwrite: bool = False
    save_at_iteration: bool = True
    _valid: Optional[bool] = None

    def __post_init__(self) -> None:
        if self.root is None:
            self.root = tempfile.mkdtemp()
        self.root = Path(self.root)

    @property
    def dataset_path(self) -> Path:
        return self.root / self.dataset

    @property
    def al_args_path(self) -> Path:
        return self.root / self.al_args

    @property
    def io_args_path(self) -> Path:
        return self.root / self.io_args

    @property
    def valid(self) -> bool:
        valid = not self.root.exists() or is_directory_empty(self.root) or self.overwrite
        if self._valid is None:
            self._valid = valid
        return self._valid

    def models_path(self, iteration: int) -> Path:
        return self.root / str(iteration) / self.models

    def batch_path(self, iteration: int) -> Path:
        return self.root / str(iteration) / self.batch

    def results_path(self, iteration: int) -> Path:
        return self.root / str(iteration) / self.results

    def querier_wrapper_path(self, iteration: int) -> Path:
        return self.root / str(iteration) / self.querier_wrapper

    def stopper_wrapper_path(self, iteration: int) -> Path:
        return self.root / str(iteration) / self.stopper_wrapper


@dataclass
class ALArgs(SaveLoadPickleMixin):

    n_start: int = 100
    n_query: int = 100
    validation_set_size: int | float = 0.1
    disable_tqdm: bool = True
    do_train: bool = True
    do_test: bool = True


class ActiveLearner(ABC):
    def __init__(
        self,
        dataset: DatasetDict,
        pool: Pool,
        querier_wrapper: Optional[QuerierWrapper] = None,
        stopper_wrapper: Optional[StopperWrapper] = None,
        al_args: Optional[ALArgs] = None,
        io_args: Optional[IOArgs] = None,
        _iteration: int = 0,
    ) -> None:
        self.dataset = dataset
        self.pool = pool = Pool(len(self.dataset["train"])) if pool is None else pool
        self.querier_wrapper = querier_wrapper
        self.stopper_wrapper = stopper_wrapper
        self.al_args = ALArgs() if al_args is None else al_args
        self.io_args = IOArgs() if io_args is None else io_args
        self.iteration = _iteration

        if not self.io_args.valid:
            raise FileExistsError(self.io_args.root)

        if self.iteration == 0:
            self.save_first()
            self.batch = self.get_first_batch()
            self.model = self.get_model()
            self.train_dataset, self.eval_dataset = self.get_train_and_eval_datasets()
            self.trainer = self.get_trainer()
            if self.al_args.do_train:
                self.output: TrainOutput = self.trainer.train()
            else:
                self.output = None
            if self.al_args.do_test:
                self.metrics = self.trainer.evaluate(self.dataset["test"])
            self.callbacks()
        else:
            self.model, self.batch = self.load_iter()

    def __repr__(self) -> str:
        raise NotImplementedError()

    def __str__(self) -> str:
        raise NotImplementedError()

    def __call__(self) -> None:
        iterable = range(self.iteration, self.n_iterations + 1)
        if not self.al_args.disable_tqdm:
            iterable = tqdm(iterable, total=self.n_iterations, initial=self.iteration)

        for i in iterable:
            self.iteration = i
            self.batch = self.query()
            self.model = self.get_model()
            self.train_dataset, self.eval_dataset = self.get_train_and_eval_datasets()
            self.trainer = self.get_trainer()
            if self.al_args.do_train:
                self.output = self.trainer.train()
            else:
                self.output = None
            if self.al_args.do_test:
                self.metrics = self.trainer.evaluate(self.dataset["test"])
            else:
                self.metrics = None
            self.callbacks()

    @property
    def n_rows(self) -> int:
        return len(self.dataset["train"])

    @property
    def n_iterations(self) -> int:
        q, r = divmod(self.n_rows, self.al_args.n_query)
        if r == 0:
            return q
        return q + 1

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
            self._querier_wrapper = RandomQuerierWrapper(RandomQuerier(), self.pool)
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

    @classmethod
    def from_experiment(cls, output_dir: Path, iteration: int = 0) -> ActiveLearner:
        io_args = IOArgs.load(IOArgs(output_dir).io_args_path)
        al_args = ALArgs.load(io_args.al_args_path)
        dataset = Dataset.load_from_disk(io_args.dataset_path)
        querier_wrapper = QuerierWrapper.load(io_args.querier_wrapper_path)
        stopper_wrapper = StopperWrapper.load(io_args.stopper_wrapper_path)
        return cls.__init__(dataset, querier_wrapper, stopper_wrapper, al_args, io_args, iteration)

    @abstractmethod
    def get_model(self) -> PreTrainedModel:
        ...

    def get_trainer(self) -> Trainer:
        """Subclass and override to inject custom behavior."""
        return Trainer(
            model=self.model,
            args=self._get_training_args(),
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
        )

    def get_training_args(self) -> TrainingArguments:
        """Subclass and override to inject custom behavior."""
        return TrainingArguments(output_dir=None)

    def query(self) -> np.ndarray:
        """Subclass and override to inject custom behavior."""
        return self.querier_wrapper(self.al_args.n_query)

    def stop(self) -> bool:
        """Subclass and override to inject custom behavior."""
        return self.stopper_wrapper()

    def callbacks(self) -> None:
        """Subclass and override to inject custom behavior."""
        self.save_iter()

    def get_train_and_eval_datasets(self) -> tuple[Dataset, Optional[Dataset]]:
        return self.dataset["train"].select(self.pool.labeled).train_test_split(self.al_args.validation_set_size)

    def _get_training_args(self) -> TrainingArguments:
        training_args = self.get_training_args()
        training_args.output_dir = self.io_args.models_path(self.iteration)
        return training_args

    def get_first_batch(self) -> np.ndarray:
        return RandomQuerierWrapper(RandomQuerier(), self.pool)(self.al_args.n_start)

    def save_first(self) -> None:
        self.dataset.save_to_disk(self.io_args.dataset_path)
        self.al_args.save(self.io_args.al_args_path)

    def save_iter(self) -> None:
        self.querier_wrapper.save(self.io_args.querier_wrapper_path(self.iteration))
        self.stopper_wrapper.save(self.io_args.stopper_wrapper_path(self.iteration))
        np.savetxt(self.io_args.batch_path(self.iteration), self.batch)
        if self.output is not None:
            raise NotImplementedError()
        if self.metrics is not None:
            with open(self.io_args.results_path(self.iteration), "w") as handle:
                json.dump(self.metrics, handle)

    def load_iter(self) -> tuple[PreTrainedModel, np.ndarray]:
        raise NotImplementedError()
