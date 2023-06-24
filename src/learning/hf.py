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

from src.pool import Pool
from src.querying.core import Querier, RandomQuerier
from src.querying.hf import QuerierWrapper, RandomQuerierWrapper
from src.stopping.core import Stopper, ContinuousStopper
from src.stopping.hf import StopperWrapper, ContinuousStopperWrapper
from src.utils import (
    get_highest_path,
    is_directory_empty,
    load_with_pickle,
    save_with_pickle,
)


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
            | - al_args.pickle
            | - io_args.pickle
    """

    _meta: ClassVar[str] = "meta/"
    _dataset: ClassVar[str] = "dataset/"
    _querier_wrapper: ClassVar[str] = "querier_wrapper.pickle"
    _stopper_wrapper: ClassVar[str] = "stopper_wrapper.pickle"
    _al_args: ClassVar[str] = "al_args.pickle"
    _io_args: ClassVar[str] = "io_args.pickle"

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
    def al_args_path(self) -> Path:
        return self.meta_path / self._al_args

    @property
    def io_args_path(self) -> Path:
        return self.meta_path / self._io_args

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


@dataclass
class ALArgs:
    """Arguments for the generalized active learning loop."""

    n_start: int = 100
    n_query: int = 100
    validation_set_size: int | float = 0.1
    do_train: bool = True
    do_test: bool = True


class ActiveLearner(ABC):
    """Generalized active learning loop with subclassing for custom behavior."""

    def __init__(
        self,
        dataset: DatasetDict,
        pool: Pool,
        querier_wrapper: Optional[QuerierWrapper] = None,
        stopper_wrapper: Optional[StopperWrapper] = None,
        al_args: Optional[ALArgs] = None,
        io_args: Optional[IOHelper] = None,
        _iteration: int = 0,
    ) -> None:
        self.dataset = dataset
        self.pool = pool = Pool(len(self.dataset["train"])) if pool is None else pool
        self.querier_wrapper = querier_wrapper
        self.stopper_wrapper = stopper_wrapper
        self.al_args = ALArgs() if al_args is None else al_args
        self.io_args = IOHelper() if io_args is None else io_args
        self.iteration = _iteration
        self.model = None
        self.batch = None
        self.train_dataset = None
        self.eval_dataset = None
        self.test_metrics = None
        self.trainer_output = None
        self.trainer = None
        if not self.io_args.valid:
            raise FileExistsError(self.io_args.root_path)

    def __call__(self) -> None:
        if self.iteration != 0:
            raise RuntimeError()

        self.batch = self.get_first_batch()
        self._train_one_iteration()

    def __iter__(self) -> ActiveLearner:
        return self

    def __len__(self) -> int:
        q, r = divmod(self.n_rows, self.al_args.n_query)
        if r == 0:
            return q
        return q + 1

    def __next__(self) -> None:
        if self.iteration == 0:
            raise RuntimeError()
        if self.iteration > len(self):
            raise StopIteration()

        self.iteration += 1
        self.batch = self.query()
        self._train_one_iteration()

    def __str__(self) -> str:
        raise NotImplementedError()

    @property
    def n_rows(self) -> int:
        return len(self.dataset["train"])

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
    def load_from_disk(cls, output_dir: Path, iteration: Optional[int] = None) -> ActiveLearner:
        if iteration is None:
            iteration = get_highest_path(output_dir.glob("*"))
        io_args = load_with_pickle(IOHelper(output_dir).io_args_path)
        al_args = load_with_pickle(io_args.al_args_path)
        dataset = Dataset.load_from_disk(io_args.dataset_path)
        querier_wrapper = load_with_pickle(io_args.querier_wrapper_path)
        stopper_wrapper = load_with_pickle(io_args.stopper_wrapper_path)
        return cls.__init__(dataset, querier_wrapper, stopper_wrapper, al_args, io_args, iteration)

    def save_to_disk(self) -> None:
        if self.iteration == 0:
            self.dataset.save_to_disk(self.io_args.dataset_path)
            save_with_pickle(self.io_args.querier_wrapper_path, self.querier_wrapper)
            save_with_pickle(self.io_args.stopper_wrapper_path, self.stopper_wrapper)
            save_with_pickle(self.io_args.al_args_path, self.al_args)
            save_with_pickle(self.io_args.io_args_path, self.io_args)

        np.savetxt(self.io_args.batch_path(self.iteration), self.batch)
        if self.al_args.do_train:
            save_with_pickle(self.io_args.trainer_output_path(self.iteration), self.trainer_output)
        if self.al_args.do_test:
            with open(self.io_args.test_metrics_path(self.iteration), "w") as handle:
                json.dump(self.test_metrics, handle)

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
        self.save_to_disk()

    def get_train_and_eval_datasets(self) -> tuple[Dataset, Optional[Dataset]]:
        """Subclass and override to inject custom behavior."""
        return self.dataset["train"].select(self.pool.labeled).train_test_split(self.al_args.validation_set_size)

    def get_first_batch(self) -> np.ndarray:
        """Subclass and override to inject custom behavior."""
        return RandomQuerierWrapper(RandomQuerier(), self.pool)(self.al_args.n_start)

    def _get_training_args(self) -> TrainingArguments:
        training_args = self.get_training_args()
        training_args.output_dir = self.io_args.models_path(self.iteration)
        return training_args

    def _train_one_iteration(self) -> TrainOutput:
        self.model = self.get_model()
        self.train_dataset, self.eval_dataset = self.get_train_and_eval_datasets()
        self.trainer = self.get_trainer()
        if self.al_args.do_train:
            self.trainer_output: TrainOutput = self.trainer.train()
        if self.al_args.do_test:
            self.test_metrics = self.trainer.evaluate(self.dataset["test"])
        self.callbacks()
