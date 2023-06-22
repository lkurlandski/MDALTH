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

from querying.wrappers import QuerierWrapper, RandomWrapper
from querying.queriers import RandomQuerier
from stopping.stoppers import NoOpStopper
from stopping.wrappers import StopperWrapper, NoOpWrapper
from utils import is_directory_empty, pickler


@dataclass
class IOArgs:

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

    def __post_init__(self) -> None:
        if self.root is None:
            self.root = tempfile.mkdtemp()[1]
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
    def models_path(self, iteration: int) -> Path:
        return self.root / str(iteration) / self.models

    @property
    def batch_path(self, iteration: int) -> Path:
        return self.root / str(iteration) / self.batch

    @property
    def results_path(self, iteration: int) -> Path:
        return self.root / str(iteration) / self.results

    @property
    def querier_wrapper_path(self, iteration: int) -> Path:
        return self.root / str(iteration) / self.querier_wrapper

    @property
    def stopper_wrapper_path(self, iteration: int) -> Path:
        return self.root / str(iteration) / self.stopper_wrapper

    @classmethod
    def load(cls, path: Path, **kwds) -> IOArgs:
        return pickler(path, **kwds)

    def save(self, path: Path, **kwds) -> None:
        pickler(path, self, **kwds)


@dataclass
class ALArgs:

    n_start: int = 1000
    n_query: int = 500
    validation_set_size: int | float = 0.1
    disable_tqdm: bool = True
    do_train: bool = True
    do_test: bool = True

    @classmethod
    def load(cls, path: Path, **kwds) -> ALArgs:
        return pickler(path, **kwds)

    def save(self, path: Path, **kwds) -> None:
        pickler(path, self, **kwds)


class ActiveLearner(ABC):
    def __init__(
        self,
        dataset: DatasetDict,
        *,
        querier_wrapper: Optional[QuerierWrapper] = None,
        stopper_wrapper: Optional[StopperWrapper] = None,
        al_args: Optional[ALArgs] = None,
        io_args: Optional[IOArgs] = None,
        _iteration: Optional[int] = 0,
    ) -> None:
        self.dataset = dataset

        if querier_wrapper is None:
            querier = RandomQuerier(len(dataset["train"]))
            self.querier_wrapper = RandomWrapper(querier)
        else:
            self.querier_wrapper = querier_wrapper

        if stopper_wrapper is None:
            stopper = NoOpStopper()
            self.stopper_wrapper = NoOpWrapper(stopper)
        else:
            self.stopper_wrapper = stopper_wrapper

        self.al_args = ALArgs() if al_args is None else al_args
        self.io_args = IOArgs() if io_args is None else io_args
        self.iteration = _iteration

        print(io_args)
        if self.io_args.root.exists() and not is_directory_empty(self.io_args.root) and not self.io_args.overwrite:
            raise FileExistsError(
                f"{self.io_args.root.as_posix()} is not empty and {self.io_args.overwrite=}."
                "Either manually delete the directory or set overwrite=True in __init__."
            )

        self._save(first=True)

    def __repr__(self) -> str:
        raise NotImplementedError()

    def __str__(self) -> str:
        raise NotImplementedError()

    def __call__(self) -> None:
        iterable = range(self.iteration, self.n_iterations)
        if not self.al_args.disable_tqdm:
            iterable = tqdm(iterable, total=self.n_iterations, initial=self.iteration)

        for iteration in iterable:
            self.iteration = iteration
            model = self.get_model()
            batch = self.querier_wrapper(model, self.dataset["train"], self.al_args.n_query)
            train_dataset, eval_dataset = self.get_train_eval_dataset()
            trainer = Trainer(
                model=model,
                args=self._get_training_args(),
                data_collator=self.get_data_collator(),
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=self.get_tokenizer(),
                compute_metrics=self.get_compute_metrics(),
                callbacks=self.get_callbacks(),
                optimizers=self.get_optimizers(),
                disable_tqdm=self.al_args.disable_tqdm,
            )

            output: TrainOutput = trainer.train() if self.al_args.do_train else None
            metrics = trainer.evaluate(self.dataset["test"]) if self.al_args.do_test else None
            self._save(batch, output, metrics)

    @property
    def n_rows(self) -> int:
        return len(self.dataset["train"])

    @property
    def n_iterations(self) -> int:
        q, r = divmod(self.n_rows, self.al_args.n_query)
        if r == 0:
            return q
        return q + 1

    @classmethod
    def load(cls, output_dir: Path, iteration: int = 0) -> ActiveLearner:
        io_args = IOArgs.load(IOArgs(output_dir).io_args_path)
        al_args = ALArgs.load(io_args.al_args_path)
        dataset = Dataset.load_from_disk(io_args.dataset_path)
        querier_wrapper = QuerierWrapper.load(io_args.querier_wrapper_path)
        stopper_wrapper = StopperWrapper.load(io_args.stopper_wrapper_path)
        return cls.__init__(dataset, querier_wrapper, stopper_wrapper, al_args, io_args, iteration)

    def get_data_collator(self) -> Optional[DataCollator]:
        return None

    def get_tokenizer(self) -> Optional[PreTrainedTokenizerBase]:
        return None

    def get_compute_metrics() -> Optional[Callable[[EvalPrediction], dict]]:
        return None

    def get_callbacks(self) -> Optional[list[TrainerCallback]]:
        return None

    def get_optimizers(self) -> Optional[Optimizer | tuple[Optimizer, LambdaLR]]:
        return None

    def get_train_eval_dataset(self) -> tuple[Dataset, Optional[Dataset]]:
        dataset: Dataset = self.dataset["train"][self.querier_wrapper.labeled]
        dataset: DatasetDict = dataset.train_test_split(self.al_args.validation_set_size)
        return dataset["train"], dataset["test"]

    @abstractmethod
    def get_model(self) -> PreTrainedModel:
        ...

    @abstractmethod
    def get_training_args(self) -> TrainingArguments:
        ...

    def _get_training_args(self) -> TrainingArguments:
        training_args = self.get_training_args()
        training_args.output_dir = self.io_args.models_path(self.iteration)
        return training_args

    def _save(
        self,
        batch: Optional[np.ndarray] = None,
        output: Optional[TrainOutput] = None,
        metrics: Optional[dict[str, float]] = None,
        *,
        first: bool = False,
    ) -> None:
        if first:
            self.dataset.save_to_disk(self.io_args.dataset_path)
            self.al_args.save(self.io_args.al_args_path)
            return

        self.querier_wrapper.save(self.io_args.querier_wrapper_path(self.iteration))
        self.stopper_wrapper.save(self.io_args.stopper_wrapper_path(self.iteration))
        np.savetxt(self.io_args.batch_path(self.iteration), batch)
        if output is not None:
            raise NotImplementedError()
        if metrics is not None:
            with open(self.io_args.results_path(self.iteration), "w") as handle:
                json.dump(metrics, handle)
