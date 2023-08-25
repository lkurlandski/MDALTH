"""
Active learning loop.
"""

from __future__ import annotations
import json
from pprint import pformat
import shutil
from typing import Optional

from datasets import Dataset
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import TrainOutput

from mdalth.helpers import IOHelper, Pool, TrainerFactory
from mdalth.querying import RandomQuerier
from mdalth.utils import load_with_pickle, save_with_pickle


def validate(training_args: TrainingArguments) -> None:
    assert training_args.load_best_model_at_end is True


class Config:
    def __init__(
        self,
        n_rows: Optional[int] = None,
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
        if isinstance(self._n_query, float) or isinstance(self._n_start, float):
            if self.n_rows is None:
                raise ValueError("n_rows must be specified if n_start or n_query is a float.")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(\n{pformat(vars(self))}\n)"

    @property
    def n_iterations(self) -> int:
        q, r = divmod(self.n_rows - self.n_start, self.n_query)
        if r == 0:
            return q
        return q + 1

    @property
    def n_start(self) -> int:
        if isinstance(self._n_start, float):
            return int(self._n_start * self.n_rows)
        return self._n_start

    @property
    def n_query(self) -> int:
        if isinstance(self._n_query, float):
            return int(self._n_query * self.n_rows)
        return self._n_query

    def validation_set_size(self, num_labeled: Optional[int] = None) -> int:
        if isinstance(self._validation_set_size, int):
            return self._validation_set_size
        if isinstance(self._validation_set_size, float) and num_labeled is not None:
            return int(self._validation_set_size * num_labeled)
        raise RuntimeError()


class Learner:
    def __init__(
        self,
        pool: Pool,
        config: Config,
        io_helper: IOHelper,
        trainer_fact: TrainerFactory,
        _iteration: int = 0,
    ) -> None:
        self.pool = pool
        self.config = config
        self.io_helper = io_helper
        self.trainer_fact = trainer_fact
        self.iteration = _iteration
        if not self.io_helper.valid:
            raise FileExistsError(self.io_helper.root_path)

    def __call__(self) -> Learner:
        if self.iteration != 0:
            raise RuntimeError("The zeroth iteration has already been run.")
        self.pre()
        batch = self.query()
        trainer, train_output = self.train(batch)
        self.save_to_disk(batch, trainer.model, train_output)
        self.post()
        self.iteration += 1
        return self

    def __iter__(self) -> Learner:
        return self

    def __len__(self) -> int:
        return self.config.n_iterations

    def __next__(self) -> tuple[PreTrainedModel, TrainOutput]:
        if self.iteration == 0:
            raise RuntimeError("The zeroth iteration has not been run.")
        if self.iteration > len(self):
            raise StopIteration()
        self.pre()
        batch = self.query_first()
        trainer, train_output = self.train(batch)
        self.save_to_disk(batch, trainer.model, train_output)
        self.post()
        self.iteration += 1
        return trainer.model, train_output

    @property
    def dataset(self) -> Dataset:
        return self.pool.dataset

    @property
    def num_rows(self) -> int:
        return self.dataset.num_rows

    # TODO: implement checkpointing system.
    # @classmethod
    # def load_from_disk(cls, output_dir: Path, iteration: Optional[int] = None) -> Learner:
    #     io_helper = IOHelper(output_dir)
    #     if not iteration:
    #         iteration = get_highest_path(io_helper.iterations_path.glob("*"))
    #     dataset = Dataset.load_from_disk(io_helper.tr_dataset_path)
    #     batches = [np.loadtxt(io_helper.batch_path(i)) for i in range(iteration)]
    #     labeled_idx = np.concatenate(batches)
    #     pool = Pool.from_pools(dataset, labeled_idx)
    #     config = load_with_pickle(io_helper.config_path)
    #     trainer_fact = load_with_pickle()
    #     return cls(pool, config, io_helper, trainer_fact, iteration)

    def save_to_disk(
        self, batch: np.ndarray, model: PreTrainedModel, train_output: TrainOutput
    ) -> None:
        if self.iteration == 0:
            self.io_helper.mkdir(exist_ok=True)
            save_with_pickle(self.io_helper.config_path, self.config)
            self.dataset.save_to_disk(self.io_helper.tr_dataset_path)

        np.savetxt(self.io_helper.batch_path(self.iteration), batch)
        model.save_pretrained(self.io_helper.model_path(self.iteration))
        save_with_pickle(self.io_helper.trainer_output_path(self.iteration), train_output)

    def train(self, batch: np.ndarray) -> tuple[Trainer, TrainOutput]:
        self.pool.label(batch)
        test_size = self.config.validation_set_size(len(self.pool.labeled_idx))
        dataset = self.pool.labeled.train_test_split(test_size=test_size)
        trainer = self.trainer_fact(dataset["train"], dataset["test"])
        trainer.args.output_dir = self.io_helper.checkpoints_path(self.iteration)
        train_output = trainer.train()
        return trainer, train_output

    def pre(self) -> None:
        ...

    def post(self) -> None:
        shutil.rmtree(self.io_helper.checkpoints_path(self.iteration))

    def stop(self) -> bool:
        return False

    def query_first(self) -> np.ndarray:
        return RandomQuerier()(self.config.n_start, self.pool.unlabeled_idx)

    def query(self) -> np.ndarray:
        return RandomQuerier()(self.config.n_query, self.pool.unlabeled_idx)


class Evaluator:
    def __init__(
        self,
        ts_trainer_fact: TrainerFactory,
        ts_dataset: Dataset,
        io_helper: IOHelper,
        _iteration: int = 0,
    ) -> None:
        self.ts_trainer_fact = ts_trainer_fact
        self.ts_dataset = ts_dataset
        self.io_helper = io_helper
        if not self.io_helper.exists(ts_dataset=False):
            raise FileNotFoundError(self.io_helper.root_path)
        self.iteration = _iteration
        self.config: Config = load_with_pickle(io_helper.config_path)
        self.pool = Pool(Dataset.load_from_disk(io_helper.tr_dataset_path))

    def __call__(self) -> Evaluator:
        return self

    def __iter__(self) -> Learner:
        return self

    def __len__(self) -> int:
        return self.config.n_iterations

    def __next__(self) -> tuple[PreTrainedModel, TrainOutput]:
        if self.iteration > len(self):
            raise StopIteration()
        self.pre()
        batch = np.loadtxt(self.io_helper.batch_path(self.iteration), dtype=np.int64)
        self.pool.label(batch)
        trainer, results = self.eval()
        self.save_to_disk(results)
        self.post()
        self.iteration += 1
        return trainer.model, results

    @property
    def tr_dataset(self) -> Dataset:
        return self.pool.dataset

    @property
    def tr_num_rows(self) -> int:
        return self.tr_dataset.num_rows

    @property
    def ts_num_rows(self) -> int:
        return self.ts_dataset.num_rows

    # TODO: implement checkpointing system.
    # @classmethod
    # def load_from_disk(cls, output_dir: Path, iteration: Optional[int] = None) -> Evaluator:
    #     io_helper = IOHelper(output_dir)
    #     if not iteration:
    #         complete = []
    #         for p in io_helper.iterations_path.glob("*"):
    #             if io_helper._test_metrics in [p_.name for p_ in p.iterdir()]:
    #                 complete.append(io_helper.iterations_path / p.name)
    #         iteration = get_highest_path(complete)

    def save_to_disk(self, results: dict[str, float]) -> None:
        if self.iteration == 0:
            self.ts_dataset.save_to_disk(self.io_helper.ts_dataset_path)
        with open(self.io_helper.test_metrics_path(self.iteration), "w") as f:
            json.dump(results, f)

    def eval(self) -> tuple[Trainer, dict[str, float]]:
        model = AutoModelForSequenceClassification.from_pretrained(
            self.io_helper.model_path(self.iteration)
        )
        ts_trainer = self.ts_trainer_fact(None, self.ts_dataset, model)
        results = ts_trainer.evaluate(self.ts_dataset)
        return ts_trainer, results

    def pre(self) -> None:
        ...

    def post(self) -> None:
        ...
