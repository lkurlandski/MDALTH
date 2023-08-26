"""
Run active learning experiments for text classification.
"""

from argparse import ArgumentParser
import os
from pathlib import Path
import random
import sys
from typing import Any, Optional

sys.path.insert(0, ".")

from datasets import concatenate_datasets, load_dataset
import evaluate
import numpy as np
from transformers import (
    logging,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    PreTrainedModel,
    TrainingArguments,
)
import torch
from torch import tensor
from torch.nn.functional import softmax
from tqdm import tqdm

from mdalth.analysis import Analyzer
from mdalth.cfg import BR
from mdalth.helpers import IOHelper, Pool
from mdalth.learning import Config, IOHelper, Learner, Evaluator, TrainerFactory
from mdalth.querying import UncertaintyQuerier
from mdalth.stopping import StabilizingPredictions


parser = ArgumentParser()
parser.add_argument("--learn", action="store_true")
parser.add_argument("--evaluate", action="store_true")
parser.add_argument("--analyze", action="store_true")
parser.add_argument("--subset", type=int, default=-1)
parser.add_argument("--dataset", type=str, default="imdb", choices=["imdb", "ag_news"])
parser.add_argument("--pretrained_model_name_or_path", type=str, default="distilbert-base-uncased")
parser.add_argument("--metric", type=str, default="accuracy")
parser.add_argument("--output_root", type=str, default="./output")
parser.add_argument("--querier", type=str, required=False, choices=["random", "uncertainty"])
parser.add_argument(
    "--stopper",
    type=str,
    required=False,
    choices=["null", "stabilizing_predictions", "changing_confidence"],
)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--verbosity", type=int, choices=[10, 20, 30, 40, 50], default=logging.WARNING)
parser.add_argument("--n_iterations", type=int, required=False)
args = parser.parse_args()


LEARN: bool = args.learn
EVALUATE: bool = args.evaluate
ANALYZE: bool = args.analyze
SUBSET: Optional[int] = args.subset if args.subset > 0 else None
DATASET: str = args.dataset
PRETRAINED_MODEL_NAME_OR_PATH: str = args.pretrained_model_name_or_path
METRIC: str = args.metric
OUTPUT_ROOT: str = args.output_root
QUERIER: Optional[str] = args.querier
STOPPER: Optional[str] = args.stopper
SEED: int = args.seed
VERBOSITY: int = args.verbosity
N_ITERATIONS: Optional[int] = args.n_iterations


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# First, we need to define the interface between the sampling algorithms, the
# stopping methods, and the Learner.
# This is quite clumsy, but we don't have a unified API for querying and
# stopping yet.
# Note that none if this is necessary if you use the default settings, i.e.,
# random sampling and no stopping.
# If that is the case, just use the vanilla Learner :).


class UncertaintyLearner(Learner):
    """Interfaces between the uncertainty sampling algorithm and the Learner."""

    def query(self) -> np.ndarray:
        local_to_global_idx = {
            i: self.pool.unlabeled_idx[i] for i in range(len(self.pool.unlabeled_idx))
        }
        local_dataset = self.dataset.select(self.pool.unlabeled_idx)
        trainer = self.trainer_fact(eval_dataset=local_dataset, model=self.state.trainer.model)
        prediction_output = trainer.predict(local_dataset)
        classwise_probs = softmax(tensor(prediction_output.predictions), dim=1).numpy()
        local_idx = self.querier(self.config.n_query, classwise_probs)
        global_idx = np.array([local_to_global_idx[i] for i in local_idx])
        return global_idx


class StabilizingPredictionsLearner(Learner):
    """Interfaces between stabilizing predictions and the Learner."""

    def stop(self) -> bool:
        raise NotImplementedError()


# Collect the required base classes and construct a new derived type.
# For Method Resolution Order to work out properly, the order at which base
# classes are passed to `type` matters.
# In particular, make sure `Learner` is the last base class, else none of the
# methods overridden will be available.


bases = []
if QUERIER == "uncertainty":
    bases.append(UncertaintyLearner)
if STOPPER == "stabilizing_predictions":
    bases.append(StabilizingPredictionsLearner)
Learner = type("Learner", tuple(bases + [Learner]), {})


# Finally, build the querier and stopper.
# The rest is thankfully fairly straight forward.
# Hopefully, we will be able to simplify that bit above in the future.

querier = None
if QUERIER == "uncertainty":
    querier = UncertaintyQuerier("U")
stopper = None
if STOPPER == "stabilizing_predictions":
    stopper = StabilizingPredictions(3, 0.99)


dataset = load_dataset(DATASET)
if SUBSET:
    dataset["train"] = dataset["train"].select(range(SUBSET))
    dataset["test"] = dataset["test"].select(range(SUBSET))


tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME_OR_PATH)


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)


dataset = dataset.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
metric = evaluate.load(METRIC)
callbacks = [EarlyStoppingCallback(early_stopping_patience=2)]


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)


def model_init() -> PreTrainedModel:
    return AutoModelForSequenceClassification.from_pretrained(
        PRETRAINED_MODEL_NAME_OR_PATH,
        num_labels=dataset["train"].features["label"].num_classes,
    )


training_args = TrainingArguments(
    output_dir="PLACEHOLDER_WILL_BE_SET_BY_CONFIG",
    learning_rate=2e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=1024,
    num_train_epochs=10,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    save_total_limit=1,
    optim="adamw_torch",
)
pool = Pool(dataset["train"])
config = Config(
    n_rows=dataset["train"].num_rows,
    n_start=64,
    n_query=64,
    n_iterations=N_ITERATIONS,
)
io_helper = IOHelper(Path(OUTPUT_ROOT), overwrite=True)
trainer_fact = TrainerFactory(
    model_init=model_init,
    args=training_args,
    data_collator=data_collator,
    tokenizer=tokenizer,
    callbacks=callbacks,
    compute_metrics=compute_metrics,
)

print(f"{dataset=}\n", BR)
print(f"{data_collator=}\n", BR)
print(f"{tokenizer=}\n", BR)
print(f"{callbacks=}\n", BR)
print(f"{model_init()=}\n", BR)
print(f"{training_args=}\n", BR)
print(f"{pool=}\n", BR)
print(f"{config=}\n", BR)
print(f"{io_helper=}\n", BR)
print(flush=True)


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

logging.set_verbosity(VERBOSITY)

if LEARN:
    learner = Learner(pool, config, io_helper, trainer_fact, querier, stopper)
    learner_state = learner()
    for i, learner_state in enumerate(tqdm(learner), 1):
        ...

if EVALUATE:
    evaluator = Evaluator(trainer_fact, dataset["test"], io_helper)
    evaluator = evaluator()
    for i, (model, results) in enumerate(tqdm(evaluator), 1):
        ...

if ANALYZE:
    analyzer = Analyzer(io_helper)
    fig, ax = analyzer()
    fig.savefig(io_helper.learning_curve_path, dpi=400)
