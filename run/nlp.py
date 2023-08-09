"""
Run active learning experiments for text classification.
"""

from argparse import ArgumentParser
import os
from pathlib import Path
from pprint import pprint
import random
import sys
from typing import Optional

sys.path.insert(0, ".")

from datasets import load_dataset
import evaluate
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollator,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    logging,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
import torch
from tqdm import tqdm

from mdalt.analysis import Analyzer
from mdalt.cfg import BR
from mdalt.helpers import IOHelper, Pool
from mdalt.learning import validate, Config, IOHelper, Learner, Evaluator, TrainerFactory


parser = ArgumentParser()
parser.add_argument("--learn", action="store_true")
parser.add_argument("--evaluate", action="store_true")
parser.add_argument("--analyze", action="store_true")
parser.add_argument("--subset", type=int, default=-1)
parser.add_argument("--dataset", type=str, default="imdb", choices=["imdb", "ag_news"])
parser.add_argument("--pretrained_model_name_or_path", type=str, default="distilbert-base-uncased")
parser.add_argument("--metric", type=str, default="accuracy")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--verbosity", type=int, choices=[10, 20, 30, 40, 50], default=logging.WARNING)
args = parser.parse_args()


LEARN = args.learn
EVALUATE = args.evaluate
ANALYZE = args.analyze
SUBSET = args.subset if args.subset > 0 else None
DATASET = args.dataset
PRETRAINED_MODEL_NAME_OR_PATH = args.pretrained_model_name_or_path
METRIC = args.metric
SEED = args.seed
VERBOSITY = args.verbosity


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


dataset = load_dataset(DATASET)
if SUBSET:
    dataset["train"] = dataset["train"].select(range(SUBSET))
    dataset["test"] = dataset["test"].select(range(SUBSET))


tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME_OR_PATH)


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)


dataset = dataset.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
metric = evaluate.load(METRIC)


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
    output_dir="/tmp/PLACEHOLDER_WILL_BE_SET_BY_CONFIG",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    save_total_limit=1,
    optim="adamw_torch",
)
validate(training_args)
callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]

pool = Pool(dataset["train"])
config = Config(n_rows=dataset["train"].num_rows)
io_helper = IOHelper(Path("./output"), overwrite=True)
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
    learner = Learner(pool, config, io_helper, trainer_fact)
    learner = learner()
    for model, train_output in tqdm(learner):
        ...

if EVALUATE:
    evaluator = Evaluator(trainer_fact, dataset["test"], io_helper)
    evaluator = evaluator()
    for model, results in tqdm(evaluator):
        ...

if ANALYZE:
    analyzer = Analyzer(io_helper)
    fig, ax = analyzer()
    fig.savefig(io_helper.learning_curve_path, dpi=400)
