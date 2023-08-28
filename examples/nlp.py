"""
Run active learning experiments for text classification.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
import random
import sys
from typing import Optional

from datasets import load_dataset
import evaluate
import numpy as np
from transformers import (
    logging,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    HfArgumentParser,
    PreTrainedModel,
    TrainingArguments,
)
import torch
from tqdm import tqdm

sys.path.insert(0, ".")  # pylint: disable=wrong-import-position

from mdalth.cfg import BR
from mdalth.helpers import IOHelper, Pool
from mdalth.learning import Config, Learner, Evaluator, TrainerFactory
from mdalth.querying import (
    EntropyQuerier,
    MarginQuerier,
    UncertaintyQuerier,
)
from mdalth.stopping import (
    ChangingConfidence,
    StabilizingPredictions,
)
from mdalth.tp import ProportionOrInteger
from mdalth.utils import proportion_or_integer_to_int


@dataclass
class Arguments:

    dataset: str = field(default="ag_news", metadata={"help": "Dataset to use."})
    pretrained_model_name_or_path: str = field(
        default="distilbert-base-uncased", metadata={"help": "Pretrained model to use."}
    )
    metric: str = field(default="accuracy", metadata={"help": "Metric to use."})
    querier: str = field(default="random", metadata={"help": "Querier to use."})
    stopper: str = field(default="null", metadata={"help": "Stopper to use."})
    subset: ProportionOrInteger = field(default=-1, metadata={"help": "Subset of the dataset to use."})
    log_level: int = field(default=40, metadata={"help": "Sets the log level."})
    seed: int = field(default=0, metadata={"help": "Random seed."})


parser = HfArgumentParser([Arguments, Config])
args, config = parser.parse_args_into_dataclasses()
args: Arguments = args
config: Config = config

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)


if args.querier == "random":
    querier = None
elif args.querier == "uncertainty":
    querier = UncertaintyQuerier()
elif args.querier == "margin":
    querier = MarginQuerier()
elif args.querier == "entropy":
    querier = EntropyQuerier()
else:
    raise ValueError(f"Unknown querier {args.querier}.")

if args.stopper == "null":
    stopper = None
elif args.stopper == "stabilizing_predictions":
    stopper = StabilizingPredictions(3, 0.99)
elif args.stopper == "changing_confidence":
    stopper = ChangingConfidence(3, 0.99)
else:
    raise ValueError(f"Unknown stopper {args.stopper}.")


output_root = Path(f"./examples/output/nlp/{args.querier}/{args.stopper}")
output_root.mkdir(parents=True, exist_ok=True)


dataset = load_dataset(args.dataset)
if args.subset > 0:
    dataset["train"] = dataset["train"].select(
        range(proportion_or_integer_to_int(args.subset, len(dataset["train"])))
    )
    dataset["test"] = dataset["test"].select(
        range(proportion_or_integer_to_int(args.subset, len(dataset["test"])))
    )


config.configure(len(dataset["train"]))


tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)


dataset = dataset.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
metric = evaluate.load(args.metric)
callbacks = [EarlyStoppingCallback(early_stopping_patience=2)]


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)


def model_init() -> PreTrainedModel:
    return AutoModelForSequenceClassification.from_pretrained(
        args.pretrained_model_name_or_path,
        num_labels=dataset["train"].features["label"].num_classes,
    )


training_args = TrainingArguments(
    output_dir="WILL_BE_IGNORED",
    learning_rate=2e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=768,
    num_train_epochs=25,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    save_total_limit=1,
    optim="adamw_torch",
    group_by_length=True,
    fp16=True,
)
pool = Pool(dataset["train"])
io_helper = IOHelper(output_root / config.output_root(), overwrite=True)
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

logging.set_verbosity(args.log_level)

if config.learn:
    if config.resume or config.resume_from_checkpoint:
        learner = Learner.load_from_disk(
            io_helper.root_path,
            config,
            trainer_fact,
            querier,
            stopper,
            iteration=config.resume_from_checkpoint,
        )
    else:
        learner = Learner(pool, config, io_helper, trainer_fact, querier, stopper)
        learner()
    for i, learner_state in enumerate(tqdm(learner), learner.iteration):
        ...

if config.evaluate:
    if config.resume or config.resume_from_checkpoint:
        raise NotImplementedError()
        evaluator = Evaluator.load_from_disk()
    else:
        evaluator = Evaluator(trainer_fact, dataset["test"], io_helper)
        evaluator()
    for i, (model, results) in enumerate(tqdm(evaluator), 1):
        ...
