"""
Run active learning experiments for classification tasks.
"""

from collections import defaultdict
from dataclasses import dataclass, field
import os
from pathlib import Path
import random
import sys
from typing import Optional

from datasets import load_dataset, Dataset
import evaluate
import numpy as np
from transformers import (
    EarlyStoppingCallback,
    EvalPrediction,
    HfArgumentParser,
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

from example.task_manager import TextTaskManager, ImageTaskManager, AudioTaskManager


# Additional keyword arguments supplied to load_dataset for specific datasets.
DATASET_KWARGS = defaultdict(
    dict,
    {
        "PolyAI/minds14": dict(name="en-US"),
        "speech_commands": dict(name="v0.02"),
    }
)


# Additional dataset wrangling functions for specific datasets.
DATASET_WRANGLING = defaultdict(
    lambda: (lambda ds: ds),
    {
        "PolyAI/minds14": lambda ds: ds.rename_column("intent_class", "label"),
    }
)


# Evaluation metrics for classification.
ACCURACY = evaluate.load("accuracy")


@dataclass
class Arguments:

    task: str = field(metadata={"help": "Task to run."})
    dataset: str = field(metadata={"help": "Dataset to use."})
    pretrained_model_name_or_path: str = field(metadata={"help": "Pretrained model to use."})
    querier: str = field(default="random", metadata={"help": "Querier to use."})
    stopper: str = field(default="null", metadata={"help": "Stopper to use."})
    subset: Optional[ProportionOrInteger] = field(
        default=None, metadata={"help": "Subset of the dataset to use."}
    )


def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return ACCURACY.compute(predictions=predictions, references=labels)


def main(args: Arguments, config: Config, training_args: TrainingArguments) -> None:

    # Initialize seeds.
    random.seed(training_args.seed)
    np.random.seed(training_args.seed)
    torch.manual_seed(training_args.seed)

    # Initialize querier.
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

    # Initialize stopper.
    if args.stopper == "null":
        stopper = None
    elif args.stopper == "stabilizing_predictions":
        stopper = StabilizingPredictions(3, 0.99)
    elif args.stopper == "changing_confidence":
        stopper = ChangingConfidence(3, 0.99)
    else:
        raise ValueError(f"Unknown stopper {args.stopper}.")

    # Load the dataset and ensure it has the required splits.
    dataset = load_dataset(args.dataset, **DATASET_KWARGS[args.dataset])
    if isinstance(dataset, Dataset):
        dataset = dataset.train_test_split()
    if "validation" in dataset.keys():
        dataset["test"] = dataset.pop("validation")
    if "train" in dataset.keys() and "test" not in dataset.keys():
        dataset = dataset["train"].train_test_split()
    if "train" not in dataset.keys() or "test" not in dataset.keys():
        raise ValueError(f"Unable to find train and test splits in {dataset=}.")

    # Apply some pre-preprocessing to the dataset.
    dataset = DATASET_WRANGLING[args.dataset](dataset)
    if args.subset is not None and args.subset > 0:
        _train_idx = range(proportion_or_integer_to_int(args.subset, len(dataset["train"])))
        _test_idx = range(proportion_or_integer_to_int(args.subset, len(dataset["test"])))
        dataset["train"] = dataset["train"].select(_train_idx)
        dataset["test"] = dataset["test"].select(_test_idx)

    # Initialize the task manager, which abstracts the actions for text, image, and audio.
    id2label = {str(i): l for i, l in enumerate(dataset["train"].features["label"].names)}
    if args.task == "text":
        task_manager = TextTaskManager(args.pretrained_model_name_or_path, id2label)
    elif args.task == "image":
        task_manager = ImageTaskManager(args.pretrained_model_name_or_path, id2label)
    elif args.task == "audio":
        task_manager = AudioTaskManager(args.pretrained_model_name_or_path, id2label)
    else:
        raise ValueError(f"Unknown task {args.task}.")

    # Preprocess the dataset accordingly.
    dataset = dataset.map(
        task_manager.preprocess_function,
        remove_columns=task_manager.remove_columns,
        batched=True,
    )

    # Make output directories.
    output_root = Path(f"./example/output/{args.task}/{args.querier}/{args.stopper}")
    output_root.mkdir(parents=True, exist_ok=True)

    # Create/setup the MDALTH helpers.
    config.configure(len(dataset["train"]))
    pool = Pool(dataset["train"])
    io_helper = IOHelper(output_root / config.output_root(), overwrite=True)
    trainer_fact = TrainerFactory(
        model_init=lambda: task_manager.model_init(),
        args=training_args,
        data_collator=task_manager.data_collator,
        tokenizer=task_manager.tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        compute_metrics=compute_metrics,
    )

    print(f"{dataset=}\n", BR)
    print(f"{task_manager.data_collator=}\n", BR)
    print(f"{task_manager.tokenizer=}\n", BR)
    print(f"{task_manager.model_init()=}\n", BR)
    print(f"{training_args=}\n", BR)
    print(f"{pool=}\n", BR)
    print(f"{config=}\n", BR)
    print(f"{io_helper=}\n", BR)
    print(flush=True)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

    if config.learn:
        learner = Learner(pool, config, io_helper, trainer_fact, querier, stopper)
        learner()
        for i, learner_state in enumerate(tqdm(learner), learner.iteration):
            ...

    if config.evaluate:
        evaluator = Evaluator(trainer_fact, dataset["test"], io_helper)
        evaluator()
        for i, (model, results) in enumerate(tqdm(evaluator), 1):  # pylint: disable=unused-variable
            ...


def cli() -> None:
    parser = HfArgumentParser([Arguments, Config, TrainingArguments])
    args, config, training_args = parser.parse_args_into_dataclasses()
    main(args, config, training_args)


if __name__ == "__main__":
    cli()
