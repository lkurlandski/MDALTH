"""
Run active learning experiments for classification tasks.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
import random
import sys
from typing import Optional

from datasets import load_dataset
import evaluate
from evaluate import EvaluationModule
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


@dataclass
class Arguments:

    task: str = field(metadata={"help": "Task to run."})
    dataset: str = field(metadata={"help": "Dataset to use."})
    pretrained_model_name_or_path: str = field(metadata={"help": "Pretrained model to use."})
    metric: str = field(default="accuracy", metadata={"help": "Metric to use."})
    querier: str = field(default="random", metadata={"help": "Querier to use."})
    stopper: str = field(default="null", metadata={"help": "Stopper to use."})
    subset: Optional[ProportionOrInteger] = field(
        default=None, metadata={"help": "Subset of the dataset to use."}
    )


def compute_metrics(metric: EvaluationModule, eval_pred: EvalPrediction):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)


def main(args: Arguments, config: Config, training_args: TrainingArguments) -> None:

    random.seed(training_args.seed)
    np.random.seed(training_args.seed)
    torch.manual_seed(training_args.seed)

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

    dataset = load_dataset(args.dataset)
    if args.subset is not None and args.subset > 0:
        _train_idx = range(proportion_or_integer_to_int(args.subset, len(dataset["train"])))
        _test_idx = range(proportion_or_integer_to_int(args.subset, len(dataset["test"])))
        dataset["train"] = dataset["train"].select(_train_idx)
        dataset["test"] = dataset["test"].select(_test_idx)

    config.configure(len(dataset["train"]))
    id2label = {str(i): l for i, l in enumerate(dataset["train"].features["label"].names)}

    if args.task == "text":
        task_manager = TextTaskManager(args.pretrained_model_name_or_path, id2label)
    elif args.task == "image":
        task_manager = ImageTaskManager(args.pretrained_model_name_or_path, id2label)
    elif args.task == "audio":
        task_manager = AudioTaskManager(args.pretrained_model_name_or_path, id2label)
    else:
        raise ValueError(f"Unknown task {args.task}.")

    dataset = dataset.map(
        task_manager.preprocess_function,  # TODO: this does not get hashed properly
        remove_columns=task_manager.remove_columns,
        batched=True,
    )

    output_root = Path(f"./example/output/{args.task}/{args.querier}/{args.stopper}")
    output_root.mkdir(parents=True, exist_ok=True)

    pool = Pool(dataset["train"])
    io_helper = IOHelper(output_root / config.output_root(), overwrite=True)
    callbacks = [EarlyStoppingCallback(early_stopping_patience=2)]
    metric = evaluate.load(args.metric)
    trainer_fact = TrainerFactory(
        model_init=lambda: task_manager.model_init(),
        args=training_args,
        data_collator=task_manager.data_collator,
        tokenizer=task_manager.tokenizer,
        callbacks=callbacks,
        compute_metrics=lambda eval_pred: compute_metrics(metric, eval_pred),  # TODO: use partial
    )

    print(f"{dataset=}\n", BR)
    print(f"{task_manager.data_collator=}\n", BR)
    print(f"{task_manager.tokenizer=}\n", BR)
    print(f"{callbacks=}\n", BR)
    print(f"{task_manager.model_init()=}\n", BR)
    print(f"{training_args=}\n", BR)
    print(f"{pool=}\n", BR)
    print(f"{config=}\n", BR)
    print(f"{io_helper=}\n", BR)
    print(flush=True)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

    if config.learn:
        if config.resume or config.resume_from_al_checkpoint:
            learner = Learner.load_from_disk(
                io_helper.root_path,
                config,
                trainer_fact,
                querier,
                stopper,
                iteration=config.resume_from_al_checkpoint,
            )
        else:
            learner = Learner(pool, config, io_helper, trainer_fact, querier, stopper)
            learner()
        for i, learner_state in enumerate(
            tqdm(learner), learner.iteration
        ):  # pylint: disable=unused-variable
            ...

    if config.evaluate:
        if config.resume or config.resume_from_al_checkpoint:
            raise NotImplementedError()
            evaluator = Evaluator.load_from_disk()
        else:
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
