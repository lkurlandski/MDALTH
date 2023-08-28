"""
Run active learning experiments for classification tasks.

TODO
----
    - hash issue: https://github.com/huggingface/datasets/issues/4506#issuecomment-1157417219
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import partial
import os
from pathlib import Path
import random
import sys
from typing import Optional

from datasets import load_dataset
import evaluate
import numpy as np
from transformers import (
    AutoFeatureExtractor,
    AutoImageProcessor,
    AutoModelForAudioClassification,
    AutoModelForImageClassification,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    DefaultDataCollator,
    EarlyStoppingCallback,
    HfArgumentParser,
    PreTrainedModel,
    SequenceFeatureExtractor,
    TrainingArguments,
)
from transformers.image_processing_utils import BaseImageProcessor
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
import torch
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor
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


PAD_TO_MULTIPLE_OF = 8


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


class TaskManager(ABC):
    """Provides a unified API for the various classification tasks."""

    _data_collator: DefaultDataCollator | DataCollatorWithPadding
    remove_columns = tuple()

    def __init__(self, pretrained_model_name_or_path: str, id2label: dict[str, str]) -> None:
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.id2label = id2label
        self.label2id = {v: k for k, v in id2label.items()}
        self.num_labels = len(id2label)

    @property
    def data_collator(self) -> DefaultDataCollator | DataCollatorWithPadding:
        return self._data_collator

    @property
    @abstractmethod
    def tokenizer(self) -> PreTrainedTokenizerBase | BaseImageProcessor | SequenceFeatureExtractor:
        ...

    @property
    @abstractmethod
    def AutoModelClass(self) -> type:
        ...

    @abstractmethod
    def preprocess_function(self, examples):
        ...

    def model_init(self) -> PreTrainedModel:
        return self.AutoModelClass.from_pretrained(
            self.pretrained_model_name_or_path,
            id2label=self.id2label,
            label2id=self.label2id,
            num_labels=self.num_labels,
        )


class TextTaskManager(TaskManager):
    def __init__(self, pretrained_model_name_or_path: str, id2label: dict[str, str]) -> None:
        super().__init__(pretrained_model_name_or_path, id2label)
        self._tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name_or_path)
        self._data_collator = DataCollatorWithPadding(
            tokenizer=self._tokenizer, pad_to_multiple_of=PAD_TO_MULTIPLE_OF
        )
        self.remove_columns = ("text",)

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        return self._tokenizer

    @property
    def AutoModelClass(self) -> type:
        return AutoModelForSequenceClassification

    def preprocess_function(self, examples):
        return self._tokenizer(examples["text"], truncation=True)


class ImageTaskManager(TaskManager):
    def __init__(self, pretrained_model_name_or_path: str, id2label: dict[str, str]) -> None:
        super().__init__(pretrained_model_name_or_path, id2label)
        self._image_processor = AutoImageProcessor.from_pretrained(pretrained_model_name_or_path)
        self._data_collator = DefaultDataCollator()
        normalize = Normalize(
            mean=self._image_processor.image_mean, std=self._image_processor.image_std
        )
        size = self._image_processor.size
        size = size["shortest_edge"] if "shortest_edge" in size else (size["height"], size["width"])
        resize = RandomResizedCrop(size)
        self.transforms = Compose([resize, ToTensor(), normalize])
        self.remove_columns = ("image",)

    @property
    def tokenizer(self) -> BaseImageProcessor:
        return self._image_processor

    @property
    def AutoModelClass(self) -> type:
        return AutoModelForImageClassification

    def preprocess_function(self, examples):
        examples["pixel_values"] = [
            self.transforms(img.convert("RGB")) for img in examples["image"]
        ]
        # del examples["image"]
        return examples


class AudioTaskManager(TaskManager):
    def __init__(self, pretrained_model_name_or_path: str, id2label: dict[str, str]) -> None:
        super().__init__(pretrained_model_name_or_path, id2label)
        self._feature_extractor = AutoFeatureExtractor.from_pretrained(
            pretrained_model_name_or_path
        )
        self._data_collator = DataCollatorWithPadding(
            tokenizer=self._feature_extractor, pad_to_multiple_of=PAD_TO_MULTIPLE_OF
        )
        self.remove_columns = ("audio",)

    @property
    def tokenizer(self) -> SequenceFeatureExtractor:
        return self._feature_extractor

    @property
    def AutoModelClass(self) -> type:
        return AutoModelForAudioClassification

    def preprocess_function(self, examples):
        audio_arrays = [x["array"] for x in examples["audio"]]
        inputs = self._feature_extractor(
            audio_arrays, sampling_rate=self._feature_extractor.sampling_rate
        )
        return inputs


def compute_metrics(metric, eval_pred):
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
        lambda examples: task_manager.preprocess_function(examples),  # TODO: potential hash issue?
        remove_columns=task_manager.remove_columns,
        batched=True,
    )

    output_root = Path(f"./examples/output/{args.task}/{args.querier}/{args.stopper}")
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
        for i, learner_state in enumerate(tqdm(learner), learner.iteration):  # pylint: disable=unused-variable
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
