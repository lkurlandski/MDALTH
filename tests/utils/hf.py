import sys
import tempfile
from typing import Optional

sys.path.insert(0, ".")

from datasets import load_dataset, Dataset, DatasetDict
import evaluate
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def preprocess_function(examples):
    return TOKENIZER(examples["text"], truncation=True)


def get_dataset() -> DatasetDict:
    dataset = load_dataset("imdb")
    dataset["train"] = dataset["train"].select(range(SUBSET))
    dataset["test"] = dataset["test"].select(range(SUBSET))
    return dataset.map(preprocess_function, batched=True)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return ACCURACY.compute(predictions=predictions, references=labels)


def get_trainer(train_dataset: Optional[Dataset], eval_dataset: Optional[Dataset]) -> Trainer:
    trainer = Trainer(
        model=AutoModelForSequenceClassification.from_pretrained(
            PRETRAINED_MODEL_NAME,
            num_labels=2,
            id2label={0: "NEGATIVE", 1: "POSITIVE"},
            label2id={"NEGATIVE": 0, "POSITIVE": 1},
        ),
        args=TrainingArguments(
            output_dir=tempfile.mkdtemp(),
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=1,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=TOKENIZER,
        data_collator=DATA_COLLATOR,
        compute_metrics=compute_metrics,
    )

    return trainer


SUBSET = 1000
PRETRAINED_MODEL_NAME = "distilbert-base-uncased"
TOKENIZER = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
DATASET = get_dataset()
DATA_COLLATOR = DataCollatorWithPadding(tokenizer=TOKENIZER)
ACCURACY = evaluate.load("accuracy")
