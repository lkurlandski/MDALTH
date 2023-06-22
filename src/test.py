"""
"""

from datasets import load_dataset
import evaluate
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)

from learning.learner import ActiveLearner


SUBSET = 500


class MyActiveLeaner(ActiveLearner):
    def get_model(self) -> PreTrainedModel:
        return AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path, num_labels=2, id2label=id2label, label2id=label2id
        )

    def get_training_args(self) -> TrainingArguments:
        return TrainingArguments(
            output_dir=None,
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=2,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )


imdb = load_dataset("imdb")
imdb["train"] = imdb["train"].select(range(SUBSET))
imdb["test"] = imdb["test"].select(range(SUBSET))
pretrained_model_name_or_path = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)


tokenized_imdb = imdb.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

MyActiveLeaner(tokenized_imdb)()
