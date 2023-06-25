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
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from src.learning.hf import ActiveLearner, IOHelper
from src.pool import Pool
from src.querying.queriers import RandomQuerier
from src.querying.hf import RandomQuerierWrapper


SUBSET = 500


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


class MyActiveLeaner(ActiveLearner):
    def get_model(self):
        return AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path,
            num_labels=2,
            id2label=id2label,
            label2id=label2id,
        )

    def get_data_collator(self):
        return data_collator

    def get_tokenizer(self):
        return tokenizer

    def compute_metrics(self):
        return compute_metrics


# MyActiveLeaner(
#     tokenized_imdb,
#     Pool(len(tokenized_imdb["train"])),
#     io_args=IOHelper(overwrite=True),
# )()

training_args = TrainingArguments(
    output_dir="my_awesome_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
    ),
    args=training_args,
    train_dataset=tokenized_imdb["train"],
    eval_dataset=tokenized_imdb["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
