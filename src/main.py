"""
Command line interface to run active learning experiments.
"""

from dataclasses import dataclass, field
from pprint import pprint, pformat

from argparse import (
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
    HelpFormatter,
)

import datasets
import transformers
from transformers import (
    Config,
    PreTrainedModel,
    PreTrainedTokenizer,
    TrainingArguments,
    HfArgumentParser,
)


CHOICES = {
    "domain": [
        "NaturalLanguageProcessing",
        "ComputerVision",
        "Audio",
        "MultiModal",
    ],
    "task": [
        "CausalLM",
        "MaskedLM",
        "MaskGeneration",
        "Seq2SeqLM",
        "SequenceClassification",
        "MultipleChoice",
        "NextSentencePrediction",
        "TokenClassification",
        "QuestionAnswering",
        "QuestionAnswering",
    ],
}


@dataclass
class ALArguments:
    n_query: int = field(
        default=100,
        metadata={
            "help": "The number of examples to be selected at each iteration of AL.",
        },
    )


@dataclass
class HFArguments:
    domain: str = field(
        default="NaturalLanguageProcessing",
        metadata={
            "help": f"The domain of the task. One of {pformat(CHOICES['domain'])}",
        },
    )
    task: str = field(
        default="CausalLM",
        metadata={
            "help": f"The task to be performed. One of {pformat(CHOICES['task'])}",
        },
    )
    pretrained_model_name_or_path: str = field(
        default="bert-base-uncased",
        metadata={
            "help": "The pretrained model name or path",
        },
    )


@dataclass
class DatasetArguments:
    path: str
    name: str
    task: str


parser = transformers.HfArgumentParser(
    (ALArguments, HFArguments, TrainingArguments),
    formatter_class=type("Formatter", (ArgumentDefaultsHelpFormatter, HelpFormatter), {}),
)
args = parser.parse_args()

config = transformers.AutoConfig.from_pretrained(args.pretrained_model_name_or_path)

if args.task == "CausalLM":
    model = transformers.AutoModelForCausalLM.from_pretrained(args.pretrained_model_name_or_path)
    task = datasets.LanguageModeling()
elif args.task == "MaskedLM":
    model = transformers.AutoModelForMaskedLM.from_pretrained(args.pretrained_model_name_or_path)
    task = datasets.LanguageModeling()
elif args.task == "MaskGeneration":
    model = transformers.AutoModelForMaskGeneration.from_pretrained(args.pretrained_model_name_or_path)
    task = datasets.LanguageModeling()
elif args.task == "Seq2SeqLM":
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(args.pretrained_model_name_or_path)
elif args.task == "SequenceClassification":
    model = transformers.AutoModelForSequenceClassification.from_pretrained(args.pretrained_model_name_or_path)
elif args.task == "MultipleChoice":
    model = transformers.AutoModelForMultipleChoice.from_pretrained(args.pretrained_model_name_or_path)
elif args.task == "NextSentencePrediction":
    model = transformers.AutoModelForNextSentencePrediction.from_pretrained(args.pretrained_model_name_or_path)
elif args.task == "TokenClassification":
    model = transformers.AutoModelForTokenClassification.from_pretrained(args.pretrained_model_name_or_path)
elif args.task == "QuestionAnswering":
    model = transformers.AutoModelForQuestionAnswering.from_pretrained(args.pretrained_model_name_or_path)

if args.domain == "NaturalLanguageProcessing":
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)
elif args.domain == "ComputerVision":
    image_processor = transformers.AutoImageProcessor.from_pretrained(args.pretrained_model_name_or_path)
elif args.domain == "Audio":
    audio_processor = transformers.AutoFeatureExtractor.from_pretrained(args.pretrained_model_name_or_path)
elif args.domain == "MultiModal":
    auto_processor = transformers.AutoFeatureExtractor.from_pretrained(args.pretrained_model_name_or_path)
