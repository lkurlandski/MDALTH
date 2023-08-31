"""
Provide a unified API to run text, image, and audio classification experiments.
"""


from abc import ABC, abstractmethod
import sys

from transformers import (
    AutoFeatureExtractor,
    AutoImageProcessor,
    AutoModelForAudioClassification,
    AutoModelForImageClassification,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    DefaultDataCollator,
    PreTrainedModel,
    SequenceFeatureExtractor,
)
from transformers.image_processing_utils import BaseImageProcessor
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor

sys.path.insert(0, ".")  # pylint: disable=wrong-import-position


PAD_TO_MULTIPLE_OF = 8


class TaskManager(ABC):

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
