# -*- coding: utf-8 -*-
import os

import pandas as pd
import torch
from constants import WAV2VEC2_MODEL_DIR, SAMPLING_RATE
from decorators import log_function_name
from datasets import Dataset
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    EarlyStoppingCallback,
    WhisperTokenizer,
    WhisperProcessor,
)


@log_function_name
def get_device() -> torch.device:
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


@log_function_name
def load_datasets(data_path):
    train_df = pd.read_feather(
        os.path.join(data_path, "train", "train.feather")
    )
    val_df = pd.read_feather(
        os.path.join(data_path, "val", "val.feather")
    )

    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)

    return train_ds, val_ds


# FACEBOOK WAV2VEC2


@log_function_name
def load_processor_wav2vec2(processor_path):

    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
        processor_path,
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
    )

    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=SAMPLING_RATE,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=False,
    )

    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor, tokenizer=tokenizer
    )

    return processor


@log_function_name
def load_trained_model_and_processor_wav2vec2(device):
    model = Wav2Vec2ForCTC.from_pretrained(WAV2VEC2_MODEL_DIR)
    processor = Wav2Vec2Processor.from_pretrained(WAV2VEC2_MODEL_DIR)
    model.to(device)
    return model, processor


# TRAINING


class DataCollatorCTCWithPadding:
    """
    _summary_
    """

    def __init__(
        self,
        processor: Wav2Vec2Processor,
        padding: bool | str = True,
        max_length: int | None = None,
        max_length_labels: int | None = None,
        pad_to_multiple_of: int | None = None,
        pad_to_multiple_of_labels: int | None = None,
    ):
        self.processor = processor
        self.padding = padding
        self.max_length = max_length
        self.max_length_labels = max_length_labels
        self.pad_to_multiple_of = pad_to_multiple_of
        self.pad_to_multiple_of_labels = pad_to_multiple_of_labels

    def __call__(
        self,
        features: list[dict[str, list[int] | torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [
            {"input_values": feature["input_values"]}
            for feature in features
        ]
        label_features = [
            {"input_ids": feature["labels"]} for feature in features
        ]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            max_length=self.max_length_labels,
            pad_to_multiple_of=self.pad_to_multiple_of_labels,
            return_tensors="pt",
        )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels

        return batch


class EearlyStoppingCallbackAfterNumEpochs(EarlyStoppingCallback):
    "A callback that prints a message at the beginning of training"

    def __init__(self, start_epoch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_epoch = start_epoch

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if state.epoch > self.start_epoch:
            super().on_evaluate(
                args, state, control, metrics, **kwargs
            )
