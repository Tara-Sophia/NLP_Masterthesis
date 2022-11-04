# -*- coding: utf-8 -*-
import os
from typing import Optional, Union

import pandas as pd
import numpy as np
import torch
from constants import WAV2VEC2_MODEL_DIR, SAMPLING_RATE, VOCAB_DIR
from decorators import log_function_name
from datasets import Dataset
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from evaluate import load


@log_function_name
def get_device() -> torch.device:
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


@log_function_name
def load_datasets(data_path):
    train_df = pd.read_feather(
        os.path.join(data_path, "train.feather")
    )
    val_df = pd.read_feather(os.path.join(data_path, "val.feather"))

    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)

    return train_ds, val_ds


@log_function_name
def load_processor(processor_path):

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


# FACEBOOK WAV2VEC2
@log_function_name
def load_trained_model_and_processor_wav2vec2(device):
    model = Wav2Vec2ForCTC.from_pretrained(WAV2VEC2_MODEL_DIR)
    processor = Wav2Vec2Processor.from_pretrained(VOCAB_DIR)
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
        padding: Union[bool, str] = True,
        max_length: Optional[int] = None,
        max_length_labels: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        pad_to_multiple_of_labels: Optional[int] = None,
    ):
        self.processor = processor
        self.padding = padding
        self.max_length = max_length
        self.max_length_labels = max_length_labels
        self.pad_to_multiple_of = pad_to_multiple_of
        self.pad_to_multiple_of_labels = pad_to_multiple_of_labels

    def __call__(
        self,
        features: list[dict[str, Union[list[int], torch.Tensor]]],
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


def compute_metrics(pred):
    processor = load_processor(VOCAB_DIR)
    cer_metric = load("cer")
    wer_metric = load("wer")

    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[
        pred.label_ids == -100
    ] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(
        pred.label_ids, group_tokens=False
    )

    cer = cer_metric.compute(
        predictions=pred_str, references=label_str
    )
    wer = wer_metric.compute(
        predictions=pred_str, references=label_str
    )

    return {
        "cer": cer,
        "wer": wer,
        "pred_str": pred_str[0],
        "label_str": label_str[0],
    }


@log_function_name
def load_training_args(
    output_dir, batch_size_train, batch_size_val, num_epochs
):
    training_args = TrainingArguments(
        output_dir=output_dir,
        group_by_length=True,
        per_device_train_batch_size=batch_size_train,
        per_device_eval_batch_size=batch_size_val,
        gradient_accumulation_steps=2,
        num_train_epochs=num_epochs,
        gradient_checkpointing=True,
        fp16=True,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=1e-4,
        weight_decay=0.005,
        warmup_steps=1000,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        report_to="wandb",
    )
    return training_args


@log_function_name
def load_trainer(
    model, data_collator, training_args, train_ds, val_ds, processor
):

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=processor.feature_extractor,
        callbacks=[
            EearlyStoppingCallbackAfterNumEpochs(
                start_epoch=15,
                early_stopping_patience=5,
            )
        ],
    )
    return trainer
