# -*- coding: utf-8 -*-
import os
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from constants import (
    WAV2VEC2_BATCH_SIZE,
    WAV2VEC2_MODEL,
    WAV2VEC2_MODEL_CHECKPOINTS,
    WAV2VEC2_MODEL_DIR,
    WAV2VEC2_NUM_EPOCHS,
    WAV2VEC2_PROCESSED_DIR,
)
from datasets import load_from_disk, Dataset
from decorators import log_function_name
from evaluate import load
from transformers import (
    AutoModelForCTC,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    Wav2Vec2Processor,
)
from transformers.trainer_utils import get_last_checkpoint
from utils import get_device, load_processor_wav2vec2, load_datasets

import wandb

wandb.init(
    name="facebook-wav2vec2",
    project="speech-to-text",
    entity="nlp_masterthesis",
)


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
        features: List[Dict[str, Union[List[int], torch.Tensor]]],
    ) -> Dict[str, torch.Tensor]:
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


def compute_metrics(pred):

    processor = load_processor_wav2vec2(WAV2VEC2_MODEL_DIR)
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
def load_model(processor, device):

    model = AutoModelForCTC.from_pretrained(
        WAV2VEC2_MODEL,
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.1,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    ).to(device)

    if hasattr(model, "freeze_feature_encoder"):
        model.freeze_feature_encoder()

    return model


@log_function_name
def load_training_args(output_dir):
    training_args = TrainingArguments(
        output_dir=output_dir,
        group_by_length=True,
        per_device_train_batch_size=WAV2VEC2_BATCH_SIZE,
        gradient_accumulation_steps=2,
        num_train_epochs=WAV2VEC2_NUM_EPOCHS,
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
    )
    return trainer


@log_function_name
def main():
    train_ds, val_ds = load_datasets(WAV2VEC2_PROCESSED_DIR)

    processor = load_processor_wav2vec2(WAV2VEC2_MODEL_DIR)

    data_collator = DataCollatorCTCWithPadding(
        processor=processor, padding=True
    )

    device = get_device()
    model = load_model(processor, device)
    training_args = load_training_args(WAV2VEC2_MODEL_CHECKPOINTS)
    trainer = load_trainer(
        model,
        data_collator,
        training_args,
        train_ds,
        val_ds,
        processor,
    )

    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(
            training_args.output_dir
        )
    else:
        last_checkpoint = None

    trainer.train(resume_from_checkpoint=last_checkpoint)

    trainer.save_model(WAV2VEC2_MODEL_DIR)
    trainer.save_state()


if __name__ == "__main__":
    main()
