# -*- coding: utf-8 -*-
import os
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from constants import (
    WAV2VEC2_BATCH_SIZE_TRAIN,
    WAV2VEC2_BATCH_SIZE_EVAL,
    WAV2VEC2_MODEL,
    WAV2VEC2_MODEL_CHECKPOINTS,
    VOCAB_DIR,
    WAV2VEC2_NUM_EPOCHS,
    PROCESSED_DIR,
)
from decorators import log_function_name
from evaluate import load
from transformers import (
    Wav2Vec2ForCTC,
    Trainer,
    TrainingArguments,
    Wav2Vec2Processor,
)
from transformers.trainer_utils import get_last_checkpoint
from utils import (
    DataCollatorCTCWithPadding,
    EearlyStoppingCallbackAfterNumEpochs,
    get_device,
    load_processor_wav2vec2,
    load_datasets,
)

import wandb

wandb.init(
    name="facebook-wav2vec2",
    project="speech-to-text",
    entity="nlp_masterthesis",
)

processor = load_processor_wav2vec2(VOCAB_DIR)
cer_metric = load("cer")
wer_metric = load("wer")


def compute_metrics(pred):
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

    model = Wav2Vec2ForCTC.from_pretrained(
        WAV2VEC2_MODEL,
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
        per_device_train_batch_size=WAV2VEC2_BATCH_SIZE_TRAIN,
        per_device_eval_batch_size=WAV2VEC2_BATCH_SIZE_EVAL,
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
        callbacks=[
            EearlyStoppingCallbackAfterNumEpochs(
                start_epoch=15,
                early_stopping_patience=5,
            )
        ],
    )
    return trainer


@log_function_name
def main():
    train_ds, val_ds = load_datasets(PROCESSED_DIR)

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

    trainer.train(resume_from_checkpoint=None)
    # processor.save_pretrained(WAV2VEC2_MODEL_DIR)
    # trainer.save_model(WAV2VEC2_MODEL_DIR)
    # trainer.save_state()


if __name__ == "__main__":
    main()
