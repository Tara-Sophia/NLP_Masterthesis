# -*- coding: utf-8 -*-
import os
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from constants import (
    PROCESSED_DATA_DIR,
    WHISPER_BATCH_SIZE,
    WHISPER_MODEL,
    WHISPER_MODEL_CHECKPOINTS,
    WHISPER_MODEL_DIR,
    WHISPER_NUM_EPOCHS,
)
from datasets.load import load_from_disk
from decorators import log_function_name
from evaluate import load
from transformers import (
    AutoModelForCTC,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint
from utils import get_device, load_processor_whisper

import wandb

wandb.init(
    name="openai-whisper",
    project="speech-to-text",
    entity="nlp_masterthesis",
)


class WhisperDataCollatorWhithPadding:
    """
    Data collator that dynamically pads the audio inputs received. An EOS token is appended to the labels sequences.
    They are then dynamically padded to max length.
    Args:
        eos_token_id (`int`)
            The end-of-sentence token for the Whisper tokenizer. Ensure to set for sequences to terminate before
            generation max length.
    """

    def __init__(self, eos_token_id: int, time_stamp_token_id: int):
        self.eos_token_id = eos_token_id
        self.time_stamp_token_id = time_stamp_token_id

    def __call__(
        self,
        features: List[Dict[str, Union[List[int], torch.Tensor]]],
    ) -> Dict[str, torch.Tensor]:
        """
        Since Whisper models don't have a HF processor defined (feature extractor + tokenizer), we'll pad by hand...
        """
        # split inputs and labels since they have to be of different lengths
        # and need different padding methods
        input_ids = [feature["input_ids"] for feature in features]
        labels = [feature["labels"] for feature in features]

        # first, pad the audio inputs to max_len
        input_ids = torch.concat(
            [
                to_pad_to_mel(input_val)[None, :]
                for input_val in input_ids
            ]
        )

        # next, append the eos token to our sequence of labels
        labels = [lab + [self.eos_token_id] for lab in labels]
        # finally, pad the target labels to max_len
        label_lengths = [len(lab) for lab in labels]
        max_label_len = max(label_lengths)
        labels = [
            np.pad(
                lab,
                (0, max_label_len - lab_len),
                "constant",
                constant_values=-100,
            )
            for lab, lab_len in zip(labels, label_lengths)
        ]

        batch = {"labels": labels}
        batch = {
            k: torch.tensor(np.array(v), requires_grad=False)
            for k, v in batch.items()
        }

        batch["input_ids"] = input_ids

        return batch


@log_function_name
def load_datasets(data_path):

    train_ds = load_from_disk(os.path.join(data_path, "train"))
    val_ds = load_from_disk(os.path.join(data_path, "val"))

    return train_ds, val_ds


def compute_metrics(pred):
    pred_ids = pred.predictions
    pred.label_ids[pred.label_ids == -100] = tokenizer.eos_token_id

    pred_str = tokenizer.batch_decode(
        pred_ids, skip_special_tokens=True
    )
    pred_str = [x.lstrip().strip() for x in pred_str]

    # we do not want to group tokens when computing the metrics
    label_str = tokenizer.batch_decode(
        pred.label_ids, skip_special_tokens=True
    )

    wer = metric_wer.compute(
        predictions=pred_str, references=label_str
    )
    cer = metric_cer.compute(
        predictions=pred_str, references=label_str
    )

    return {
        "wer": wer,
        "cer": cer,
        "pred_str": pred_str,
        "label_str": label_str,
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
        callbacks=[EarlyStoppingCallback()],
    )
    return trainer


@log_function_name
def main():
    train_ds, val_ds = load_datasets(PROCESSED_DATA_DIR)

    processor = load_processor_whisper(WHISPER_MODEL_DIR)

    # data_collator = DataCollatorCTCWithPadding(
    #     processor=processor, padding=True
    # )

    device = get_device()
    model = load_model(processor, device)
    # training_args = load_training_args(WAV2VEC2_MODEL_CHECKPOINTS)
    # trainer = load_trainer(
    #     model,
    #     data_collator,
    #     training_args,
    #     train_ds,
    #     val_ds,
    #     processor,
    # )

    # last_checkpoint = get_last_checkpoint(training_args.output_dir)
    # if last_checkpoint is None:
    #     resume_from_checkpoint = None
    # else:
    #     resume_from_checkpoint = True

    # trainer.train()

    # trainer.save_model(WAV2VEC2_MODEL_DIR)
    # trainer.save_state()


if __name__ == "__main__":
    main()
    # print("Done")
