# -*- coding: utf-8 -*-
import os
from typing import Dict, List, Optional, Union

import pandas as pd
import numpy as np
import torch
from constants import (
    WAV2VEC2_PROCESSED_DIR,
    WHISPER_BATCH_SIZE,
    WHISPER_MODEL,
    WHISPER_MODEL_CHECKPOINTS,
    WHISPER_MODEL_DIR,
    WHISPER_NUM_EPOCHS,
)
from datasets import Dataset
from decorators import log_function_name
from evaluate import load
from transformers import (
    AutoModelForSpeechSeq2Seq,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    Seq2SeqTrainer,
)
from transformers.trainer_utils import get_last_checkpoint
from utils import get_device, load_tokenizer_whisper, load_datasets

import wandb

wandb.init(
    name="openai-whisper",
    project="speech-to-text",
    entity="nlp_masterthesis",
)


class WhisperDataCollatorWithPadding:
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


class WhisperTrainer(Seq2SeqTrainer):
    def _save(
        self, output_dir: Optional[str] = None, state_dict=None
    ):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = (
            output_dir
            if output_dir is not None
            else self.args.output_dir
        )
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        self.model.save_to(
            save_path=os.path.join(
                output_dir,
                model_args.model_name_or_path + ".whisper",
            )
        )
        # Good practice: save your training arguments together with the trained model
        torch.save(
            self.args,
            os.path.join(output_dir, "training_args.bin"),
        )


def compute_metrics(pred):

    tokenizer = 2  # load_processor_wav2vec2(WAV2VEC2_MODEL_DIR)
    cer_metric = load("cer")
    wer_metric = load("wer")

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
def get_data_collator(tokenizer):
    # Define data collator
    eos = tokenizer.eos_token_id
    t_stamp = tokenizer("<|notimestamps|>").input_ids[0]
    whisper_data_collator = WhisperDataCollatorWithPadding(
        eos_token_id=eos, time_stamp_token_id=t_stamp
    )
    return whisper_data_collator


@log_function_name
def load_model(tokenizer, device):

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        WHISPER_MODEL
    ).to(device)

    if hasattr(model, "freeze_encoder"):
        model.freeze_encoder()

    # make sure model uses 50257 as BOS
    bos = tokenizer("<|startoftranscript|>").input_ids[0]
    model.config.decoder_start_token_id = bos

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
    model, data_collator, training_args, train_ds, val_ds
):
    trainer = WhisperTrainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
    )

    return trainer


@log_function_name
def main():
    train_ds, val_ds = load_datasets(WAV2VEC2_PROCESSED_DIR)

    tokenizer = load_tokenizer_whisper()

    data_collator = get_data_collator(tokenizer)

    device = get_device()
    # model = load_model(tokenizer, device)
    # training_args = load_training_args(WHISPER_MODEL_CHECKPOINTS)
    # trainer = load_trainer(
    #     model,
    #     data_collator,
    #     training_args,
    #     train_ds,
    #     val_ds,
    # )

    # last_checkpoint = get_last_checkpoint(training_args.output_dir)
    # if last_checkpoint is None:
    #     resume_from_checkpoint = None
    # else:
    #     resume_from_checkpoint = True

    # trainer.train()

    # trainer.save_model(WHISPER_MODEL_DIR)
    # trainer.save_state()


if __name__ == "__main__":
    main()
    # print("Done")
