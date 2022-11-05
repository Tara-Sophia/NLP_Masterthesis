# -*- coding: utf-8 -*-
"""
Description:
    Training script for the Hubert model

Usage:
    $ python src/data/hubert_training.py
"""
import os

import torch
from constants import (
    HUBERT_BATCH_SIZE_EVAL,
    HUBERT_BATCH_SIZE_TRAIN,
    HUBERT_MODEL,
    HUBERT_MODEL_CHECKPOINTS,
    HUBERT_MODEL_DIR,
    HUBERT_NUM_EPOCHS,
    PROCESSED_DIR,
    VOCAB_DIR,
)
from decorators import log_function_name
from transformers import HubertForCTC, Wav2Vec2Processor
from transformers.trainer_utils import get_last_checkpoint
from utils import (
    DataCollatorCTCWithPadding,
    get_device,
    load_datasets,
    load_processor,
    load_trainer,
    load_training_args,
)

import wandb

wandb.init(
    project="speech-to-text",
    entity="nlp_masterthesis",
    tags=["facebook-hubert"],
)


@log_function_name
def load_model(
    processor: Wav2Vec2Processor, device: torch.device
) -> HubertForCTC:
    """
    Load the model

    Parameters
    ----------
    processor : Wav2Vec2Processor
        Processor to use
    device : torch.device
        Torch device

    Returns
    -------
    HubertForCTC
        Model to use
    """

    model = HubertForCTC.from_pretrained(
        HUBERT_MODEL,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    ).to(device)

    if hasattr(model, "freeze_feature_encoder"):
        model.freeze_feature_encoder()

    return model


@log_function_name
def main():
    """
    Main function
    """
    train_ds, val_ds = load_datasets(PROCESSED_DIR)
    processor = load_processor(VOCAB_DIR)

    data_collator = DataCollatorCTCWithPadding(
        processor=processor, padding=True
    )

    device = get_device()
    model = load_model(processor, device)
    training_args = load_training_args(
        HUBERT_MODEL_CHECKPOINTS,
        HUBERT_BATCH_SIZE_TRAIN,
        HUBERT_BATCH_SIZE_EVAL,
        HUBERT_NUM_EPOCHS,
    )
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
    trainer.save_model(HUBERT_MODEL_DIR)
    trainer.save_state()


if __name__ == "__main__":
    main()
