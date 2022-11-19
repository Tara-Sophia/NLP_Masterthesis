# -*- coding: utf-8 -*-

"""
Description:
    Training script for the Wav2vec2 model

Usage:
    $ python src/data/wav2vec2_training.py -s

Possible arguments:
    * -s or --save: Save the data
"""
import os

import click
import torch
import wandb
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from transformers.trainer_utils import get_last_checkpoint

from src.decorators import log_function_name
from src.stt.constants import (
    PROCESSED_DIR,
    VOCAB_DIR,
    WAV2VEC2_BATCH_SIZE_EVAL,
    WAV2VEC2_BATCH_SIZE_TRAIN,
    WAV2VEC2_MODEL,
    WAV2VEC2_MODEL_CHECKPOINTS,
    WAV2VEC2_MODEL_DIR,
    WAV2VEC2_NUM_EPOCHS,
)
from src.stt.utils import (
    DataCollatorCTCWithPadding,
    get_device,
    load_datasets,
    load_processor,
    load_trainer,
    load_training_args,
)

wandb.init(
    project="speech-to-text",
    entity="nlp_masterthesis",
    tags=["facebook-wav2vec2"],
)


@log_function_name
def load_model(
    processor: Wav2Vec2Processor, device: torch.device
) -> Wav2Vec2ForCTC:
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
    Wav2Vec2ForCTC
        Model to use
    """

    model = Wav2Vec2ForCTC.from_pretrained(
        WAV2VEC2_MODEL,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    ).to(device)

    if hasattr(model, "freeze_feature_encoder"):
        model.freeze_feature_encoder()

    return model


@click.command()
@click.option(
    "--save",
    "-s",
    help="Save dataframe",
    default=False,
    is_flag=True,
    required=False,
)
@log_function_name
def main(save: bool) -> None:
    """
    Main function

    Parameters
    ----------
    save : bool
        Flag if training should be saved
    """
    if not save:
        print("No data will be saved")

    train_ds, val_ds = load_datasets(PROCESSED_DIR)
    processor = load_processor(VOCAB_DIR)

    data_collator = DataCollatorCTCWithPadding(
        processor=processor, padding=True
    )

    device = get_device()
    model = load_model(processor, device)
    training_args = load_training_args(
        WAV2VEC2_MODEL_CHECKPOINTS,
        WAV2VEC2_BATCH_SIZE_TRAIN,
        WAV2VEC2_BATCH_SIZE_EVAL,
        WAV2VEC2_NUM_EPOCHS,
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

    if save:
        trainer.train(resume_from_checkpoint=last_checkpoint)
        processor.save_pretrained(WAV2VEC2_MODEL_DIR)
        trainer.save_model(WAV2VEC2_MODEL_DIR)
        trainer.save_state()


if __name__ == "__main__":
    main()
