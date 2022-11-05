# -*- coding: utf-8 -*-
import os

from constants import (
    PROCESSED_DIR,
    VOCAB_DIR,
    WAV2VEC2_BATCH_SIZE_EVAL,
    WAV2VEC2_BATCH_SIZE_TRAIN,
    WAV2VEC2_MODEL,
    WAV2VEC2_MODEL_CHECKPOINTS,
    WAV2VEC2_MODEL_DIR,
    WAV2VEC2_NUM_EPOCHS,
)
from decorators import log_function_name
from transformers import Wav2Vec2ForCTC
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
    tags=["facebook-wav2vec2"],
)


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
def main():
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

    trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model(WAV2VEC2_MODEL_DIR)
    trainer.save_state()


if __name__ == "__main__":
    main()
