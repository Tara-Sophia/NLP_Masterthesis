# -*- coding: utf-8 -*-
# imports
import multiprocessing
import os

import pandas as pd
import torch
from constants import (
    MODEL_MLM_CHECKPOINTS_DIR,
    MODEL_MLM_DIR,
    MTSAMPLES_PROCESSED_PATH_DIR,
    SEED_SPLIT,
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import BertForMaskedLM
from transformers.trainer_utils import get_last_checkpoint
from utils import (
    get_device,
    load_tokenizer,
    load_trainer,
    load_training_args,
    tokenize_function,
)

import wandb

wandb.init(project="nlp", entity="nlp_masterthesis", tags=["mlm"])


def load_datasets(data_path: str) -> tuple[Dataset, Dataset]:

    df_mlm = pd.read_csv(data_path)
    # Train/Valid Split
    df_train, df_valid = train_test_split(
        df_mlm, test_size=0.15, random_state=SEED_SPLIT
    )
    # Convert to Dataset object
    dataset_train = Dataset.from_pandas(
        df_train[["transcription"]].dropna()
    )
    dataset_val = Dataset.from_pandas(
        df_valid[["transcription"]].dropna()
    )
    return dataset_train, dataset_val


def tokenize_dataset(dataset: Dataset, tokenizer):
    column_names = dataset.column_names

    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=multiprocessing.cpu_count(),
        remove_columns=column_names,
        fn_kwargs={"tokenizer": tokenizer, "special_token": True},
    )
    return tokenized_datasets


def load_model(device: torch.device) -> BertForMaskedLM:
    """
    Load the model from the checkpoint directory

    Parameters
    ----------
    device : torch.device
        The device to load the model to

    Returns
    -------
    BertForMaskedLM
        The model
    """
    model = BertForMaskedLM.from_pretrained(
        MODEL_MLM_CHECKPOINTS_DIR
    ).to(device)
    # AutoModelForSequenceClassification.from_pretrained(
    #    MODEL_SEMI_SUPERVISED_NAME, num_labels=39
    # ).to(device)

    return model


def main():
    train_ds, val_ds = load_datasets(
        os.path.join(
            MTSAMPLES_PROCESSED_PATH_DIR, "mtsamples_cleaned.csv"
        )
    )

    tokenizer = load_tokenizer()
    tokenized_train_ds = tokenize_dataset(train_ds, tokenizer)
    tokenized_val_ds = tokenize_dataset(val_ds, tokenizer)

    device = get_device()
    model = load_model(device)
    training_args = load_training_args(MODEL_MLM_CHECKPOINTS_DIR)
    trainer = load_trainer(
        model,
        training_args,
        tokenized_train_ds,
        tokenized_val_ds,
        tokenizer,
        modeltype="MLM",
    )

    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is None:
        resume_from_checkpoint = None
    else:
        resume_from_checkpoint = True

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(MODEL_MLM_DIR)
    trainer.save_state()


if __name__ == "__main__":
    main()
