# -*- coding: utf-8 -*-
# imports
import multiprocessing
import os

import numpy as np
import pandas as pd
import torch
from constants import (
    MODEL_UNSUPERVISED_CHECKPOINTS_DIR,
    MODEL_UNSUPERVISED_MODEL_DIR,
    MTSAMPLES_PROCESSED_PATH_DIR,
    SEED_SPLIT,
)
from utils import (
    get_device,
    load_training_args,
    load_trainer,
    load_tokenizer,
    tokenize_function,
)
from datasets import Dataset, metric
from sklearn.model_selection import train_test_split
from transformers import BertForMaskedLM
from transformers.trainer_utils import get_last_checkpoint

import wandb

wandb.init(project="nlp", entity="nlp_masterthesis")


def load_datasets(data_path: str) -> tuple[Dataset, Dataset]:

    df_mlm = pd.read_csv(data_path)
    df_mlm = df_mlm.head(20)
    # Train/Valid Split
    df_train, df_valid = train_test_split(
        df_mlm, test_size=0.15, random_state=SEED_SPLIT
    )
    # Convert to Dataset object
    dataset_train = Dataset.from_pandas(df_train[["transcription"]].dropna())
    dataset_val = Dataset.from_pandas(df_valid[["transcription"]].dropna())
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
    ModelClass = BertForMaskedLM
    bert_type = "bert-base-cased"
    model = ModelClass.from_pretrained(bert_type).to(device)
    # AutoModelForSequenceClassification.from_pretrained(
    #    MODEL_SEMI_SUPERVISED_NAME, num_labels=39
    # ).to(device)

    return model


def main():
    train_ds, val_ds = load_datasets(
        os.path.join(MTSAMPLES_PROCESSED_PATH_DIR, "mtsamples_cleaned.csv")
    )

    tokenizer = load_tokenizer()
    tokenized_train_ds = tokenize_dataset(train_ds, tokenizer)
    tokenized_val_ds = tokenize_dataset(val_ds, tokenizer)

    device = get_device()
    model = load_model(device)
    training_args = load_training_args(MODEL_UNSUPERVISED_CHECKPOINTS_DIR)
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
    trainer.save_model(MODEL_UNSUPERVISED_MODEL_DIR)
    trainer.save_state()


if __name__ == "__main__":
    main()
