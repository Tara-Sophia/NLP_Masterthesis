import multiprocessing
import os

import pandas as pd
import torch
import wandb
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, BertForMaskedLM
from transformers.trainer_utils import get_last_checkpoint

from src.nlp.masked_language_model_training import tokenize_dataset, load_model

from src.nlp.constants import (
    MODEL_MLM_CHECKPOINTS_DIR,
    MODEL_MLM_DIR,
    MTSAMPLES_PROCESSED_PATH_DIR,
    SEED_SPLIT,
)
from src.nlp.utils import (
    get_device,
    load_tokenizer,
    load_trainer,
    load_training_args,
    tokenize_function,
)

wandb.init(project="nlp", entity="nlp_masterthesis", tags=["mlm_mimic_iii"])


def load_datasets(data_path: str) -> tuple[Dataset, Dataset]:
    """
    Load the datasets

    Parameters
    ----------
    data_path : str
        Path to the dataset

    Returns
    -------
    tuple[Dataset, Dataset]
        The train and validation datasets
    """

    df_mlm = pd.read_csv(data_path)
    # Train/Valid Split
    df_train, df_valid = train_test_split(
        df_mlm, test_size=0.15, random_state=SEED_SPLIT
    )
    # Convert to Dataset object
    dataset_train = Dataset.from_pandas(df_train[["TEXT_final_cleaned"]].dropna())
    dataset_val = Dataset.from_pandas(df_valid[["TEXT_final_cleaned"]].dropna())
    return dataset_train, dataset_val


def main():
    """
    Main function
    """
    train_ds, val_ds = load_datasets(
        os.path.join("../data/processed/mimic_iii/diagnoses_noteevents_cleaned.csv")
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
