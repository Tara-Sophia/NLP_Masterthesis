import multiprocessing
import os
from datasets.arrow_dataset import Batch

import pandas as pd
import torch
import wandb
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, BertForMaskedLM
from transformers.trainer_utils import get_last_checkpoint

from masked_language_model_training import load_model

from constants import (
    MODEL_MLM_CHECKPOINTS_DIR,
    MODEL_MLM_DIR,
    SEED_SPLIT,
)
from utils import (
    get_device,
    load_tokenizer,
    load_trainer,
    load_training_args,
    
)

from tqdm import tqdm
from tqdm.notebook import tqdm

tqdm.pandas()
# dont show warnings
import warnings

wandb.init(project="nlp", entity="nlp_masterthesis", tags=["mlm_mimic_iii"])
import torch
torch.cuda.empty_cache()

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


def tokenize_dataset(dataset: Dataset, tokenizer: AutoTokenizer) -> Dataset:
    """
    Tokenize the dataset

    Parameters
    ----------
    dataset : Dataset
        The dataset to tokenize
    tokenizer : AutoTokenizer
        The tokenizer to use

    Returns
    -------
    Dataset
        The tokenized dataset
    """
    column_names = dataset.column_names

    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=multiprocessing.cpu_count(),
        remove_columns=column_names,
        fn_kwargs={"tokenizer": tokenizer, "special_token": True},
    )
    return tokenized_datasets


def tokenize_function(
    batch: Batch, tokenizer: AutoTokenizer, special_token: bool
) -> Batch:
    """
    Tokenize the input batch

    Parameters
    ----------
    batch : Batch
        Batch to tokenize
    tokenizer : AutoTokenizer
        Tokenizer to use
    special_token : bool
        Whether to add special tokens

    Returns
    -------
    Batch
        Tokenized batch
    """
    # spcial_token = false for Text classification
    return tokenizer(
        batch["TEXT_final_cleaned"],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_special_tokens_mask=special_token,
    )

def main():
    """
    Main function
    """
    train_ds, val_ds = load_datasets(
        os.path.join("data/processed/mimic_iii/diagnoses_noteevents_cleaned.csv")
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

    trainer.train()
    trainer.save_model(MODEL_MLM_DIR)
    trainer.save_state()

    
if __name__ == "__main__":
    main()
    
