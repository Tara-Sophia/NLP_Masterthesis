# -*- coding: utf-8 -*-
"""
Description:
    This script is used to train the text classification model.
    The model is trained on the MTSamples dataset.
    The model is trained using the HuggingFace Trainer class.
"""
import os

import pandas as pd
import torch
import wandb
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint

from constants import MODEL_BASE_NAME, MODEL_TC_CHECKPOINTS_DIR, MODEL_TC_DIR
from utils import (
    get_device,
    load_tokenizer,
    load_trainer,
    load_training_args,
)
from text_classification_model_training import load_datasets
from datasets.arrow_dataset import Batch

# wandb.init(
#     project="nlp",
#     entity="nlp_masterthesis",
#     tags=["textclassification mimic"],
# )
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


def tokenize_dataset(dataset: Dataset, tokenizer: AutoTokenizer) -> Dataset:
    """
    Tokenize the dataset

    Parameters
    ----------
    dataset : Dataset
        Dataset to tokenize
    tokenizer : AutoTokenizer
        Tokenizer

    Returns
    -------
    Dataset
        Tokenized dataset
    """
    tokenized_datasets = dataset.map(
        tokenize_function,
        fn_kwargs={"tokenizer": tokenizer, "special_token": False},
        batched=True,
    )
    return tokenized_datasets


def map_medical_specialty_to_labels(path: str) -> pd.DataFrame:
    """
    Read the csv file
    Map the medical specialty to labels

    Parameters
    ----------
    path : str
        Path to the csv file

    Returns
    -------
    pd.DataFrame
        Dataframe with the mapped labels
    """
    print("Loading dataset")
    df = pd.read_csv(path)
    # drop rows with empty text
    # df = df.dropna(subset=["TEXT_final_cleaned"])
    df = df.dropna()
    print("Mapping medical specialty to labels")
    dict_medical_specialty = {
        value: idx for idx, value in enumerate(df.specialty.unique())
    }
    df["labels"] = df.specialty.map(dict_medical_specialty)
    return df


def load_datasets(data_path: str) -> tuple[Dataset, Dataset]:
    """
    Load the train and validation datasets

    Parameters
    ----------
    data_path : str
        Path to the dataset

    Returns
    -------
    tuple[Dataset, Dataset]
        train and validation datasets
    """
    print("in dataset")
    dataset = Dataset.from_pandas(map_medical_specialty_to_labels(data_path))
    dataset_train_test = dataset.train_test_split(test_size=0.1)
    # train dataset
    dataset_train_val = dataset_train_test["train"].train_test_split(test_size=0.1)
    dataset_train = dataset_train_val["train"]
    # validation dataset
    dataset_val = dataset_train_val["test"]

    return dataset_train, dataset_val


def clean_remove_column(tokenized_dataset: Dataset) -> Dataset:
    """
    Remove all unneded columns from the dataset

    Parameters
    ----------
    tokenized_dataset : Dataset
        Tokenized dataset

    Returns
    -------
    Dataset
        Dataset with only the needed columns
    """
    print(tokenized_dataset.column_names)
    tokenized_dataset = tokenized_dataset.remove_columns(
        ["TEXT", "specialty", "TEXT_final"]
    )
    # tokenized_dataset = tokenized_dataset.rename_column(
    #     "labels_val", "labels"
    # )
    tokenized_dataset.set_format("torch")
    return tokenized_dataset


def load_model(
    device: torch.device,
) -> AutoModelForSequenceClassification:
    """
    Load sequence classification model

    Parameters
    ----------
    device : torch.device
        Device

    Returns
    -------
    AutoModelForSequenceClassification
        Sequence classification model
    """

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_BASE_NAME, num_labels=16
    ).to(device)

    return model


def main() -> None:
    """
    Main function
    """
    # path to diagnosis dataset

    train_ds, val_ds = load_datasets(
        os.path.join(
            "data", "processed", "mimic_iii", "diagnoses_noteevents_cleaned.csv"
        )
        # "../../data/processed/mimic_iii/diagnoses_noteevents_cleaned.csv")
    )

    tokenizer = load_tokenizer()
    tokenized_train_ds = tokenize_dataset(train_ds, tokenizer)
    tokenized_val_ds = tokenize_dataset(val_ds, tokenizer)

    tokenized_train_ds = clean_remove_column(tokenized_train_ds)
    tokenized_val_ds = clean_remove_column(tokenized_val_ds)

    device = get_device()
    model = load_model(device)
    training_args = load_training_args(MODEL_TC_CHECKPOINTS_DIR)
    trainer = load_trainer(
        model,
        training_args,
        tokenized_train_ds,
        tokenized_val_ds,
        tokenizer,
        modeltype="sequence_classification",
    )


    trainer.train()
    trainer.save_model(MODEL_TC_DIR)
    trainer.save_state()


if __name__ == "__main__":
    main()
