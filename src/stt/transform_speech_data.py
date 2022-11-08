# -*- coding: utf-8 -*-
"""
Description:
    Transform the speech data to be used by the model

Usage:
    $ python src/data/transform_speech_data.py
"""
import json
import os
import sys
import re
import shutil
from typing import Optional, Union

import pandas as pd
from constants import (
    CHARS_TO_IGNORE_REGEX,
    NUM_PROC,
    PROCESSED_DIR,
    RAW_DATA_DIR,
    SAMPLING_RATE,
    VOCAB_DIR,
    SRC_DIR,
)
from datasets import Audio, Dataset
from datasets.arrow_dataset import Batch, Example
from transformers import Wav2Vec2Processor
from unidecode import unidecode
from utils import load_processor

sys.path.insert(0, SRC_DIR)
from decorators import log_function_name


def remove_special_characters(
    batch: Example, train: Optional[bool] = True
) -> Example:
    """
    Remove special characters from the dataset

    Parameters
    ----------
    batch : Example
        Batch of data
    train : Optional[bool], optional
        Checks if batch belongs to train dataset, by default True

    Returns
    -------
    Example
        Batch with special characters removed
    """
    batch["sentence"] = (
        re.sub(
            CHARS_TO_IGNORE_REGEX, "", unidecode(batch["sentence"])
        )
        .lower()
        .strip()
    )
    if train:
        batch["sentence"] += " "

    return batch


def transform_dataset(
    batch: Example, processor: Wav2Vec2Processor
) -> Example:
    """
    Transform dataset to be used by the model

    Parameters
    ----------
    batch : Example
        Batch of data
    processor : Wav2Vec2Processor
        Processor to use

    Returns
    -------
    Example
        Batch with transformed data
    """
    audio = batch["audio"]
    batch["input_values"] = processor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_values[0]
    batch["labels"] = processor(text=batch["sentence"]).input_ids
    return batch


@log_function_name
def create_vocab(
    folder_path: str,
    train_ds: Dataset,
    val_ds: Dataset,
    test_ds: Dataset,
) -> None:
    """
    Create vocab file from datasets

    Parameters
    ----------
    folder_path : str
        Path to the folder where the vocab file will be saved
    train_ds : Dataset
        Train dataset
    val_ds : Dataset
        Validation dataset
    test_ds : Dataset
        Test dataset
    """

    def extract_all_chars(
        batch: Batch,
    ) -> dict[str, list[Union[list[str], str]]]:
        """
        Extract all characters from a batch

        Parameters
        ----------
        batch : Batch
            Batch to extract characters from

        Returns
        -------
        dict[str, list[Union[list[str], str]]]
            Dictionary containing all characters from the batch
        """
        all_text = " ".join(batch["sentence"])
        vocab = list(set(all_text))
        return {"vocab": [vocab], "all_text": [all_text]}

    vocab_train = train_ds.map(
        extract_all_chars,
        batched=True,
        batch_size=-1,
        keep_in_memory=True,
        remove_columns=train_ds.column_names,
    )
    vocab_val = val_ds.map(
        extract_all_chars,
        batched=True,
        batch_size=-1,
        keep_in_memory=True,
        remove_columns=val_ds.column_names,
    )
    vocab_test = test_ds.map(
        extract_all_chars,
        batched=True,
        batch_size=-1,
        keep_in_memory=True,
        remove_columns=test_ds.column_names,
    )

    vocab_list = list(
        set(vocab_train["vocab"][0])
        | set(vocab_val["vocab"][0])
        | set(vocab_test["vocab"][0])
    )
    vocab_list.sort()
    vocab_dict = {v: k for k, v in enumerate(vocab_list)}

    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]

    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    recreate_folder(folder_path)

    with open(
        os.path.join(folder_path, "vocab.json"), "w"
    ) as vocab_file:
        json.dump(vocab_dict, vocab_file)


@log_function_name
def preprocess_data(
    train_ds: Dataset, val_ds: Dataset, test_ds: Dataset
) -> tuple[Dataset, Dataset, Dataset]:
    """
    Preprocess the data.
    Remove special characters and resample the audio to 16kHz.

    Parameters
    ----------
    train_ds : Dataset
        Training dataset
    val_ds : Dataset
        Validation dataset
    test_ds : Dataset
        Test dataset

    Returns
    -------
    tuple[Dataset, Dataset, Dataset]
        Preprocessed training, validation and test datasets
    """
    train_ds = train_ds.cast_column(
        "audio", Audio(sampling_rate=SAMPLING_RATE)
    )
    val_ds = val_ds.cast_column(
        "audio", Audio(sampling_rate=SAMPLING_RATE)
    )
    test_ds = test_ds.cast_column(
        "audio", Audio(sampling_rate=SAMPLING_RATE)
    )

    train_ds = train_ds.map(remove_special_characters)

    val_ds = val_ds.map(remove_special_characters)

    test_ds = test_ds.map(
        remove_special_characters, fn_kwargs={"train": False}
    )

    return train_ds, val_ds, test_ds


@log_function_name
def create_labels_and_input_values(
    train_ds: Dataset, val_ds: Dataset, test_ds: Dataset
) -> tuple[Dataset, Dataset, Dataset, Wav2Vec2Processor]:
    """
    Create labels and input values for the model

    Parameters
    ----------
    train_ds : Dataset
        Training dataset
    val_ds : Dataset
        Validation dataset
    test_ds : Dataset
        Test dataset

    Returns
    -------
    tuple[Dataset, Dataset, Dataset, Wav2Vec2Processor]
        Training, validation and test datasets with labels and input values
    """
    processor = load_processor(VOCAB_DIR)
    train_ds = train_ds.map(
        transform_dataset,
        fn_kwargs={"processor": processor},
        remove_columns=train_ds.column_names,
        num_proc=NUM_PROC,
    )
    val_ds = val_ds.map(
        transform_dataset,
        fn_kwargs={"processor": processor},
        remove_columns=val_ds.column_names,
        num_proc=NUM_PROC,
    )
    test_ds = test_ds.map(
        transform_dataset,
        fn_kwargs={"processor": processor},
        remove_columns=test_ds.column_names,
        num_proc=NUM_PROC,
    )

    return train_ds, val_ds, test_ds, processor


@log_function_name
def recreate_folder(folder_path: str):
    """
    Recreates a folder if it exists

    Parameters
    ----------
    folder_path : str
        Path to the folder to be recreated
    """
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path, exist_ok=True)


@log_function_name
def save_datasets(
    train_ds: Dataset, val_ds: Dataset, test_ds: Dataset
) -> None:
    """
    Save the datasets to the processed folder

    Parameters
    ----------
    train_ds : Dataset
        Training dataset
    val_ds : Dataset
        Validation dataset
    test_ds : Dataset
        Test dataset
    """
    recreate_folder(PROCESSED_DIR)

    train_ds.to_pandas().to_feather(
        os.path.join(PROCESSED_DIR, "train.feather")
    )
    val_ds.to_pandas().to_feather(
        os.path.join(PROCESSED_DIR, "val.feather")
    )
    test_ds.to_pandas().to_feather(
        os.path.join(PROCESSED_DIR, "test.feather")
    )


@log_function_name
def main():
    """
    Main function
    """
    train_ds = Dataset.from_pandas(
        pd.read_csv(os.path.join(RAW_DATA_DIR, "train.csv"))
    )
    val_ds = Dataset.from_pandas(
        pd.read_csv(os.path.join(RAW_DATA_DIR, "val.csv"))
    )
    test_ds = Dataset.from_pandas(
        pd.read_csv(os.path.join(RAW_DATA_DIR, "test.csv"))
    )

    train_ds, val_ds, test_ds = preprocess_data(
        train_ds, val_ds, test_ds
    )

    create_vocab(
        folder_path=VOCAB_DIR,
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
    )

    (
        train_ds,
        val_ds,
        test_ds,
        processor,
    ) = create_labels_and_input_values(train_ds, val_ds, test_ds)

    save_datasets(train_ds, val_ds, test_ds)
    processor.save_pretrained(VOCAB_DIR)


if __name__ == "__main__":
    main()
