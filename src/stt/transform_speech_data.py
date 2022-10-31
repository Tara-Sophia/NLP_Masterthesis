# -*- coding: utf-8 -*-
import json
import os
import re
import shutil

import numpy as np
import pandas as pd
from constants import (
    CHARS_TO_IGNORE_REGEX,
    DATA_PATH_WAV,
    MODEL_DIR,
    SAMPLING_RATE,
    VOCAB_PATH,
)
from datasets import Audio, Dataset
from decorators import log_function_name
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
)
from unidecode import unidecode


def remove_special_characters(batch, train=True):
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


def transform_dataset(batch, processor):
    audio = batch["audio"]
    # batched output is "un-batched"
    batch["input_values"] = processor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_values[0]
    batch["input_length"] = len(batch["input_values"])
    batch["labels"] = processor(text=batch["sentence"]).input_ids
    return audio


@log_function_name
def create_vocab(folder_path, train_ds, val_ds, test_ds):
    def extract_all_chars(batch):
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
    vocab_dict = {v: k for k, v in enumerate(vocab_list)}

    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]

    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    os.makedirs(folder_path, exist_ok=True)

    with open(
        os.path.join(folder_path, "vocab.json"), "w"
    ) as vocab_file:
        json.dump(vocab_dict, vocab_file)


@log_function_name
def preprocess_data(custom_train, custom_val, custom_test):
    custom_train = custom_train.cast_column(
        "audio", Audio(sampling_rate=SAMPLING_RATE)
    )
    custom_val = custom_val.cast_column(
        "audio", Audio(sampling_rate=SAMPLING_RATE)
    )
    custom_test = custom_test.cast_column(
        "audio", Audio(sampling_rate=SAMPLING_RATE)
    )

    custom_train = custom_train.map(remove_special_characters)

    custom_val = custom_val.map(remove_special_characters)

    custom_test = custom_test.map(
        remove_special_characters, fn_kwargs={"train": False}
    )

    return custom_train, custom_val, custom_test


@log_function_name
def load_processor(processor_path):

    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
        processor_path,
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
    )

    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=False,
    )

    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor, tokenizer=tokenizer
    )

    return processor


@log_function_name
def resample_data(train_ds, val_ds, test_ds):
    processor = load_processor(VOCAB_PATH)
    train_ds = train_ds.map(
        transform_dataset,
        fn_kwargs={"processor": processor},
        remove_columns=train_ds.column_names,
        num_proc=4,
    )
    val_ds = val_ds.map(
        transform_dataset,
        fn_kwargs={"processor": processor},
        remove_columns=val_ds.column_names,
        num_proc=4,
    )
    test_ds = test_ds.map(
        transform_dataset,
        fn_kwargs={"processor": processor},
        remove_columns=test_ds.column_names,
        num_proc=4,
    )

    return train_ds, val_ds, test_ds, processor


@log_function_name
def recreate_folder(folder_path):
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path, exist_ok=True)


@log_function_name
def save_datasets(train_ds, val_ds, test_ds):
    train_processed_dir = os.path.join(
        "data", "interim", "stt", "train"
    )
    val_processed_dir = os.path.join("data", "interim", "stt", "val")
    test_processed_dir = os.path.join(
        "data", "interim", "stt", "test"
    )

    recreate_folder(train_processed_dir)
    recreate_folder(val_processed_dir)
    recreate_folder(test_processed_dir)

    train_ds.save_to_disk(train_processed_dir)
    val_ds.save_to_disk(val_processed_dir)
    test_ds.save_to_disk(test_processed_dir)


def main():

    train_ds = Dataset.from_pandas(
        pd.read_csv(os.path.join(DATA_PATH_WAV, "train.csv"))
    )
    val_ds = Dataset.from_pandas(
        pd.read_csv(os.path.join(DATA_PATH_WAV, "val.csv"))
    )
    test_ds = Dataset.from_pandas(
        pd.read_csv(os.path.join(DATA_PATH_WAV, "test.csv"))
    )

    train_ds, val_ds, test_ds = preprocess_data(
        train_ds, val_ds, test_ds
    )

    create_vocab(
        folder_path=VOCAB_PATH,
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
    )

    train_ds, val_ds, test_ds, processor = resample_data(
        train_ds, val_ds, test_ds
    )

    save_datasets(train_ds, val_ds, test_ds)
    processor.save_pretrained(MODEL_DIR)


if __name__ == "__main__":
    main()
