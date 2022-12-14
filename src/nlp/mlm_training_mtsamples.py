# -*- coding: utf-8 -*-
"""
Description:
    This script is used for training the MLM model.
    It is based on the  HuggingFace Trainer class.
    The model is trained on the mtsamples dataset.
"""
# imports
import multiprocessing
import os

import pandas as pd
import torch
import wandb
from constants import (
    MODEL_BASE_NAME,
    MODEL_MLM_MT_CHECKPOINTS_DIR,
    MODEL_MLM_MT_DIR,
    MOST_COMMON_WORDS_FILTERED,
    MTSAMPLES_PROCESSED_PATH_DIR,
    SEED_SPLIT,
)
from datasets import Dataset, load_metric
from sklearn.model_selection import train_test_split

# import trainingarguments
from transformers import AutoTokenizer, BertForMaskedLM, Trainer, TrainingArguments
from transformers.data.data_collator import DataCollatorForLanguageModeling
from src.nlp.utils import (  # load_trainer,
    get_device,
    load_tokenizer,
    load_training_args,
    tokenize_function,
)

from transformers import EvalPrediction

wandb.init(project="nlp", entity="nlp_masterthesis", tags=["mlm"])
# first remove MOST_COMMON_WORDS_FILTERED
def remove_most_common(df: pd.DataFrame) -> pd.DataFrame:

    """
    Remove most common words from text
    """
    for word in MOST_COMMON_WORDS_FILTERED:
        df["transcription"] = df["transcription"].str.replace(word, "")
    return df


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
    df_mlm = df_mlm.head(20)
    df_mlm = df_mlm.dropna()
    # remove most common words
    df_mlm = remove_most_common(df_mlm)
    # Train/Valid Split
    df_train, df_valid = train_test_split(
        df_mlm, test_size=0.15, random_state=SEED_SPLIT
    )
    # Convert to Dataset object
    dataset_train = Dataset.from_pandas(df_train[["transcription"]].dropna())
    dataset_val = Dataset.from_pandas(df_valid[["transcription"]].dropna())
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
        num_proc=multiprocessing.cpu_count() - 1,
        remove_columns=column_names,
        fn_kwargs={"tokenizer": tokenizer, "special_token": True},
    )
    return tokenized_datasets


def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    """
    Compute the accuracy of the model for the evaluation dataset

    Parameters
    ----------
    eval_pred : EvalPrediction
        Prediction for evaluation dataset
    modeltype : str
        Masked Language Model or Sequence Classification

    Returns
    -------
    dict[str, float]
        Accuracy score
    """

    metric_MLM = load_metric("accuracy")
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    references = [[label] for label in labels]
    metric_MLM.add_batch(predictions=predictions, references=references)
    return metric_MLM.compute(predictions=predictions, references=references)


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
    model = BertForMaskedLM.from_pretrained(MODEL_BASE_NAME).to(device)  # .half()
    return model


def load_trainer(
    model,  # : AutoModelForSequenceClassification,BertForMaskedLM
    training_args: TrainingArguments,
    train_ds: Dataset,
    val_ds: Dataset,
    tokenizer: AutoTokenizer,
    modeltype: str,
) -> Trainer:
    """
    Load Trainer for model training

    Parameters
    ----------
    model : AutoModelForSequenceClassification
        Model to train
    training_args : TrainingArguments
        Training arguments for model
    train_ds : Dataset
        Training dataset
    val_ds : Dataset
        Validation dataset
    tokenizer : AutoTokenizer
        Tokenizer for data encoding
    modeltype : str
        Masked Language Model or Sequence Classification

    Returns
    -------
    Trainer
        Trainer with set arguments
    """
    if modeltype == "MLM":
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=0.15
        )
    else:
        data_collator = None

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    return trainer


def main():
    """
    Main function
    """

    train_ds, val_ds = load_datasets(
        os.path.join("data", "processed", "nlp", "mtsamples", "mtsamples_cleaned.csv")
    )

    tokenizer = load_tokenizer()
    tokenized_train_ds = tokenize_dataset(train_ds, tokenizer)
    tokenized_val_ds = tokenize_dataset(val_ds, tokenizer)

    device = get_device()
    model = load_model(device)
    training_args = load_training_args(MODEL_MLM_MT_CHECKPOINTS_DIR)
    trainer = load_trainer(
        model,
        training_args,
        tokenized_train_ds,
        tokenized_val_ds,
        tokenizer,
        modeltype="MLM",
    )
    trainer.train()
    trainer.save_model(MODEL_MLM_MT_DIR)
    trainer.save_state()


if __name__ == "__main__":
    main()
