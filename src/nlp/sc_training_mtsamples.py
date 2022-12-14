# -*- coding: utf-8 -*-
"""
Description:
    This script is used to train the text classification model.
    The model is trained on the MTSamples dataset.
    The model is trained using the HuggingFace Trainer class.
"""
import os

import numpy as np
import pandas as pd
import torch
import wandb
from datasets import Dataset, load_metric
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.trainer_utils import get_last_checkpoint

from src.nlp.constants import (
    MODEL_BASE_NAME,
    MODEL_TC_MT_CHECKPOINTS_DIR,
    MODEL_TC_MT_DIR,
    MOST_COMMON_WORDS_FILTERED,
)
from src.nlp.utils import (
    get_device,
    load_tokenizer,
    load_training_args,
    tokenize_function,
)

wandb.init(
    project="nlp",
    entity="nlp_masterthesis",
    tags=["textclassification"],
)


def load_trainer(
    model,
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

    metric = load_metric("accuracy", average="macro")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


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
    df = pd.read_csv(path)
    dict_medical_specialty = {
        value: idx for idx, value in enumerate(df.medical_specialty.unique())
    }
    df["labels"] = df.medical_specialty.map(dict_medical_specialty)
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
    dataset = Dataset.from_pandas(map_medical_specialty_to_labels(data_path))
    # remove most common words from list MOST_COMMON_WORDS from transcription column
    dataset = dataset.map(
        lambda x: {
            "transcription": " ".join(
                [
                    word
                    for word in x["transcription"].split()
                    if word not in MOST_COMMON_WORDS_FILTERED
                ]
            )
        }
    )
    dataset_train_test = dataset.train_test_split(test_size=0.1)
    # train dataset
    dataset_train_val = dataset_train_test["train"].train_test_split(test_size=0.1)
    dataset_train = dataset_train_val["train"]
    # validation dataset
    dataset_val = dataset_train_val["test"]

    return dataset_train, dataset_val


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
    tokenized_dataset = tokenized_dataset.remove_columns(
        [
            "Unnamed: 0",
            "description",
            "medical_specialty",
            "sample_name",
            "transcription",
            "keywords",
            "keywords_list",
            "location",
        ]
    )
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
        MODEL_BASE_NAME, num_labels=39
    ).to(device)

    return model


def main() -> None:
    """
    Main function
    """

    train_ds, val_ds = load_datasets(
        os.path.join("data", "processed", "nlp", "mtsamples", "mtsamples_cleaned.csv")
        # MTSAMPLES_PROCESSED_PATH_DIR, "mtsamples_cleaned.csv")
    )

    tokenizer = load_tokenizer()
    tokenized_train_ds = tokenize_dataset(train_ds, tokenizer)
    tokenized_val_ds = tokenize_dataset(val_ds, tokenizer)

    tokenized_train_ds = clean_remove_column(tokenized_train_ds)
    tokenized_val_ds = clean_remove_column(tokenized_val_ds)

    device = get_device()
    model = load_model(device)
    training_args = load_training_args(MODEL_TC_MT_CHECKPOINTS_DIR)
    trainer = load_trainer(
        model,
        training_args,
        tokenized_train_ds,
        tokenized_val_ds,
        tokenizer,
        modeltype="sequence_classification",
    )

    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is None:
        resume_from_checkpoint = None
    else:
        resume_from_checkpoint = True

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(MODEL_TC_MT_DIR)
    trainer.save_state()


if __name__ == "__main__":
    main()
