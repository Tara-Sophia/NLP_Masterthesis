# -*- coding: utf-8 -*-
import os

import numpy as np
import pandas as pd
import torch
from constants import (
    MODEL_SEMI_SUPERVISED_CHECKPOINTS_DIR,
    MODEL_SEMI_SUPERVISED_MODEL_DIR,
    MODEL_SEMI_SUPERVISED_NAME,
    MTSAMPLES_PROCESSED_PATH_DIR,
)
from datasets import Dataset, load_metric
from datasets.arrow_dataset import Batch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EvalPrediction,
)
from transformers.trainer_utils import get_last_checkpoint

import wandb

wandb.init(project="nlp", entity="nlp_masterthesis")


def map_medical_specialty_to_labels(path: str) -> pd.DataFrame:
    """
    Read the csv file
    Map the medical specialty to labels

    Parameters
    ----------
    path : str
        path to the csv file

    Returns
    -------
    pd.DataFrame
        dataframe with the mapped labels
    """
    df = pd.read_csv(path)
    dict_medical_specialty = {
        value: idx
        for idx, value in enumerate(df.medical_specialty.unique())
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
    dataset = Dataset.from_pandas(
        map_medical_specialty_to_labels(data_path)
    )
    dataset_train_test = dataset.train_test_split(test_size=0.1)
    # train dataset
    dataset_train_val = dataset_train_test["train"].train_test_split(
        test_size=0.1
    )
    dataset_train = dataset_train_val["train"]
    # validation dataset
    dataset_val = dataset_train_val["test"]

    return dataset_train, dataset_val


def tokenize_function(
    batch: Batch, tokenizer: AutoTokenizer
) -> Batch:
    return tokenizer(
        batch["transcription"],
        padding="max_length",
        truncation=True,
        max_length=512,
    )


def tokenize_dataset(
    dataset: Dataset, tokenizer: AutoTokenizer
) -> Dataset:
    tokenized_datasets = dataset.map(
        tokenize_function,
        fn_kwargs={"tokenizer": tokenizer},
        batched=True,
    )
    return tokenized_datasets


def clean_remove_column(tokenized_dataset):
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
    # tokenized_dataset = tokenized_dataset.rename_column(
    #     "labels_val", "labels"
    # )
    tokenized_dataset.set_format("torch")
    return tokenized_dataset


# def load multiple metrics f1, precision, recall, accuracy


def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    """
    Compute the accuracy of the model for the evaluation dataset

    Parameters
    ----------
    eval_pred : EvalPrediction
        Prediction for evaluation dataset

    Returns
    -------
    dict[str, float]
        Accuracy score
    """
    # load multiple metrics
    metric = load_metric("accuracy", average="macro")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)


def get_device() -> torch.device:
    """
    Get the device

    Returns
    -------
    torch.device
        Device
    """
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


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
        MODEL_SEMI_SUPERVISED_NAME, num_labels=39
    ).to(device)

    return model


def load_tokenizer() -> AutoTokenizer:
    """
    Load tokenizer

    Returns
    -------
    AutoTokenizer
        Tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_SEMI_SUPERVISED_NAME,
        model_max_length=512,
        truncate=True,
        max_length=512,
        padding=True,
    )

    return tokenizer


def load_training_args(output_dir: str) -> TrainingArguments:
    """
    Load training arguments

    Parameters
    ----------
    output_dir : str
        Directory to save the model checkpoints

    Returns
    -------
    TrainingArguments
        Training arguments
    """
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=30,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        # logging_dir="./logs",
        # logging_steps=10,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="wandb",
    )
    return training_args


def load_trainer(
    model: AutoModelForSequenceClassification,
    training_args: TrainingArguments,
    train_ds: Dataset,
    val_ds: Dataset,
    tokenizer: AutoTokenizer,
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

    Returns
    -------
    Trainer
        Trainer with set arguments
    """
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        # callbacks=[EarlyStoppingCallback()],
    )
    return trainer


def main() -> None:
    """
    Main function
    """

    train_ds, val_ds = load_datasets(
        os.path.join(
            MTSAMPLES_PROCESSED_PATH_DIR, "mtsamples_cleaned.csv"
        )
    )

    tokenizer = load_tokenizer()
    tokenized_train_ds = tokenize_dataset(train_ds, tokenizer)
    tokenized_val_ds = tokenize_dataset(val_ds, tokenizer)

    tokenized_train_ds = clean_remove_column(tokenized_train_ds)
    tokenized_val_ds = clean_remove_column(tokenized_val_ds)

    device = get_device()
    model = load_model(device)
    training_args = load_training_args(
        MODEL_SEMI_SUPERVISED_CHECKPOINTS_DIR
    )
    trainer = load_trainer(
        model,
        training_args,
        tokenized_train_ds,
        tokenized_val_ds,
        tokenizer,
    )

    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is None:
        resume_from_checkpoint = None
    else:
        resume_from_checkpoint = True

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(MODEL_SEMI_SUPERVISED_MODEL_DIR)
    trainer.save_state()


if __name__ == "__main__":
    main()
