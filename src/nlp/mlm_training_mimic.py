# -*- coding: utf-8 -*-

"""
Training MLM on MIMIC III dataset
"""

import os

import numpy as np
import pandas as pd
import wandb
from datasets import Dataset, load_metric
from datasets.arrow_dataset import Batch
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.trainer_utils import get_last_checkpoint

from src.nlp.constants import MODEL_MLM_CHECKPOINTS_DIR, MODEL_MLM_DIR, SEED_SPLIT
from src.nlp.mlm_training_mtsamples import load_model
from src.nlp.utils import (  # load_trainer,
    get_device,
    load_tokenizer,
    load_training_args,
)

wandb.init(project="nlp", entity="nlp_masterthesis", tags=["mlm_mimic_iii"])


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
    metric = load_metric("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    predictions = predictions.reshape(-1)
    labels = labels.reshape(-1)

    return metric.compute(predictions=predictions, references=labels)


def load_trainer(
    model: AutoModelForSequenceClassification,
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
    # df_mlm = df_mlm.sample(frac=0.1)
    df_mlm = df_mlm.dropna()
    # Train/Valid Split
    df_train, df_valid = train_test_split(
        df_mlm, test_size=0.15, random_state=SEED_SPLIT
    )
    # Convert to Dataset object
    dataset_train = Dataset.from_pandas(df_train[["TEXT_final_cleaned"]])  # .dropna())
    dataset_val = Dataset.from_pandas(df_valid[["TEXT_final_cleaned"]])  # .dropna())
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
        # num_proc=multiprocessing.cpu_count() - 1,
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
    model = load_model(device)  # .half()
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
