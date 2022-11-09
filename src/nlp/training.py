# -*- coding: utf-8 -*-
# this is the training for the keyword Bert model

import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset, DatasetDict, load_dataset
from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    BertTokenizer,
    utoModelForSequenceClassification,
)


def load_dataset(data_path):
    # df = pd.read_csv(data_path)
    # df['transcription'] = df['transcription'].tolist()
    # df.to_csv('../data/raw/mtsamples_cleaned.csv', index=False)
    dataset = load_dataset(
        "csv", data_files=data_path
    )  # '../data/raw/mtsamples_cleaned.csv'
    dataset = dataset["train"]
    dataset = dataset.train_test_split(test_size=0.1)
    return dataset


checkpoint = "emilyalsentzer/Bio_ClinicalBERT"  # "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(batch):
    return tokenizer(
        batch["transcription"],
        padding="max_length",
        truncation=True,
        max_length=512,
    )


def tokenize_dataset(dataset):
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
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
    tokenized_dataset = tokenized_dataset.rename_column(
        "labels_val", "labels"
    )
    tokenized_dataset.set_format("torch")
    return tokenized_dataset


import numpy as np
from datasets import load_metric
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)


def compute_metrics(eval_pred):
    metric = load_metric("accuracy", average="macro")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)


def training_model(tokenized_dataset):
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        report_to="wandb",
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=39
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model("./results_modelBert")
    return model


def main():
    data_path = "../data/raw/mtsamples_cleaned.csv"
    dataset = load_dataset(data_path)
    tokenized_dataset = tokenize_dataset(dataset)
    tokenized_dataset = clean_remove_column(tokenized_dataset)
    model = training_model(tokenized_dataset)


# Path: src/Keyword_Bert_Training.py
if __name__ == "__main__":
    main()
