# this is the training for the keyword Bert model

import torch
import torch.nn as nn
from datasets import Dataset, DatasetDict
from datasets import load_dataset
import pandas as pd

from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AutoTokenizer, utoModelForSequenceClassification


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


def tokenize_function(examples):
    return tokenizer(
        examples["transcription"], padding="max_length", truncation=True, max_length=512
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
    tokenized_dataset = tokenized_dataset.rename_column("labels_val", "labels")
    tokenized_dataset.set_format("torch")
    return tokenized_dataset


from datasets import load_metric
import numpy as np
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification


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
        checkpoint, num_labels=38
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


from keybert import KeyBERT


def KeywordExtraction(model, text):
    model = KeyBERT(model_path="./results_modelBert")
    keywords = model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),
        stop_words="english",
        use_maxsum=True,
        nr_candidates=25,
        top_n=10,
        use_mmr=True,
    )
    return keywords


def apply_keyword_on_Dataframe(model, df):
    df["keywords_outcome"] = df["transcription"].apply(
        lambda x: KeywordExtraction(model, x)
    )
    return df


def save_dataframe(df):
    df.to_csv("../data/raw/mtsamples_outcome_bert.csv", index=False)


if __name__ == "__main__":
    df_outcome = main()
    save_dataframe(df_outcome)


def main():
    data_path = "../data/raw/mtsamples_cleaned.csv"
    dataset = load_dataset(data_path)
    tokenized_dataset = tokenize_dataset(dataset)
    tokenized_dataset = clean_remove_column(tokenized_dataset)
    model = training_model(tokenized_dataset)
    df = pd.read_csv(data_path)
    df_outcome = apply_keyword_on_Dataframe(model, df)
    return df_outcome


# Path: src/Keyword_Bert_Training.py
