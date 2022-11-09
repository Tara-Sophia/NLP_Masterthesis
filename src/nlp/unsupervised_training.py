# -*- coding: utf-8 -*-
# imports
import multiprocessing
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import transformers
from constants import (
    EVAL_BATCH_SIZE,
    LEARNING_RATE,
    LR_WARMUP_STEPS,
    MODEL_UNSUPERVISED_CHECKPOINTS_DIR,
    MODEL_UNSUPERVISED_MODEL_DIR,
    MTSAMPLES_PROCESSED_PATH_DIR,
    SEED_SPLIT,
    SEED_TRAIN,
    TRAIN_BATCH_SIZE,
    WEIGHT_DECAY,
)
from datasets import Dataset, load_dataset, load_metric, metric
from sklearn.model_selection import train_test_split
from tokenizers import BertWordPieceTokenizer
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertForMaskedLM,
    BertTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

import wandb

wandb.init(project="nlp", entity="nlp_masterthesis")


def load_datasets(data_path):

    dtf_mlm = pd.read_csv(data_path)
    # Train/Valid Split
    df_train, df_valid = train_test_split(
        dtf_mlm, test_size=0.15, random_state=SEED_SPLIT
    )
    # Convert to Dataset object
    dataset_train = Dataset.from_pandas(
        df_train[["transcription"]].dropna()
    )
    dataset_val = Dataset.from_pandas(
        df_valid[["transcription"]].dropna()
    )
    return dataset_train, dataset_val


def tokenize_function(batch, tokenizer):  # before row
    return tokenizer(
        batch["transcription"],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_special_tokens_mask=True,
    )


def tokenize_dataset(dataset, tokenizer):
    column_names = dataset.column_names

    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=multiprocessing.cpu_count(),
        remove_columns=column_names,
        fn_kwargs={"tokenizer": tokenizer}
        # batched=True,
    )
    return tokenized_datasets


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)


def load_training_args(output_dir):
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=30,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        warmup_steps=LR_WARMUP_STEPS,
        save_total_limit=3,
        weight_decay=WEIGHT_DECAY,
        learning_rate=LEARNING_RATE,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        seed=SEED_TRAIN,
        report_to="wandb",
    )
    return training_args


def load_trainer(model, training_args, train_ds, val_ds, tokenizer):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,  # masks the tokens
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    return trainer


def get_device() -> torch.device:
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def load_model(device):
    ModelClass = BertForMaskedLM
    bert_type = "bert-base-cased"
    model = ModelClass.from_pretrained(bert_type).to(device)
    # AutoModelForSequenceClassification.from_pretrained(
    #    MODEL_SEMI_SUPERVISED_NAME, num_labels=39
    # ).to(device)

    return model


def load_tokenizer():
    # "bert-base-cased"
    TokenizerClass = BertTokenizer
    bert_type = "bert-base-cased"

    tokenizer = TokenizerClass.from_pretrained(
        bert_type,
        model_max_length=512,  # MAX_SEQ_LEN
        truncate=True,
        max_length=512,
        padding=True,
    )  # autotokenizer

    return tokenizer


def main():
    train_ds, val_ds = load_datasets(
        os.path.join(
            MTSAMPLES_PROCESSED_PATH_DIR, "mtsamples_cleaned.csv"
        )
    )

    tokenizer = load_tokenizer()
    tokenized_train_ds = tokenize_dataset(train_ds, tokenizer)
    tokenized_val_ds = tokenize_dataset(val_ds, tokenizer)

    device = get_device()
    # empty cache  with torch.cuda.empty_cache()
    model = load_model(device)
    training_args = load_training_args(
        MODEL_UNSUPERVISED_CHECKPOINTS_DIR
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
    eval_results = trainer.evaluate()

    trainer.save_model(MODEL_UNSUPERVISED_MODEL_DIR)
    trainer.save_state()


if __name__ == "__main__":
    main()
