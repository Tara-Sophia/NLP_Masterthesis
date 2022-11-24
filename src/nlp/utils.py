# -*- coding: utf-8 -*-
"""
Description:
    This script has all the functions which are used in mutiple files
"""
import string

import numpy as np
import torch
from datasets import Dataset, load_metric

# import Batch
from datasets.arrow_dataset import Batch
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# import EvalPrediction
from transformers import AutoTokenizer, EvalPrediction, Trainer, TrainingArguments

# import DataCollatorForLanguageModeling
from transformers.data.data_collator import DataCollatorForLanguageModeling

from constants import (
    EVAL_BATCH_SIZE,
    LEARNING_RATE,
    LR_WARMUP_STEPS,
    MODEL_BASE_NAME,
    SEED_TRAIN,
    TRAIN_BATCH_SIZE,
    WEIGHT_DECAY,
)

# logic behind most common inputs the stopwords where manually filtered
# def find_most_common_words_by_count(df):
#     word_count_dict = {}
#     for index, row in df.iterrows():
#         for word in row["transcription_c"]:
#             print(word)
#             if word in word_count_dict:
#                 word_count_dict[word] += 1
#             else:
#                 word_count_dict[word] = 1
#     return word_count_dict


# common_words = find_most_common_words_by_count(df)
# list(common_words_sorted_df.word)[:150]


def cleaning_input(sentence: str, handmadestopwords: list[str]) -> str:
    """
    This function cleans the input sentence.
    Removing stopwords, numbers and punctuation,
    and lemmatizing the words.

    Parameters
    ----------
    sentence : str
        The sentence to be cleaned.

    Returns
    -------
    str
        The cleaned sentence.
    """
    # Basic cleaning
    sentence = sentence.strip()  # Remove whitespaces
    sentence = sentence.lower()  # Lowercase
    sentence = "".join(
        char for char in sentence if not char.isdigit()
    )  # Remove numbers

    # Advanced cleaning
    for punctuation in string.punctuation:
        sentence = sentence.replace(punctuation, "")  # Remove punctuation

    tokenized_sentence = word_tokenize(sentence)  # Rokenize
    stop_words = set(stopwords.words("english"))  # Define stopwords

    # w not in stop_words and not in handmadestopwords
    tokenized_sentence = [w for w in tokenized_sentence if w not in stop_words]

    tokenized_sentence_cleaned = [
        w for w in tokenized_sentence if w not in handmadestopwords
    ]  # Remove stopwords

    lemmatized = [
        WordNetLemmatizer().lemmatize(word, pos="v")
        for word in tokenized_sentence_cleaned
    ]

    cleaned_sentence = " ".join(word for word in lemmatized)

    return cleaned_sentence


def get_device() -> torch.device:
    """
    Get the device

    Returns
    -------
    torch.device
        Device
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")




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
        num_train_epochs=1,  # 30,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        warmup_steps=LR_WARMUP_STEPS,
        save_total_limit=1,
        fp16=True, 
        fp16_full_eval=True,
        weight_decay=WEIGHT_DECAY,
        learning_rate=LEARNING_RATE,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        seed=SEED_TRAIN,
        report_to="wandb",
        eval_accumulation_steps=10 #, gradient_checkpointing=True
    )
    return training_args


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


def load_tokenizer() -> AutoTokenizer:
    """
    Load tokenizer

    Returns
    -------
    AutoTokenizer
        Tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=MODEL_BASE_NAME,
        model_max_length=512,
        truncate=True,
        max_length=512,
        padding=True,
    )

    return tokenizer


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
        batch["transcription"],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_special_tokens_mask=special_token,
    )
