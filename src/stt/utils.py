# -*- coding: utf-8 -*-
import os

import pandas as pd
import torch
from constants import WAV2VEC2_MODEL_DIR, SAMPLING_RATE
from decorators import log_function_name
from datasets import Dataset
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
)


@log_function_name
def get_device() -> torch.device:
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


@log_function_name
def load_datasets(data_path):
    train_df = pd.read_feather(
        os.path.join(data_path, "train", "train.feather")
    )
    val_df = pd.read_feather(
        os.path.join(data_path, "val", "val.feather")
    )

    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)

    return train_ds, val_ds


# FACEBOOK WAV2VEC2


@log_function_name
def load_processor_wav2vec2(processor_path):

    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
        processor_path,
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
    )

    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=SAMPLING_RATE,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=False,
    )

    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor, tokenizer=tokenizer
    )

    return processor


@log_function_name
def load_trained_model_and_processor_wav2vec2(device):
    model = Wav2Vec2ForCTC.from_pretrained(WAV2VEC2_MODEL_DIR)
    processor = Wav2Vec2Processor.from_pretrained(WAV2VEC2_MODEL_DIR)
    model.to(device)
    return model, processor


# OPENAI WHISPER


# @log_function_name
# def load_tokenizer_whisper():
#     # load the tokenizer
#     whisper_tok = whisper.tokenizer.get_tokenizer(
#         False, task="transcribe", language="en"
#     )
#     tokenizer = whisper_tok.tokenizer
#     tokenizer.pad_token = tokenizer.eos_token
#     return tokenizer


@log_function_name
def load_processor_whisper(tokenizer_path):

    tokenizer = WhisperTokenizer.from_pretrained(
        tokenizer_path,
    )

    feature_extractor = WhisperFeatureExtractor(
        sampling_rate=SAMPLING_RATE,
    )

    processor = WhisperProcessor(
        feature_extractor=feature_extractor, tokenizer=tokenizer
    )
    return processor


@log_function_name
def load_trained_model_and_processor_whisper(device):
    model = ""
    processor = ""
    model.to(device)
    return model, processor
