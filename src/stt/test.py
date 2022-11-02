#!/usr/bin/env python
# coding=utf-8
# Copyright 2022 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning OpenAI Whisper models for speech recognition.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.
# flake8: noqa: E501
import logging
import os
import re

import torchaudio
import whisper
import sys
from dataclasses import dataclass, field

from typing import Optional, Dict, Union, List

import numpy as np
import torch


from transformers.trainer_utils import (
    get_last_checkpoint,
    is_main_process,
)
from transformers.utils import check_min_version
from transformers.utils.versions import require_version


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to pretrained model or model identifier from OpenAI Whisper NGC."
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where to store the pretrained models downloaded from huggingface.co or OpenAI Whisper NGC."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    manifest_path: str = field(
        default="data",
        metadata={"help": "Manifest path."},
    )
    tokenizer_path: str = field(
        default="tokenizers",
        metadata={"help": "Tokenizer path."},
    )
    freeze_encoder: bool = field(
        default=False,
        metadata={
            "help": "Freeze the acoustic encoder of the model. Recommend when fine-tuning on small datasets."
        },
    )
    num_beams: int = field(
        default=1,
        metadata={"help": "Number of beams for evaluation."},
    )
    length_penalty: float = field(
        default=1.0,
        metadata={"help": "Length penalty for evaluation."},
    )
    use_adam8bit: bool = field(
        default=False,
        metadata={
            "help": "Whether to use bitsandbytes 8bit AdamW optimiser."
        },
    )
    dropout_rate: float = field(
        default=0.0,
        metadata={
            "help": "The dropout ratio for all dropout layers (default=0)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: str = field(
        default=None,
        metadata={
            "help": "The name of the dataset to use (via the datasets library)."
        },
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the column in the datasets containing the full texts (for summarization)."
        },
    )
    dataset_cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to cache directory for saving and loading datasets"
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={
            "help": "Overwrite the cached training and evaluation sets"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={
            "help": "The number of processes to use for the preprocessing."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
            "value if set."
        },
    )
    audio_column_name: str = field(
        default="audio",
        metadata={
            "help": "The name of the dataset column containing the audio data. Defaults to 'audio'"
        },
    )
    text_column_name: str = field(
        default="text",
        metadata={
            "help": "The name of the dataset column containing the text data. Defaults to 'text'"
        },
    )
    max_duration_in_seconds: float = field(
        default=20.0,
        metadata={
            "help": "Truncate training audio files that are longer than `max_duration_in_seconds` seconds to 'max_duration_in_seconds`"
        },
    )
    min_duration_in_seconds: float = field(
        default=0.0,
        metadata={
            "help": "Filter audio files that are shorter than `min_duration_in_seconds` seconds"
        },
    )
    max_eval_duration_in_seconds: float = field(
        default=None,
        metadata={
            "help": "Truncate eval/test audio files that are longer than `max_duration_in_seconds` seconds to 'max_duration_in_seconds`"
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    min_target_length: Optional[int] = field(
        default=0,
        metadata={
            "help": "The minimum total sequence length for target text after tokenization. Sequences shorter "
            "than this will be filtered."
        },
    )
    preprocessing_only: bool = field(
        default=False,
        metadata={
            "help": "Whether to only do data preprocessing and skip training. "
            "This is especially useful when data preprocessing errors out in distributed training due to timeout. "
            "In this case, one should run the preprocessing in a non-distributed setup with `preprocessing_only=True` "
            "so that the cached datasets can consequently be loaded in distributed training"
        },
    )
    train_split_name: str = field(
        default="train",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"
        },
    )
    eval_split_name: str = field(
        default="validation",
        metadata={
            "help": "The name of the evaluation data set split to use (via the datasets library). Defaults to 'validation'"
        },
    )
    test_split_name: str = field(
        default="test",
        metadata={
            "help": "The name of the test data set split to use (via the datasets library). Defaults to 'test'"
        },
    )
    do_lower_case: bool = field(
        default=True,
        metadata={
            "help": "Whether the target text should be lower cased."
        },
    )
    wandb_project: str = field(
        default="speech-recognition-whisper",
        metadata={"help": "The name of the wandb project."},
    )
    ignore_verifications: bool = field(
        default=False,
        metadata={
            "help": "Ignore the verifications of the downloaded/processed dataset information in `load_dataset` (checksums/size/splits/...)."
        },
    )
    torchaudio_resampler: bool = field(
        default=False,
        metadata={
            "help": "Whether to use torchaudio to resample. If `False` (default) will use the default datataset backed."
        },
    )


def to_pad_to_mel(array):
    """Static function which:
    1. Pads/trims a list of audio arrays to a max length of 30s
    2. Computes log-mel filter coefficients from padded/trimmed audio sequences
    Inputs:
        array: list of audio arrays
    Returns:
        input_ids: torch.tensor of log-mel filter bank coefficients
    """
    padded_input = whisper.pad_or_trim(
        np.asarray(array, dtype=np.float32)
    )
    input_ids = whisper.log_mel_spectrogram(padded_input)
    return input_ids


def to_mel_to_pad(array):
    """Static function which:
    1. Computes log-mel filter coefficients from padded/trimmed audio sequences
    2. Pads/trims a list of audio arrays to a max length of 30s
    Inputs:
        array: list of audio arrays
    Returns:
        input_ids: torch.tensor of log-mel filter bank coefficients
    """
    mels = whisper.log_mel_spectrogram(
        np.asarray(array, dtype=np.float32)
    )
    input_ids = whisper.pad_or_trim(mels, 3000)
    return input_ids


@dataclass
class WhisperDataCollatorWithPadding:
    """
    Data collator that dynamically pads the audio inputs received. An EOS token is appended to the labels sequences.
    They are then dynamically padded to max length.
    Args:
        eos_token_id (`int`)
            The end-of-sentence token for the Whisper tokenizer. Ensure to set for sequences to terminate before
            generation max length.
    """

    eos_token_id: int
    time_stamp_token_id: int

    def __call__(
        self,
        features: List[Dict[str, Union[List[int], torch.Tensor]]],
    ) -> Dict[str, torch.Tensor]:
        """
        Since Whisper models don't have a HF processor defined (feature extractor + tokenizer), we'll pad by hand...
        """
        # split inputs and labels since they have to be of different lengths
        # and need different padding methods
        input_ids = [feature["input_ids"] for feature in features]
        labels = [feature["labels"] for feature in features]

        # first, pad the audio inputs to max_len
        input_ids = torch.concat(
            [
                to_pad_to_mel(input_val)[None, :]
                for input_val in input_ids
            ]
        )

        # next, append the eos token to our sequence of labels
        labels = [lab + [self.eos_token_id] for lab in labels]
        # finally, pad the target labels to max_len
        label_lengths = [len(lab) for lab in labels]
        max_label_len = max(label_lengths)
        labels = [
            np.pad(
                lab,
                (0, max_label_len - lab_len),
                "constant",
                constant_values=-100,
            )
            for lab, lab_len in zip(labels, label_lengths)
        ]

        batch = {"labels": labels}
        batch = {
            k: torch.tensor(np.array(v), requires_grad=False)
            for k, v in batch.items()
        }

        batch["input_ids"] = input_ids

        return batch


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(
            training_args.output_dir
        )
        if (
            last_checkpoint is None
            and len(os.listdir(training_args.output_dir)) > 0
        ):
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # load the model
    if os.path.isfile(model_args.model_name_or_path):
        model = whisper.Whisper.load_trained(
            model_args.model_name_or_path
        )
    else:
        model = whisper.load_model(
            model_args.model_name_or_path,
            dropout_rate=model_args.dropout_rate,
        )

    # set the dropout for the MLP layers -> we do this here as the MLP layers are written as a 'sequential'
    # so changing the modelling code gives mis-matches in the state-dict

    if not model_args.freeze_encoder:
        # only apply dropout when training the encoder
        for block_idx in range(len(model.encoder.blocks)):
            mlp_layer = model.encoder.blocks[block_idx].mlp
            # going very verbose to explain what we're doing here!
            fc1 = mlp_layer[0]
            act_fn = mlp_layer[1]
            dropout = nn.Dropout(p=model_args.dropout_rate)
            fc2 = mlp_layer[2]
            model.encoder.blocks[block_idx].mlp = nn.Sequential(
                fc1, act_fn, dropout, fc2, dropout
            )

    """for block_idx in range(len(model.decoder.blocks)):
        mlp_layer = model.decoder.blocks[block_idx].mlp
        fc1 = mlp_layer[0]
        act_fn = mlp_layer[1]
        dropout = nn.Dropout(p=model_args.dropout_rate)
        fc2 = mlp_layer[2]
        model.decoder.blocks[block_idx].mlp = nn.Sequential(fc1, act_fn, dropout, fc2, dropout)"""

    # load the tokenizer
    whisper_tok = whisper.tokenizer.get_tokenizer(
        False, task="transcribe", language="en"
    )
    tokenizer = whisper_tok.tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    # 4. Load dataset
    raw_datasets = DatasetDict()

    if training_args.do_train:
        raw_datasets["train"] = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split=data_args.train_split_name,
            cache_dir=data_args.dataset_cache_dir,
            use_auth_token=True
            if model_args.use_auth_token
            else None,
        )

    if training_args.do_eval:
        raw_datasets["eval"] = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split=data_args.eval_split_name,
            cache_dir=data_args.dataset_cache_dir,
            use_auth_token=True
            if model_args.use_auth_token
            else None,
        )

    if training_args.do_predict:
        test_split = data_args.test_split_name.split("+")
        for split in test_split:
            raw_datasets[split] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=split,
                cache_dir=data_args.dataset_cache_dir,
                use_auth_token=True
                if model_args.use_auth_token
                else None,
            )

    if (
        not training_args.do_train
        and not training_args.do_eval
        and not training_args.do_predict
    ):
        raise ValueError(
            "Cannot not train, not do evaluation and not do prediction. At least one of "
            "training, evaluation or prediction has to be done."
        )

    # if not training, there is no need to run multiple epochs
    if not training_args.do_train:
        training_args.num_train_epochs = 1

    if (
        data_args.audio_column_name
        not in next(iter(raw_datasets.values())).column_names
    ):
        raise ValueError(
            f"--audio_column_name '{data_args.audio_column_name}' not found in dataset '{data_args.dataset_name}'. "
            "Make sure to set `--audio_column_name` to the correct audio column - one of "
            f"{', '.join(next(iter(raw_datasets.values())).column_names)}."
        )

    if (
        data_args.text_column_name
        not in next(iter(raw_datasets.values())).column_names
    ):
        raise ValueError(
            f"--text_column_name {data_args.text_column_name} not found in dataset '{data_args.dataset_name}'. "
            "Make sure to set `--text_column_name` to the correct text column - one of "
            f"{', '.join(next(iter(raw_datasets.values())).column_names)}."
        )

    # 6. Resample speech dataset ALWAYS
    if data_args.torchaudio_resampler:
        # TODO: remove hardcoding of orig sr
        resampler = torchaudio.transforms.Resample(
            16_000, sample_rate
        )
    else:
        raw_datasets = raw_datasets.cast_column(
            data_args.audio_column_name,
            datasets.features.Audio(sampling_rate=sample_rate),
        )
        resampler = None

    # 7. Preprocessing the datasets.
    # We need to read the audio files as arrays and tokenize the targets.
    max_input_length = int(
        data_args.max_duration_in_seconds * sample_rate
    )
    min_input_length = min(
        int(data_args.min_duration_in_seconds * sample_rate), 1
    )
    max_eval_input_length = (
        int(data_args.max_eval_duration_in_seconds * sample_rate)
        if data_args.max_eval_duration_in_seconds
        else None
    )
    max_target_length = data_args.max_target_length
    min_target_length = data_args.min_target_length
    audio_column_name = data_args.audio_column_name
    num_workers = data_args.preprocessing_num_workers
    text_column_name = data_args.text_column_name
    do_lower_case = data_args.do_lower_case
    dataset_name = data_args.dataset_name

    # Define tokens to ignore/replace
    tedlium_contractions = [
        " 's",
        " 't",
        " 're",
        " 've",
        " 'm",
        " 'll",
        " 'd",
        " 'clock",
        " 'all",
    ]

    gigaspeech_disfluencies = ["<other>", "<sil>"]
    swb_disfluencies = [
        "[noise]",
        "[laughter]",
        "[silence]",
        "[vocalized-noise]",
        "<a_aside>",
        "<b_aside>",
        "<e_aside>",
        "[laughter-",
        "_1",
        "[laugh]",
        "[sigh]",
        "[cough]",
        "[mn]",
        "[breath]",
        "[lipsmack]",
        "[sneeze]",
        "[skip]",
        "[pause]",
        "(%hesitation)",
        "(%HESITATION)",
    ]
    swb_punctuations = [
        "{",
        "}",
        "[",
        "]-",
        "]",
        "((",
        "))",
        "(",
        ")",
    ]
    earnings_disfluencies = [
        "<noise>",
        "<crosstalk>",
        "<affirmative>",
        "<inaudible>",
        "inaudible",
        "<laugh>",
        "<silence>",
    ]
    ignore_segments = [
        "ignore_time_segment_in_scoring",
        "<noise>",
        "<music>",
        "[noise]",
        "[laughter]",
        "[silence]",
        "[vocalized-noise]",
        "<crosstalk>",
        "<affirmative>",
        "<inaudible>",
        "<laugh>",
        "",
    ]
    ignore_segments = (
        ignore_segments
        + gigaspeech_disfluencies
        + swb_disfluencies
        + earnings_disfluencies
    )

    if (
        training_args.do_train
        and data_args.max_train_samples is not None
    ):
        raw_datasets["train"] = raw_datasets["train"].select(
            range(data_args.max_train_samples)
        )

    if (
        training_args.do_eval
        and data_args.max_eval_samples is not None
    ):
        raw_datasets["eval"] = raw_datasets["eval"].select(
            range(data_args.max_eval_samples)
        )

    if (
        training_args.do_predict
        and data_args.max_predict_samples is not None
    ):
        for split in test_split:
            raw_datasets[split] = raw_datasets[split].select(
                range(data_args.max_predict_samples)
            )

    # filter data where the targets are ignored in scoring
    def is_target_labels(input_str):
        return input_str.lower() not in ignore_segments

    raw_datasets = raw_datasets.filter(
        is_target_labels,
        num_proc=num_workers,
        input_columns=[text_column_name],
        desc="filtering data where the targets are ignored in scoring",
    )

    vectorized_datasets = raw_datasets.map(
        prepare_dataset,
        remove_columns=next(iter(raw_datasets.values())).column_names,
        num_proc=num_workers,
        desc="preprocess train dataset",
    )

    # filter training data with inputs longer than max_input_length
    def is_audio_in_length_range(input_length):
        return min_input_length < input_length < max_input_length

    if training_args.do_train:
        vectorized_datasets["train"] = vectorized_datasets[
            "train"
        ].filter(
            is_audio_in_length_range,
            num_proc=num_workers,
            input_columns=["input_lengths"],
        )

    if max_eval_input_length is not None:
        # filter training data with inputs longer than max_input_length
        def is_eval_audio_in_length_range(input_length):
            return (
                min_input_length
                < input_length
                < max_eval_input_length
            )

        if training_args.do_eval:
            vectorized_datasets["eval"] = vectorized_datasets[
                "eval"
            ].filter(
                is_eval_audio_in_length_range,
                num_proc=num_workers,
                input_columns=["input_lengths"],
            )

        if training_args.do_predict:
            for split in test_split:
                vectorized_datasets[split] = vectorized_datasets[
                    split
                ].filter(
                    is_eval_audio_in_length_range,
                    num_proc=num_workers,
                    input_columns=["input_lengths"],
                )

    normalizer = EnglishTextNormalizer()

    return results


if __name__ == "__main__":
    main()
