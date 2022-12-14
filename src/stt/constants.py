# -*- coding: utf-8 -*-

"""
Description:
    This module contains the constants used in the stt folder
"""
import os

# CLEAN DATA
RAW_DATA_DIR = os.path.join("data", "raw", "stt")
RECORDINGS_FILE = os.path.join(RAW_DATA_DIR, "overview-of-recordings.csv")
RAW_RECORDINGS_DIR = os.path.join(RAW_DATA_DIR, "recordings")


# TRANSFORM DATA
CHARS_TO_IGNORE_REGEX = r'[\,\?\.\!\-\;\:"\[\]]'
SAMPLING_RATE = 16000
MAX_DURATION_LENGTH = 10
NUM_PROC = 1
PROCESSED_DIR = os.path.join("data", "processed", "stt")
VOCAB_DIR = os.path.join("models", "stt", "vocab")

# FACEBOOK WAV2VEC2
WAV2VEC2_MODEL = "facebook/wav2vec2-base"
WAV2VEC2_MODEL_DIR = os.path.join("models", "stt", "wav2vec2", "model")
WAV2VEC2_MODEL_CHECKPOINTS = os.path.join("models", "stt", "wav2vec2", "checkpoints")

WAV2VEC2_NUM_EPOCHS = 30
WAV2VEC2_BATCH_SIZE_TRAIN = 16
WAV2VEC2_BATCH_SIZE_EVAL = 32


# HUBERT TRAINING
HUBERT_MODEL = "facebook/hubert-large-ll60k"
HUBERT_MODEL_DIR = os.path.join("models", "stt", "hubert", "model")
HUBERT_MODEL_CHECKPOINTS = os.path.join("models", "stt", "hubert", "checkpoints")

HUBERT_NUM_EPOCHS = 2
HUBERT_BATCH_SIZE_TRAIN = 16
HUBERT_BATCH_SIZE_EVAL = 32


# STT Report HTML
STT_REPORT = os.path.join("reports", "stt")
