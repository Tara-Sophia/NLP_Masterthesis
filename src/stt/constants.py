# -*- coding: utf-8 -*-
import os

# CLEAN DATA
RAW_DATA_DIR = os.path.join("data", "raw", "stt")
RECORDINGS_FILE = os.path.join(
    RAW_DATA_DIR, "overview-of-recordings.csv"
)
RAW_RECORDINGS_DIR = os.path.join(RAW_DATA_DIR, "recordings")


# TRANSFORM DATA
CHARS_TO_IGNORE_REGEX = '[\,\?\.\!\-\;\:"]+|\[|\]'
SAMPLING_RATE = 16000
MAX_DURATION_LENGTH = 4.5
NUM_PROC = 7
TRAIN_PROCESSED_DIR = os.path.join(
    "data", "processed", "stt", "train"
)
VAL_PROCESSED_DIR = os.path.join("data", "processed", "stt", "val")
TEST_PROCESSED_DIR = os.path.join("data", "processed", "stt", "test")
PROCESSED_DIR = os.path.join("data", "processed", "stt")
VOCAB_DIR = os.path.join("models", "stt", "vocab")


# FACEBOOK WAV2VEC2
WAV2VEC2_MODEL = "facebook/wav2vec2-base"
WAV2VEC2_MODEL_DIR = os.path.join(
    "models", "stt", "wav2vec2", "model"
)
WAV2VEC2_MODEL_CHECKPOINTS = os.path.join(
    "models", "stt", "wav2vec2", "checkpoints"
)

WAV2VEC2_NUM_EPOCHS = 5
WAV2VEC2_BATCH_SIZE_TRAIN = 16
WAV2VEC2_BATCH_SIZE_EVAL = 16

# HUBERT TRAINING

HUBERT_MODEL = "facebook/hubert-large-ll60k"
HUBERT_MODEL_DIR = os.path.join("models", "stt", "hubert", "model")
HUBERT_MODEL_CHECKPOINTS = os.path.join(
    "models", "stt", "hubert", "checkpoints"
)

HUBERT_NUM_EPOCHS = 30
HUBERT_BATCH_SIZE = 16

# STT Report HTML
STT_REPORT = os.path.join("reports", "stt")
