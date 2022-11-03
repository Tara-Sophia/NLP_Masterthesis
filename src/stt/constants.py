# -*- coding: utf-8 -*-
import os

# CLEAN DATA
RAW_DATA_DIR = os.path.join("data", "raw", "stt")
RECORDINGS_FILE = os.path.join(
    RAW_DATA_DIR, "overview-of-recordings.csv"
)
RAW_RECORDINGS_DIR = os.path.join(RAW_DATA_DIR, "recordings")


# TRANSFORM DATA
CHARS_TO_IGNORE_REGEX = r'[\,\?\.\!\-\;\:"\“\%\‘\”\�]'
SAMPLING_RATE = 16000
MAX_DURATION_LENGTH = 4.5
NUM_PROC = 5

# FACEBOOK WAV2VEC2
WAV2VEC2_TRAIN_PROCESSED_DIR = os.path.join(
    "data", "processed", "stt", "wav2vec", "train"
)
WAV2VEC2_VAL_PROCESSED_DIR = os.path.join(
    "data", "processed", "stt", "wav2vec", "val"
)
WAV2VEC2_TEST_PROCESSED_DIR = os.path.join(
    "data", "processed", "stt", "wav2vec", "test"
)
WAV2VEC2_PROCESSED_DIR = os.path.join(
    "data", "processed", "stt", "wav2vec"
)
WAV2VEC2_MODEL = "facebook/wav2vec2-base"
WAV2VEC2_MODEL_DIR = os.path.join(
    "models", "stt", "wav2vec2", "model"
)
WAV2VEC2_VOCAB_DIR = os.path.join(
    "models", "stt", "wav2vec2", "vocab"
)
WAV2VEC2_MODEL_CHECKPOINTS = os.path.join(
    "models", "stt", "wav2vec2", "checkpoints"
)

WAV2VEC2_NUM_EPOCHS = 40
WAV2VEC2_BATCH_SIZE_TRAIN = 16
WAV2VEC2_BATCH_SIZE_EVAL = 16

# OPENAI WHISPER TRAINING

WHISPER_TRAIN_PROCESSED_DIR = os.path.join(
    "data", "processed", "stt", "whisper", "train"
)
WHISPER_VAL_PROCESSED_DIR = os.path.join(
    "data", "processed", "stt", "whisper", "val"
)
WHISPER_TEST_PROCESSED_DIR = os.path.join(
    "data", "processed", "stt", "whisper", "test"
)
WHISPER_PROCESSED_DIR = os.path.join(
    "data", "processed", "stt", "whisper"
)
WHISPER_MODEL = "openai/whisper-large"
WHISPER_MODEL_DIR = os.path.join("models", "stt", "whisper", "model")

WHISPER_VOCAB_DIR = os.path.join("models", "stt", "whisper", "vocab")
WHISPER_MODEL_CHECKPOINTS = os.path.join(
    "models", "stt", "whisper", "checkpoints"
)

WHISPER_NUM_EPOCHS = 30
WHISPER_BATCH_SIZE = 16

# STT Report HTML
STT_REPORT = os.path.join("reports", "stt")
