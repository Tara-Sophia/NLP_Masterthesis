import os

DATA_PATH_DATASETS = os.path.join("data", "interim", "stt")

MODEL = "facebook/wav2vec2-base"
BATCH_SIZE = 16
PROCESSOR_PATH = os.path.join("models", "stt", "wav2vec2", "vocab")

DATA_PATH_WAV = os.path.join("data", "raw", "stt")
PROCESSOR_PATH = os.path.join("models", "stt", "wav2vec2", "vocab")

CHARS_TO_IGNORE_REGEX = '[\,\?\.\!\-\;\:"\“\%\‘\”\�]'

SAMPLING_RATE = 16000
