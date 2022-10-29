import os

DATA_PATH_DATASETS = os.path.join("data", "processed", "stt")

MODEL = "facebook/wav2vec2-base"
MODEL_DIR = os.path.join("models", "stt", "wav2vec2", "model")

DATA_PATH_WAV = os.path.join("data", "raw", "stt")
VOCAB_PATH = os.path.join("models", "stt", "wav2vec2", "vocab")
PROCESSOR_PATH = os.path.join("models", "stt", "wav2vec2", "vocab")

CHARS_TO_IGNORE_REGEX = '[\,\?\.\!\-\;\:"\“\%\‘\”\�]'

SAMPLING_RATE = 16000

# Model training
NUM_EPOCHS = 30
BATCH_SIZE = 16
