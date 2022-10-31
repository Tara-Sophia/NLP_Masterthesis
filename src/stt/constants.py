import os

REL_PATH_RECORDINGS = os.path.join(
    "data", "raw", "stt"
)  # These can change, depending on the data we are using

CSV_FILE = os.path.join(
    REL_PATH_RECORDINGS, "overview-of-recordings.csv"
)

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


# STT Report HTML
STT_REPORT = os.path.join("reports", "stt")
