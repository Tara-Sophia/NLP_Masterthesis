import os

# CLEAN DATA
RAW_DATA_DIR = os.path.join("data", "raw", "stt")
RECORDINGS_FILE = os.path.join(
    RAW_DATA_DIR, "overview-of-recordings.csv"
)
RAW_RECORDINGS_DIR = os.path.join(RAW_DATA_DIR, "recordings")


# TRANSFORM DATA
PROCESSED_DATA_DIR = os.path.join("data", "processed", "stt")
CHARS_TO_IGNORE_REGEX = '[\,\?\.\!\-\;\:"\“\%\‘\”\�]'
SAMPLING_RATE = 16000
MAX_DURATION_LENGTH = 4.5

# FACEBOOK WAV2VEC2 TRAINING
WAV2VEC2_MODEL = "facebook/wav2vec2-base"
WAV2VEC2_MODEL_DIR = os.path.join(
    "models", "stt", "wav2vec2", "model"
)
WAV2VEC2_VOCAB_DIR = os.path.join(
    "models", "stt", "wav2vec2", "vocab"
)
WAV2VEC2_PROCESSOR_DIR = os.path.join(
    "models", "stt", "wav2vec2", "vocab"
)

WAV2VEC2_NUM_EPOCHS = 30
WAV2VEC2_BATCH_SIZE = 16

# OPENAI WHISPER TRAINING
WHISPER_MODEL = "openai/whisper"
WHISPER_MODEL_DIR = os.path.join("models", "stt", "whisper", "model")


# STT Report HTML
STT_REPORT = os.path.join("reports", "stt")
