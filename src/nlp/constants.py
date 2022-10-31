import os

# RAW DATA NLP
NLP_RAW_PATH_DIR = os.path.join("data", "raw", "nlp")
MTSAMPLES_RAW_PATH_DIR = os.path.join(NLP_RAW_PATH_DIR, "mtsamples")
PATIENT_NOTES_RAW_PATH_DIR = os.path.join(NLP_RAW_PATH_DIR, "patient_notes")

# PROCESSED DATA NLP
NLP_PROCESSED_PATH_DIR = os.path.join("data", "processed", "nlp")
MTSAMPLES_PROCESSED_PATH_DIR = os.path.join(NLP_PROCESSED_PATH_DIR, "mtsamples")
PATIENT_NOTES_PROCESSED_PATH_DIR = os.path.join(NLP_PROCESSED_PATH_DIR, "patient_notes")

# MODEL_SEMI_SUPERVISED
MODEL_UNSUPERVISED_NAME = "emilyalsentzer/Bio_ClinicalBERT"
MODEL_UNSUPERVISED_CHECKPOINTS_DIR = os.path.join(
    "models", "nlp", "unsupervised", "checkpoints"
)
MODEL_UNSUPERVISED_MODEL_DIR = os.path.join("models", "nlp", "unsupervised", "model")

# MODEL_UNSUPERVISED


# configs
# HYPERPARAMS
SEED_SPLIT = 0
SEED_TRAIN = 0

MAX_SEQ_LEN = 128
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
LEARNING_RATE = 2e-5
LR_WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01
