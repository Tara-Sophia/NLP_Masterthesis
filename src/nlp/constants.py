# -*- coding: utf-8 -*-
import os

# RAW DATA NLP
NLP_RAW_PATH_DIR = os.path.join("data", "raw", "nlp")
MTSAMPLES_RAW_PATH_DIR = os.path.join(NLP_RAW_PATH_DIR, "mtsamples")
PATIENT_NOTES_RAW_PATH_DIR = os.path.join(
    NLP_RAW_PATH_DIR, "patient_notes"
)

# PROCESSED DATA NLP
NLP_PROCESSED_PATH_DIR = os.path.join(
    "data", "processed", "nlp", "mtsamples"
)
MTSAMPLES_PROCESSED_PATH_DIR = os.path.join(
    NLP_PROCESSED_PATH_DIR, "mtsamples"
)
PATIENT_NOTES_PROCESSED_PATH_DIR = os.path.join(
    NLP_PROCESSED_PATH_DIR, "patient_notes"
)
MTSAMPLES_PROCESSED_CLEANED_DIR = os.path.join(
    NLP_PROCESSED_PATH_DIR, "mtsamples_cleaned.csv"
)
MTSAMPLES_FINAL = os.path.join(
    MTSAMPLES_PROCESSED_PATH_DIR, "mtsamples_finaloutput.csv"
)

MODEL_BASE_NAME = "emilyalsentzer/Bio_ClinicalBERT"
MODEL_MLM_NAME = "emilyalsentzer/Bio_ClinicalBERT"

# TEXT CLASSIFICATION MODEL
MODEL_TC_DIR = os.path.join(
    "models", "nlp", "tectclassification", "model"
)
MODEL_TC_CHECKPOINTS_DIR = os.path.join(
    "models", "nlp", "textclassification", "checkpoints"
)

# MASKEED LANGUAGE MODEL
MODEL_MLM_CHECKPOINTS_DIR = os.path.join(
    "models", "nlp", "maskedlanguagemodel", "checkpoints"
)
MODEL_MLM_DIR = os.path.join(
    "models", "nlp", "maskedlanguagemodel", "model"
)

# HYPERPARAMS
SEED_SPLIT = 0
SEED_TRAIN = 0

MAX_SEQ_LEN = 128
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
LEARNING_RATE = 2e-5
LR_WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01


# MTSamples List
MOST_COMMON_WORDS_FILTERED = [
    "patient",
    "leave",
    "diagnosis",
    "procedure",
    "history",
    "yearold",
    "perform",
    "diagnose",
    "present",
    "general",
    "place",
    "time",
    "use",
    "room",
    "see",
    "take",
    "without",
    "obtain",
    "cm",
    "find",
    "x",
    "cc",
    "prepped",
    "mg",
    "day",
    "also",
    "note",
    "exam",
    "position",
    "mass",
    "consent",
    "bring",
    "week",
    "risk",
    "include",
    "year",
    "make",
    "approximately",
    "give",
    "mm",
    "dr",
    "ct",
    "detail",
    "examination",
    "state",
    "show",
    "today",
    "fashion",
    "two",
    "usual",
    "inform",
    "deny",
    "follow",
    "last",
    "status",
    "significant",
    "evidence",
    "image",
    "symptom",
    "undergo",
    "minute",
    "medical",
    "l",
    "appear",
    "total",
    "extremity",
    "prior",
    "one",
    "iv",
    "hospital",
    "change",
    "reveal",
    "since",
    "past",
    "complaint",
    "tissue",
    "medication",
    "ago",
    "month",
    "tube",
    "clear",
    "ml",
    "within",
    "rate",
    "come",
    "per",
    "evaluation",
]
