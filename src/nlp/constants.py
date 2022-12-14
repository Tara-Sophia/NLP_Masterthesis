# -*- coding: utf-8 -*-
"""
Description:
    This script contains all constants used in the project.
"""
import os

# NLP MTSAMPLES
# raw data
NLP_RAW_PATH_DIR = os.path.join("data", "raw", "nlp")
MTSAMPLES_RAW_PATH_DIR = os.path.join(NLP_RAW_PATH_DIR, "mtsamples")

# processed
NLP_PROCESSED_PATH_DIR = os.path.join("data", "processed", "nlp", "mtsamples")
MTSAMPLES_PROCESSED_PATH_DIR = os.path.join(
    NLP_PROCESSED_PATH_DIR, "mtsamples_cleaned.csv"
)
# outcome
MTSAMPLES_TC_DIR = os.path.join(NLP_PROCESSED_PATH_DIR, "mtsamples_TC.csv")
MTSAMPLES_MLM_DIR = os.path.join(NLP_PROCESSED_PATH_DIR, "mtsamples_MLM.csv")

# NLP MIMIC
MIMIC_TC_DIR = os.path.join("data", "processed", "mimic_iii", "mimic_TC.csv")
MIMIC_MLM_DIR = os.path.join("data", "processed", "mimic_iii", "mimic_MLM.csv")

MIMIC_PROCESSED_CLEANED_DIR = os.path.join(
    "data", "processed", "mimic_iii", "diagnoses_noteevents_cleaned.csv"
)

# MODEL MTSAMPLES
# Masked Language Model
MODEL_MLM_MT_DIR = os.path.join(
    "models", "nlp", "maskedlanguagemodel_Mtsamples", "model"
)
MODEL_MLM_MT_CHECKPOINTS_DIR = os.path.join(
    "models", "nlp", "maskedlanguagemodel_Mtsamples", "checkpoints"
)

# Text Classification
# TEXT CLASSIFICATION MODEL
MODEL_TC_MT_DIR = os.path.join("models", "nlp", "textclassification_Mtsamples", "model")
MODEL_TC_MT_CHECKPOINTS_DIR = os.path.join(
    "models", "nlp", "textclassification_Mtsamples", "checkpoints"
)

MODEL_BASE_NAME = "emilyalsentzer/Bio_ClinicalBERT"

# MIMIC MODEL
# MASKEED LANGUAGE MODEL
MODEL_MLM_CHECKPOINTS_DIR = os.path.join(
    "models", "nlp", "maskedlanguagemodel", "checkpoints"
)
MODEL_MLM_DIR = os.path.join("models", "nlp", "maskedlanguagemodel", "model")


# textclassification model
MODEL_TC_MIMIC_CHECKPOINTS_DIR = os.path.join(
    "models", "nlp", "textclassification", "checkpoints"
)
MODEL_TC_MIMIC_DIR = os.path.join("models", "nlp", "textclassification", "model")

# HYPERPARAMS
SEED_SPLIT = 0
SEED_TRAIN = 0

MAX_SEQ_LEN = 128
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 1
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

# MIMIC-III
MIMIC_DATA_DIR = os.path.join(
    "data", "processed", "mimic_iii", "diagnoses_noteevents.csv"
)

MIMIC_DATA_CLEANED = os.path.join(
    "data",
    "processed",
    "mimic_iii",
    "diagnoses_noteevents_cleaned.csv",
)

MIMIC_PERSONALIZED_STOPWORDS_FILTERED = [
    "He",
    "She",
    "patient",
    "**]",
    "[**Hospital1",
    "The",
    "given",
    "showed",
    "also",
    "In",
    "On",
    "denies",
    "history",
    "found",
    "transferred",
    "ED",
    "Patient",
    "Name",
    "noted",
    "s/p",
    "started",
    "prior",
    "18**]",
    "admitted",
    "CT",
    "Pt",
    "2",
    "presented",
    "IV",
    "reports",
    "pt",
    "recent",
    "last",
    "received",
    "No",
    "BP",
    "ED,",
    "year",
    "old",
    "[**Known",
    "past",
    "1",
    "days",
    "lastname",
    "His",
    "OSH",
    "arrival",
    "time",
    "[**Last",
    "yo",
    "This",
    "presents",
    "well",
    "[**Hospital",
    "HR",
    "male",
    "mg",
    "x",
    "day",
    "Her",
    "admission",
    "without",
    "At",
    "home",
    "felt",
    "initial",
    "developed",
    "revealed",
    "(un)",
    "3",
    "since",
    "placed",
    "increased",
    "per",
    "A",
    "h/o",
    "recently",
    "CXR",
    "Per",
    "severe",
    "significant",
    "treated",
    "w/",
    "transfer",
    "L",
    "underwent",
    "initially",
    "[**Hospital3",
    "due",
    "states",
    "Denies",
    "one",
    "R",
    "notable",
    "symptoms",
    "seen",
    "ED.",
    "O2",
    "called",
    "RR",
    "status",
    "EKG",
    "several",
    "review",
    "Of",
    "feeling",
    "continued",
    "fevers,",
    "hospital",
    "[**Location",
    "(NI)",
    "Mr.",
    "went",
    "HTN,",
    "T",
    "(STitle)",
    "note,",
    "today",
    "VS",
    "became",
    "discharged",
    "MICU",
    "weeks",
    "ago",
    "episode",
    "4",
    "taken",
    "new",
    "sent",
    "normal",
    "[**Name",
    "medical",
    "episodes",
    "two",
    "chills,",
    "aortic",
    "100%",
    "denied",
    "improved",
    "possible",
    "unable",
    "SOB",
    "EMS",
    "morning",
    "associated",
    "elevated",
    "large",
    "reported",
    "brought",
    "week",
    "[**First",
    "RA.",
    "night",
    "course",
    "Dr.",
    "M",
    "GI",
    "decreased",
    "ICU",
    "WBC",
]
