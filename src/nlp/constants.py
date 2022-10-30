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
MODEL_SEMI_SUPERVISED_NAME = "emilyalsentzer/Bio_ClinicalBERT"
MODEL_SEMI_SUPERVISED_CHECKPOINTS_DIR = os.path.join("models", "nlp", "semi_supervised", "checkpoints")
MODEL_SEMI_SUPERVISED_MODEL_DIR = os.path.join("models", "nlp", "semi_supervised", "model")