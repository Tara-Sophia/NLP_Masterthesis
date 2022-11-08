# -*- coding: utf-8 -*-

"""
Description:
    This module contains the constants used in the clf folder
"""
import os

# path to data (current NLP output for Mt samples)
RAW_DATA_DIR = os.path.join(
    "data", "processed", "nlp", "mtsamples", "mtsamples_unsupervised_both_v2.csv"
)

# columns
X_MASKED = "transcription_f_unsupervised"
X_CLASSIFIED = "transcription_f_semisupervised"

# path to train data
TRAIN_DATA_DIR = os.path.join("data", "processed", "clf", "train.csv")

# path to test data
TEST_DATA_DIR = os.path.join("data", "processed", "clf", "test.csv")

# path to models
LR_MODEL_MASKED = os.path.join("models", "clf", "lr_model_masked.pkl")
LR_MODEL_CLASSIFIED = os.path.join("models", "clf", "lr_model_classified.pkl")
LR_MODEL_MIMIC = os.path.join("models", "clf", "lr_model_mimic.pkl")

RF_MODEL_MASKED = os.path.join("models", "clf", "rf_model_masked.pkl")
RF_MODEL_CLASSIFIED = os.path.join("models", "clf", "rf_model_classified.pkl")
RF_MODEL_MIMIC = os.path.join("models", "clf", "rf_model_mimic.pkl")

DT_MODEL_MASKED = os.path.join("models", "clf", "dt_model_masked.pkl")
DT_MODEL_CLASSIFIED = os.path.join("models", "clf", "dt_model_classified.pkl")
DT_MODEL_MIMIC = os.path.join("models", "clf", "dt_model_mimic.pkl")
