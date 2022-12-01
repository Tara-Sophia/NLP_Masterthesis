# -*- coding: utf-8 -*-

"""
Description:
    This module contains the constants used in the clf folder
"""
import os

# path to data (current NLP output for MT Samples)
RAW_DATA_DIR_MT = os.path.join(
    "data",
    "processed",
    "nlp",
    "mtsamples",
    "mtsamples_finaloutput.csv",
)

# RAW_DATA_DIR_MIMIC = os.path.join(
#     "data",
#     "processed",
#     "nlp",
#     "mtsamples",
#     "mtsamples_finaloutput.csv",
# )

# columns
X_ORIGINAL = "keywords_list"
X_MASKED = "transcription_f_MLM"
X_CLASSIFIED = "transcription_f_TC"

# path to train data
TRAIN_DATA_DIR = os.path.join("data", "processed", "clf", "train.csv")

# path to test data
TEST_DATA_DIR = os.path.join("data", "processed", "clf", "test.csv")

# path to models
LR_MT_MASKED = os.path.join("models", "clf", "lr_mt_masked.pkl")
LR_MT_CLASSIFIED = os.path.join("models", "clf", "lr_mt_classified.pkl")
LR_MIMIC_MASKED = os.path.join("models", "clf", "lr_mimic_masked.pkl")
LR_MIMIC_CLASSIFIED = os.path.join("models", "clf", "lr_mimic_classified.pkl")

RF_MT_MASKED = os.path.join("models", "clf", "rf_mt_masked.pkl")
RF_MT_CLASSIFIED = os.path.join("models", "clf", "rf_mt_classified.pkl")
RF_MIMIC_MASKED = os.path.join("models", "clf", "rf_mimic_masked.pkl")
RF_MIMIC_CLASSIFIED = os.path.join("models", "clf", "rf_mimic_classified.pkl")

DT_MT_MASKED = os.path.join("models", "clf", "dt_mt_masked.pkl")
DT_MT_CLASSIFIED = os.path.join("models", "clf", "dt_mt_classified.pkl")
DT_MIMIC_MASKED = os.path.join("models", "clf", "dt_mimic_masked.pkl")
DT_MIMIC_CLASSIFIED = os.path.join("models", "clf", "dt_mimic_classified.pkl")

SVM_MT_MASKED = os.path.join("models", "clf", "svm_mt_masked.pkl")
SVM_MT_CLASSIFIED = os.path.join("models", "clf", "svm_mt_classified.pkl")
SVM_MIMIC_MASKED = os.path.join("models", "clf", "svm_mimic_masked.pkl")
SVM_MIMIC_CLASSIFIED = os.path.join("models", "clf", "svm_mimic_classified.pkl")

XGB_MT_MASKED = os.path.join("models", "clf", "xgb_mt_masked.pkl")
XGB_MT_CLASSIFIED = os.path.join("models", "clf", "xgb_mt_classified.pkl")
XGB_MIMIC_MASKED = os.path.join("models", "clf", "xgb_mimic_masked.pkl")
XGB_MIMIC_CLASSIFIED = os.path.join("models", "clf", "xgb_mimic_classified.pkl")
