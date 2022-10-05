# -*- coding: utf-8 -*-

"""
Description:
    
Usage:
    
Possible arguments:
    * 
"""
import pandas as pd
import numpy as np
import string
import re
import nltk
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.model_selection import GridSearchCV

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import cohen_kappa_score
from sklearn.manifold import TSNE

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer

from imblearn.over_sampling import SMOTE


# Split the dataframe into test and train data
def split_data(df):
    X = tfIdfMat_reduced
    y = labels

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    data = {"train": {"X": X_train, "y": y_train}, "test": {"X": X_test, "y": y_test}}
    return data


# Train the model, return the model
def train_model(data, args):
    lr_model = LogisticRegression(**args)
    lr_model.fit(data["train"]["X"], data["train"]["y"])
    return lr_model


# Evaluate the metrics for the model
def get_model_metrics(reg_model, data):
    preds = lr_model.predict(data["test"]["X"])
    report = classification_report(data["test"]["y"], preds, target_names=category_list)
    return report


def main():
    # Load Data
    mtsamples_df = pd.read_csv("../data/raw/mtsamples.csv")
    # build model 
    # train model
    # evaluate model
    
    # General Data Cleaning
    mtsamples_df = mtsamples_df.dropna()
    mtsamples_df = mtsamples_df.drop_duplicates()
    # Data Preprocessing
    data_categories = mtsamples_df.groupby(mtsamples_df["medical_specialty"])
    filtered_data_categories = data_categories.filter(lambda x: x.shape[0] > 100)
    final_data_categories = filtered_data_categories.groupby(
        filtered_data_categories["medical_specialty"]
    )
    data = filtered_data_categories[["transcription", "medical_specialty"]]

    labels = data["medical_specialty"].tolist()

    # Split Data into Training and Validation Sets
    data = split_data(data)

    # Train Model on Training Set
    args = {random_state=42, penalty="l1", solver="saga", multi_class="multinomial", C=1}
    reg = train_model(data, args)

    # Validate Model on Validation Set
    metrics = get_model_metrics(reg, data)

    # Save Model
    model_name = "sklearn_logistic_regression_model.pkl"

    joblib.dump(value=reg, filename=model_name)


if __name__ == "__main__":
    main()
