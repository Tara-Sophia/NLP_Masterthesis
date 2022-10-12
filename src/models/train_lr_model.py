# -*- coding: utf-8 -*-

"""
Description:
    Training a logistic regression model for predicting probabilities of medical specialties
    
Usage:
    
Possible arguments:
    * 
"""
from array import array
import pandas as pd
import numpy as np
import string
import re
import nltk
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline
from sklearn.pipeline import Pipeline as skPipeline
from imblearn.pipeline import Pipeline as imbPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import cohen_kappa_score
from sklearn.manifold import TSNE

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer

from imblearn.over_sampling import SMOTE


# Retrieve labels as function
def get_labels(df: pd.DataFrame) -> list:
    """
    Get labels from dataframe

    Parameters
    ----------
    df : pd.DataFrame
        dataframe with labels and NLP features

    Returns
    -------
    list
        list of all labels
    """
    return df["medical_specialty"].tolist()


# Split the dataframe into test and train data
def split_data(df: pd.DataFrame) -> array:
    """
    Split the dataframe into test and train data

        Parameters
        ----------
        df : pd.DataFrame
            dataframe with labels and NLP features

        Returns
        -------
        dict
            dictionary with train and test data
    """
    X = df["transcription_f"].astype(str)
    y = get_labels(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


# Build pipeline
def build_pipeline() -> Pipeline:
    """
    Build pipeline for model

    """
    model_pipeline = imbPipeline(
        [
            ("preprocessing", CountVectorizer()),
            ("svd", TruncatedSVD(n_components=100)),
            # ('pca', FunctionTransformer(pca)),
            ("smote", SMOTE(random_state=42)),
            (
                "classifier",
                LogisticRegression(random_state=42, multi_class="multinomial"),
            ),  # remainder="passthrough"
        ]
    )
    return model_pipeline


# Grid search
def grid_search(
    X_train: array, y_train: array, model_pipeline: Pipeline, param_grid: list
) -> GridSearchCV:
    """
    Grid search for best model

    Parameters
    ----------
    data : dict
        dictionary with train and test data
    model_pipeline : Pipeline
        pipeline for model
    param_grid : list
        list of parameters for grid search

    Returns
    -------
    GridSearchCV
        best model
    """
    search = GridSearchCV(model_pipeline, param_grid, cv=5)
    search.fit(X_train, y_train)
    return search.best_estimator_


# Evaluate the metrics for the model
def get_model_metrics(
    best_model: GridSearchCV, X_test: array, y_test: array, category_list: list
) -> str:
    """
    get classification report for model

    Parameters
    ----------
    best_model : GridSearchCV
        best model
    data : dict
        dictionary with train and test data

    Returns
    -------
    str
        classification report
    """
    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=category_list)
    return report


def predict_proba(
    best_model: GridSearchCV, X_test: array, z: int, category_list: list
) -> pd.DataFrame:
    """
    get probabilities for sample in test data

    Parameters
    ----------
    best_model : GridSearchCV
        best model
    data : dict
        dictionary with train and test data

    Returns
    -------
    pd.DataFrame
        Probabilities for labels
    """
    prob_array = best_model.predict_proba(X_test)[z, :]
    prob_df = pd.DataFrame(
        prob_array, index=category_list, columns=["Probability"]
    ).sort_values(by="Probability", ascending=False)
    return prob_df


def main():
    # Load data
    df = df_test
    category_list = df_test.medical_specialty.unique()

    # Split data into train and test
    X_train, X_test, y_train, y_test = split_data(df)

    # build model
    model_pipeline = build_pipeline()

    # train model with grid search
    param_grid = [
        {
            "classifier__C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            "classifier": [
                LogisticRegression(multi_class="multinomial", random_state=42)
            ],
            "classifier__solver": ["saga", "lbfgs", "liblinear"],
            "classifier__penalty": ["none", "l1", "l2", "elasticnet"],
        }
    ]

    best_model = grid_search(X_train, y_train, model_pipeline, param_grid)

    # evaluate model
    report = get_model_metrics(best_model, X_test, y_test)
    print(report)

    # Predict probabilties
    prob_df = predict_proba(best_model, X_test)
    print(prob_df)

    # Save Model
    model_name = "sklearn_logistic_regression_model.pkl"
    joblib.dump(value=best_model, filename=model_name)


if __name__ == "__main__":
    main()
