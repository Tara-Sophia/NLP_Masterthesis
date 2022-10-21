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
import pickle
import imblearn
import os

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as imbPipeline

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from imblearn.over_sampling import SMOTE
from traitlets import List

# load data


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from csv file

    Parameters
    ----------
    file_path : str
        File path to csv file

    Returns
    -------
    pd.DataFrame
        dataframe with labels and NLP features
    """
    df = pd.read_csv(file_path)
    df["transcription_f"] = df["transcription_f"].apply(eval)
    return df


# Retrieve list of labels
def get_labels(df: pd.DataFrame) -> list[str]:
    """
    Get list of labels from dataframe

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
def split_data(df: pd.DataFrame) -> tuple:
    """
    Split the dataframe into test and train data

        Parameters
        ----------
        df : pd.DataFrame
            dataframe with labels and NLP features

        Returns
        -------
        X_train : pd.core.series.Serie
            train data
        X_test : pd.core.series.Serie
            test data
        y_train : list
            train labels
        y_test : list
            test labels
    """
    df["transcription_f"] = df["transcription_f"].apply(lambda x: " ".join(map(str, x)))
    X = df["transcription_f"]
    y = get_labels(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


# Build pipeline
def build_pipeline() -> imblearn.pipeline.Pipeline:
    """
    Build pipeline for model

    """
    model_pipeline = imbPipeline(
        [
            ("preprocessing", CountVectorizer()),
            # ("svd", TruncatedSVD(n_components=100)),
            # ('pca', FunctionTransformer(pca)),
            ("smote", SMOTE(random_state=42)),
            (
                "classifier",
                LogisticRegression(random_state=42, multi_class="multinomial"),
            ),  # remainder="passthrough"
        ]
    )
    return model_pipeline


# Fit model
def fit_model(
    model: imblearn.pipeline.Pipeline, X_train: pd.core.series.Series, y_train: list
) -> imblearn.pipeline.Pipeline:
    """
    Fit model

    Parameters
    ----------
    model : imblearn.pipeline.Pipeline
        pipeline for model

    Returns
    -------
    Pipeline
        fitted model
    """
    model.fit(X_train, y_train)
    return model


# Grid search
def grid_search(
    X_train: pd.core.series.Series,
    y_train: list,
    model_pipeline: imblearn.pipeline.Pipeline,
    param_grid: list,
) -> imblearn.pipeline.Pipeline:
    """
    Grid search for best model

    Parameters
    ----------
    X_train : pd.core.series.Series
        train data
    y_train : list
        train labels
    model_pipeline : imblearn.pipeline.Pipeline
        pipeline for model
    param_grid : list
        list of parameters for grid search

    Returns
    -------
    imblearn.pipeline.Pipeline
        best model
    """
    search = GridSearchCV(model_pipeline, param_grid, cv=5)
    search.fit(X_train, y_train)
    return search.best_estimator_


# Evaluate the metrics for the model
def get_model_metrics(
    best_model: imblearn.pipeline.Pipeline,
    X_test: pd.core.series.Series,
    y_test: list,
    category_list: list[str],
) -> str:
    """
    get classification report for model

    Parameters
    ----------
    best_model : GridSearchCV
        best model
    X_test: pd.core.series.Series
        test data
    y_test: list
        test labels
    Returns
    -------
    str
        classification report
    """
    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=category_list)

    y_preb_probs = best_model.predict_proba(X_test)
    return report


def main():
    # Load data
    file_path = os.path.join("data", "processed", "mtsamples_nlp.csv")
    df = load_data(file_path)
    category_list = df.medical_specialty.unique()

    # Split data into train and test
    X_train, X_test, y_train, y_test = split_data(df)

    # build model
    model_pipeline = build_pipeline()

    # fit model (without grid search)
    # model_pipeline = fit_model(model_pipeline, X_train, y_train)

    # fit model with grid search
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
    report = get_model_metrics(best_model, X_test, y_test, category_list)
    print(report)

    # Save Model
    filename = "./models/sklearn_logistic_regression_model.pkl"
    pickle.dump(best_model, open(filename, "wb"))


if __name__ == "__main__":
    main()
