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
        training_data : pd.DataFrame
            train data
        testing_data : pd.DataFrame
            test data
    """
    training_data, testing_data = train_test_split(df, test_size=0.2, random_state=42)

    return training_data, testing_data


# Build pipeline
def build_pipeline() -> imblearn.pipeline.Pipeline:
    """
    Build pipeline for model

    """
    model_pipeline = imbPipeline(
        [
            ("preprocessing", CountVectorizer()),
            ("smote", SMOTE(random_state=42)),
            (
                "classifier",
                LogisticRegression(
                    random_state=42,
                    multi_class="multinomial",
                    penalty="l1",
                    solver="saga",
                ),
            ),  # remainder="passthrough"
        ]
    )
    return model_pipeline


# Fit model
def fit_model(
    model: imblearn.pipeline.Pipeline, X_train, y_train
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
    print("Best parameters:", search.best_params_)
    return search.best_estimator_


# Evaluate the metrics for the model
def get_model_metrics(
    best_model: imblearn.pipeline.Pipeline,
    X_test: pd.core.series.Series,
    y_test: list,
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
    report = classification_report(y_test, y_pred, target_names=best_model.classes_)

    y_preb_probs = best_model.predict_proba(X_test)
    return report


# Other metrics for model
def _reciprocal_rank(true_labels: list, machine_preds: list):
    """Compute the reciprocal rank at cutoff k"""

    # add index to list only if machine predicted label exists in true labels
    tp_pos_list = [(idx + 1) for idx, r in enumerate(machine_preds) if r in true_labels]

    rr = 0
    if len(tp_pos_list) > 0:
        # for RR we need position of first correct item
        first_pos_list = tp_pos_list[0]

        # rr = 1/rank
        rr = 1 / float(first_pos_list)

    return rr


def compute_mrr_at_k(items: list):
    """Compute the MRR (average RR) at cutoff k"""
    rr_total = 0

    for item in items:
        rr_at_k = _reciprocal_rank(item[0], item[1])
        rr_total = rr_total + rr_at_k
        mrr = rr_total / 1 / float(len(items))

    return mrr


def compute_accuracy(eval_items: list):
    correct = 0
    total = 0

    for item in eval_items:
        true_pred = item[0]
        machine_pred = set(item[1])

        for cat in true_pred:
            if cat in machine_pred:
                correct += 1
                break

    accuracy = correct / float(len(eval_items))
    return accuracy


def collect_preds(Y_test, Y_preds):
    """Collect all predictions and ground truth"""

    pred_gold_list = [[[Y_test.iloc[idx]], pred] for idx, pred in enumerate(Y_preds)]
    return pred_gold_list


def get_top_k_predictions(model, X_test, k):

    # get probabilities instead of predicted labels, since we want to collect top 3
    probs = model.predict_proba(X_test)

    # GET TOP K PREDICTIONS BY PROB - note these are just index
    best_n = np.argsort(probs, axis=1)[:, -k:]

    # GET CATEGORY OF PREDICTIONS
    preds = [
        [model.classes_[predicted_cat] for predicted_cat in prediction]
        for prediction in best_n
    ]

    preds = [item[::-1] for item in preds]

    return preds


def main():
    # Load data
    file_path = os.path.join("data", "processed", "mtsamples_nlp.csv")
    df = load_data(file_path)
    category_list = df.medical_specialty.unique()

    # Split data into train and test
    training_data, testing_data = split_data(df)

    # build model
    model_pipeline = build_pipeline()

    # fit model (without grid search)
    model = fit_model(
        model_pipeline, training_data.transcription_f, training_data.medical_specialty
    )

    # # fit model with grid search (for sake of time, grid search only has few parameters)
    # param_grid = [
    #     {
    #         "classifier__C": [0.01, 0.1, 1, 10],
    #         "classifier": [
    #             LogisticRegression(
    #                 multi_class="multinomial", random_state=42, solver="saga"
    #             )
    #         ],
    #         "classifier__penalty": ["l1", "l2", "elasticnet"],
    #     }
    # ]

    # best_model = grid_search(X_train, y_train, model_pipeline, param_grid)

    # evaluate model
    report = get_model_metrics(
        model, testing_data.transcription_f, testing_data.medical_specialty
    )
    print(report)

    # evaluate model with other metrics
    # GET TOP K PREDICTIONS
    preds = get_top_k_predictions(model, testing_data.transcription_f, 3)
    # GET PREDICTED VALUES AND GROUND TRUTH INTO A LIST OF LISTS - for ease of evaluation
    eval_items = collect_preds(testing_data.medical_specialty, preds)
    # COMPUTE MRR AT K
    mrr_at_k = compute_mrr_at_k(eval_items)
    print("MRR at k: ", mrr_at_k)
    # COMPUTE ACCURACY AT K
    accuracy = compute_accuracy(eval_items)
    print("Accuracy: ", accuracy)

    # Save Model
    filename = "./models/sklearn_logistic_regression_model.pkl"
    pickle.dump(model, open(filename, "wb"))


if __name__ == "__main__":
    main()
