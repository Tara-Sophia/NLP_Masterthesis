# -*- coding: utf-8 -*-

"""
Description:
    Training a logistic regression model
    for predicting probabilities of medical specialties

Usage:

Possible arguments:
    *
"""
import ast
import os
import pickle

import imblearn
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbPipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split


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


# Transform data for model
def replace_tab(x):
    return [i.replace(" ", "_") for i in x]


def transform_column(
    df: pd.DataFrame, column_name: str
) -> pd.DataFrame:
    """
    Transform column to list

    Parameters
    ----------
    df : pd.DataFrame
        dataframe with labels and NLP features
    column_name : str
        column name

    Returns
    -------
    pd.DataFrame
        dataframe with transformed column
    """
    df[column_name] = df[column_name].apply(
        lambda x: ast.literal_eval(x)
    )
    df[column_name] = df[column_name].apply(lambda x: replace_tab(x))
    df[column_name] = df[column_name].apply(lambda x: " ".join(x))
    return df


# Build pipeline
def build_pipeline() -> imblearn.pipeline.Pipeline:
    """
    Build pipeline for model

    Parameters
    ----------
    analyzer : function
        analyzer function for CountVectorizer

    Returns
    -------
    imblearn.pipeline.Pipeline
        pipeline for model
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
                    # max_iter=5000,
                ),
            ),
        ]
    )
    return model_pipeline


# Fit model
def fit_model(
    model: imblearn.pipeline.Pipeline,
    X_train: list,
    y_train: pd.core.series.Series,
) -> imblearn.pipeline.Pipeline:
    """
    Fit model

    Parameters
    ----------
    model : imblearn.pipeline.Pipeline
        pipeline for model
    X_train : pd.core.series.Series
        train data
    y_train : pd.core.series.Series
        train labels

    Returns
    -------
    imblearn.pipeline.Pipeline
        fitted model
    """
    model.fit(X_train, y_train)
    return model


# Grid search and custom scorer with accuracy @k
def custom_accuracy_function(
    model,
    X_train: pd.core.series.Series,
    y_train: pd.core.series.Series,
) -> float:
    """
    Custom scorer with accuracy @3

    Parameters
    ----------
    model : imblearn.pipeline.Pipeline
        pipeline for model
    X_test: pd.core.series.Series
        test data
    y_test: pd.core.series.Series
        test labels

    Returns
    -------
    float
        accuracy @3
    """
    k = 3
    y_preb_probs = model.predict_proba(X_train)
    top = np.argsort(y_preb_probs, axis=1)[:, -k:]
    top = np.apply_along_axis(lambda x: model.classes_[x], 1, top)
    actual = np.array(y_train).reshape(-1, 1)
    return np.any(top == actual, axis=1).mean()


def grid_search(
    X_train: pd.core.series.Series,
    y_train: pd.core.series.Series,
    model_pipeline: imblearn.pipeline.Pipeline,
    param_grid: list,
) -> imblearn.pipeline.Pipeline:
    """
    Grid search for best model

    Parameters
    ----------
    X_train :
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
    search = GridSearchCV(
        model_pipeline,
        param_grid,
        cv=5,
        scoring=custom_accuracy_function,
    )
    search.fit(X_train, y_train)
    print("Best parameters:", search.best_params_)
    print(
        "Best cross-validation score: {:.2f}".format(
            search.best_score_
        )
    )
    return search.best_estimator_


# Classification report with common metrics for the model
# (not the metrics we are optimizing for)
def get_model_metrics(
    best_model: imblearn.pipeline.Pipeline,
    X_test: pd.Series,
    y_test: pd.Series,
) -> str:
    """
    get classification report for model

    Parameters
    ----------
    best_model : GridSearchCV
        best model
    X_test: pd.Series
        test data
    y_test: pd.Series
        test labels
    Returns
    -------
    str
        classification report
    """
    y_pred = best_model.predict(X_test)
    report = classification_report(
        y_test, y_pred, target_names=best_model.classes_
    )
    return report


# Other metrics for model evaluation (accuracy @k optimized for and MRR @k)
def _reciprocal_rank(true_labels: list, machine_preds: list) -> float:
    """
    Compute the reciprocal rank at cutoff k

    Parameters
    ----------
    true_labels : list
        true labels
    machine_preds : list
        machine predictions

    Returns
    -------
    float
        reciprocal rank
    """

    # add index to list only if machine predicted label exists in true labels
    tp_pos_list = [
        (idx + 1)
        for idx, r in enumerate(machine_preds)
        if r in true_labels
    ]

    rr = 0.0
    if len(tp_pos_list) > 0:
        # for RR we need position of first correct item
        first_pos_list = tp_pos_list[0]

        # rr = 1/rank
        rr = 1 / float(first_pos_list)

    return rr


def compute_mrr_at_k(items: list) -> float:
    """
    Compute the MRR (average RR) at cutoff k

    Parameters
    ----------
    items : list
        list of tuples (true labels, machine predictions)

    Returns
    -------
    float
        MRR @k
    """
    rr_total = 0.0

    for item in items:
        rr_at_k = _reciprocal_rank(item[0], item[1])
        rr_total = rr_total + rr_at_k
        mrr = rr_total / 1 / float(len(items))

    return mrr


def compute_accuracy(eval_items: list[tuple[str, str]]) -> float:
    """
    Compute the accuracy at cutoff k

    Parameters
    ----------
    eval_items : list[tuple[str, str]]
        list of tuples (true labels, machine predictions)

    Returns
    -------
    float
        accuracy @k
    """
    correct = 0

    for item in eval_items:
        true_pred = item[0]
        machine_pred = set(item[1])

        for cat in true_pred:
            if cat in machine_pred:
                correct += 1
                break

    accuracy = correct / float(len(eval_items))
    return accuracy


def collect_preds(Y_test: pd.Series, Y_preds: list) -> list:
    """
    Collect all predictions and ground truth

    Parameters
    ----------
    Y_test : pd.Series
        true labels
    Y_preds : list
        list of machine predictions

    Returns
    -------
    list
        list of tuples (true labels, machine predictions)
    """

    pred_gold_list = [
        [[Y_test.iloc[idx]], pred] for idx, pred in enumerate(Y_preds)
    ]
    return pred_gold_list


def get_top_k_predictions(
    model: imblearn.pipeline.Pipeline,
    X_test: pd.core.series.Series,
    k: int,
) -> list:
    """
    Get top k predictions for each test sample

    Parameters
    ----------
    model : imblearn.pipeline.Pipeline
        model
    X_test : pd.core.series.Series
        test data
    k : int
        number of predictions

    Returns
    -------
    list
        list of top k predictions
    """

    probs = model.predict_proba(X_test)
    best_n = np.argsort(probs, axis=1)[:, -k:]
    preds = [
        [
            model.classes_[predicted_cat]
            for predicted_cat in prediction
        ]
        for prediction in best_n
    ]

    preds = [item[::-1] for item in preds]
    return preds


def main():
    # Load data
    file_path = os.path.join(
        "data",
        "processed",
        "nlp",
        "mtsamples",
        "mtsamples_unsupervised_both_v2.csv",
    )
    df = load_data(file_path)
    df = transform_column(df, "transcription_f_semisupervised")
    print(df.shape)

    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        df.transcription_f_semisupervised.to_list(),
        df.medical_specialty,
        test_size=0.2,
        random_state=42,
    )

    # build model
    model_pipeline = build_pipeline()

    # fit model (without grid search)
    model = fit_model(
        model_pipeline,
        X_train,
        y_train,
    )

    # # fit model with grid search

    # param_grid = [
    #     {
    #         "classifier__C": [0.01, 0.1, 1, 10],
    #     }
    # ]

    # best_model = grid_search(
    #     X_train,
    #     y_train,
    #     model_pipeline,
    #     param_grid,
    # )

    # evaluate model
    print("Model metrics not optimized for:")
    report = get_model_metrics(
        model,
        X_test,
        y_test,
    )
    print(report)

    # evaluate model with other metrics
    # get top k predictions
    preds = get_top_k_predictions(model, X_test, 3)
    # get predicted values and ground truth into list of lists
    eval_items = collect_preds(y_test, preds)
    # compute mrr at k
    mrr_at_k = compute_mrr_at_k(eval_items)
    print("MRR at k: ", mrr_at_k)
    # compute accuracy at k
    accuracy = compute_accuracy(eval_items)
    print("Accuracy at k: ", accuracy)

    # Save Model
    filename = "./models/clf/lr_test.pkl"
    pickle.dump(model, open(filename, "wb"))


if __name__ == "__main__":
    main()
