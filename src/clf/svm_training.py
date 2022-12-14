# -*- coding: utf-8 -*-
"""
Description:
    Training a Linear Support Vector Machine classification model
    for predicting probabilities of medical specialties

Usage:
    $ python src/clf/svm_training.py

Possible arguments:
    * -g or --gridsearch: run gridsearch

"""
import pickle

import click
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from src.clf.constants import SVM_MIMIC_CLASSIFIED, TRAIN_DATA_DIR
from src.clf.utils import load_data, preprocessing_pipeline


# Build pipeline
def build_pipeline(
    pipeline: Pipeline,
) -> Pipeline:
    """
    Build pipeline for model
    Parameters
    ----------
    preprocessing_pipeline : Pipeline
        pipeline with preprocessing steps
    Returns
    -------
    Pipeline
        pipeline for model
    """
    pipeline.steps.append(
        (
            "clf",
            SVC(
                random_state=42,
                probability=True,
                C=1,
                gamma=0.01,
                kernel="rbf",
            ),
        )
    )
    return pipeline


# Grid search and custom scorer with accuracy @k
def custom_accuracy_function(
    model: Pipeline,
    X_train: pd.Series,
    y_train: pd.Series,
) -> float:
    """
    Custom scorer with accuracy @3
    Parameters
    ----------
    model : Pipeline
        pipeline for model
    X_test: pd.Series
        test data
    y_test: pd.Series
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
    X_train: pd.Series,
    y_train: pd.Series,
    model_pipeline: Pipeline,
    param_grid: list[dict[str, object]],
) -> Pipeline:
    """
    Grid search for best model
    Parameters
    ----------
    X_train : pd.Series
        train data
    y_train : pd.Series
        train labels
    model_pipeline : Pipeline
        pipeline for model
    param_grid : list[dict[str, list[float]]]
        list of parameters for grid search
    Returns
    -------
    Pipeline
        best model
    """
    search = GridSearchCV(
        model_pipeline,
        param_grid,
        cv=5,
        scoring=custom_accuracy_function,
        verbose=2,
    )
    search.fit(X_train, y_train)
    print("Best parameters:", search.best_params_)
    print("Best cross-validation score: {:.2f}".format(search.best_score_))
    return search.best_estimator_


@click.command()
@click.option(
    "--gridsearch",
    "-g",
    help="perform grid search",
    default=False,
    is_flag=True,
    required=False,
)
def main(gridsearch: bool):
    """
    Main function
    """
    # Load train data
    X_train, y_train = load_data(TRAIN_DATA_DIR)

    # Build pipeline
    preprocessing = preprocessing_pipeline()
    model_pipeline = build_pipeline(preprocessing)

    # Grid search
    if gridsearch:
        print("performing with grid search")
        param_grid = [
            {
                "clf__C": [0.1, 1, 10],
                "clf__gamma": [1, 0.1, 0.01],
                "clf__kernel": ["rbf", "linear"],
            }
        ]
        model = grid_search(
            X_train,
            y_train,
            model_pipeline,
            param_grid,
        )

    else:
        print("performing without grid search")
        model = model_pipeline.fit(X_train, y_train)

    # Save Model
    pickle.dump(model, open(SVM_MIMIC_CLASSIFIED, "wb"))


if __name__ == "__main__":
    main()
