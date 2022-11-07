"""
Description:
    Training a logistic regression model for predicting probabilities of medical specialties
"""
from array import array
import pandas as pd
import numpy as np
import pickle
import imblearn
import os

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as imbPipeline

from traitlets import List

from utils import load_data
from utils import preprocessing_pipeline

# Build pipeline
def build_pipeline(pipeline) -> imblearn.pipeline.Pipeline:
    """
    Build pipeline for model

    Parameters
    ----------
    preprocessing_pipeline : imblearn.pipeline.Pipeline
        pipeline with preprocessing steps
    Returns
    -------
    imblearn.pipeline.Pipeline
        pipeline for model
    """
    pipeline.steps.append(
        (
            "clf",
            LogisticRegression(
                random_state=42,
                multi_class="multinomial",
                penalty="l1",
                solver="saga",
            ),
        )
    )
    return pipeline


# Fit model
def fit_model(
    model: imblearn.pipeline.Pipeline,
    X_train: pd.core.series.Series,
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
    model, X_train: pd.core.series.Series, y_train: pd.core.series.Series
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
    X_train : pd.core.series.Series
        train data
    y_train : pd.core.series.Series
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
        model_pipeline, param_grid, cv=5, scoring=custom_accuracy_function
    )
    search.fit(X_train, y_train)
    print("Best parameters:", search.best_params_)
    print("Best cross-validation score: {:.2f}".format(search.best_score_))
    return search.best_estimator_


def main():
    # Load train data
    train_file_path = os.path.join("data", "processed", "clf", "train.csv")
    X_train, y_train = load_data(train_file_path)
    print(type(X_train))
    # Load test data
    test_file_path = os.path.join("data", "processed", "clf", "test.csv")
    X_test, y_test = load_data(test_file_path)

    # Build pipeline
    preprocessing = preprocessing_pipeline()
    model_pipeline = build_pipeline(preprocessing)

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

    # Save Model
    filename = "./models/clf/lr_test_2.pkl"
    pickle.dump(model, open(filename, "wb"))


if __name__ == "__main__":
    main()
