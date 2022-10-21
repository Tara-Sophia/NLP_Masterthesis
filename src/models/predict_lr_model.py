# -*- coding: utf-8 -*-

"""
Description:
    Predicting probability for medical labels from saved model 
    
Usage:
    
Possible arguments:
    * 
"""

import imblearn
import pickle
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer


def load_model(file_path: str) -> imblearn.pipeline.Pipeline:
    """
    Load model from disk

    Parameters
    ----------
    model_name : str
        path to saved model

    Returns
    -------
    imblearn.pipeline.Pipeline
        best model from train.py
    """
    model = pickle.load(open(file_path, "rb"))
    return model


def predict_probability(model: imblearn.pipeline.Pipeline, value: str) -> pd.DataFrame:
    """
    get probabilities for sample

    Parameters
    ----------
    model : imblearn.pipeline.Pipeline
        best model from train.py
    value : str
        sample
    category_list: list[str]
        list of unique labels

    Returns
    -------
    pd.DataFrame
        Probabilities for labels
    """

    prob_array = model.predict_proba(value)
    prob_df = (
        pd.DataFrame(prob_array, index=["Probability"], columns=model.classes_)
        .transpose()
        .sort_values(by="Probability", ascending=False)
    )
    return prob_df


def main():
    # Load model
    file_path = os.path.join("models", "sklearn_logistic_regression_model.pkl")
    model = load_model(file_path)

    # Predict probability
    to_pred = "subglottic patient barium lateral cookie"
    res_df = predict_probability(model, [to_pred])
    print(res_df)


if __name__ == "__main__":
    main()
