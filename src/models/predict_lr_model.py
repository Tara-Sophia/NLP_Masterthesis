# -*- coding: utf-8 -*-

"""
Description:
    Predicting medical labels from saved model 
    
Usage:
    
Possible arguments:
    * 
"""

import imblearn
import pickle
import pandas as pd
import numpy as np
import os


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


def predict_probability(
    model: imblearn.pipeline.Pipeline, value: str, category_list: list[str]
) -> pd.DataFrame:
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
    value = value.astype(str)
    prob_array = model.predict_proba(value)
    prob_df = pd.DataFrame(
        prob_array, index=category_list, columns=["Probability"]
    ).sort_values(by="Probability", ascending=False)
    return prob_df


def main():
    # Load model
    file_path = os.path.join("models", "sklearn_logistic_regression_model.pkl")
    model = load_model(file_path)

    # Predict probability
    # value = [["subglottic"], ["patient"]]
    value = {"subglottic", "patient", "barium", "lateral", "cookie"}
    category_list = [
        " Cardiovascular / Pulmonary"
        " Urology"
        " General Medicine"
        " Surgery"
        " SOAP / Chart / Progress Notes"
        " Radiology"
        " Orthopedic"
        " Obstetrics / Gynecology"
        " Neurology"
        " Gastroenterology"
        " Consult - History and Phy."
    ]

    prob_df = predict_probability(model, value, category_list)
    print(prob_df)


if __name__ == "__main__":
    main()
