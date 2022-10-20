# model neu laden und funktion, wo ein wert eingefügt wird
# parameter model und ein wert
# load_model()
# predict_value()

import imblearn
import pickle
import joblib
import pandas as pd
import numpy as np


def load_model(model_name: str) -> imblearn.pipeline.Pipeline:
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
    model = joblib.load(filename=model_name)
    # model = pickle.load(open(model_name, 'rb'))
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
    # value = np.array(value)
    prob_array = model.predict_proba(value)
    prob_df = pd.DataFrame(
        prob_array, index=category_list, columns=["Probability"]
    ).sort_values(by="Probability", ascending=False)
    return prob_df


def main():
    # Load model
    model_name = "./models/sklearn_logistic_regression_model.pkl"
    model = load_model()
    # Predict probability
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
