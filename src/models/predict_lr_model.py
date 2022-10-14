# model neu laden und funktion, wo ein wert eingefügt wird
# parameter model und ein wert
# load_model()
# predict_value()


import joblib
import pandas as pd
import numpy as np


def load_model() -> object:
    """
    Load model from disk

    Returns
    -------
    object
        model
    """
    model_name = "./models/sklearn_logistic_regression_model.pkl"
    model = joblib.load(filename=model_name)
    return model


def predict_probability(model: object, value: str, category_list: list) -> pd.DataFrame:
    """
    get probabilities for sample

    Parameters
    ----------
    model : object
        model
    value : str
        sample

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
    model = load_model()
    print(model)
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
