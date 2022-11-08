# -*- coding: utf-8 -*-

"""
Description:
    Predicting probability for medical labels from saved model
    Showing top features model uses per class in general
    Showing features from sample the model uses to predict top classes
"""

import pickle

import imblearn
import pandas as pd
from constants import LR_MODEL_MASKED
from lime.lime_text import LimeTextExplainer


def predict_probability(
    model: imblearn.pipeline.Pipeline, value: str
) -> pd.DataFrame:
    """
    get probabilities for sample

    Parameters
    ----------
    model : imblearn.pipeline.Pipeline
        best model from train.py
    value : str
        sample

    Returns
    -------
    pd.DataFrame
        Probabilities for labels
    """

    prob_array = model.predict_proba(value)
    prob_df = (
        pd.DataFrame(
            prob_array, index=["Probability"], columns=model.classes_
        )
        .transpose()
        .sort_values(by="Probability", ascending=False)
    )
    return prob_df


def top_symptoms(model: imblearn.pipeline.Pipeline) -> pd.Series:
    """
    Get top symptoms for each class

    Parameters
    ----------
    model : imblearn.pipeline.Pipeline
        best model from train.py

    Returns
    -------
    pd.Series
        Top symptoms for each class
    """

    coef = model.named_steps["clf"].coef_
    vectorizer = model.named_steps["vect"]
    feat = vectorizer.get_feature_names()
    coef_df = pd.DataFrame(coef, columns=feat, index=model.classes_)
    top_symptoms = coef_df.apply(
        lambda x: x.nlargest(5).index.tolist(), axis=1
    )
    return top_symptoms


def lime_explainer(model: imblearn.pipeline.Pipeline, value: str):
    """
    Get features the model used for top predicted classes

    Parameters
    ----------
    model : imblearn.pipeline.Pipeline
        best model from train.py
    value : str
        sample

    Returns
    -------
    dict
        Features from sample the model used to predict classes
    """
    explainer = LimeTextExplainer(class_names=model.classes_)
    num_features = len(value.split())
    exp = explainer.explain_instance(
        value,
        model.predict_proba,
        num_features=num_features,
        top_labels=3,
    )
    feat_importance = exp.as_map()
    feat_importance = {
        model.classes_[k]: v for k, v in feat_importance.items()
    }
    feat_importance = {
        k: [(value.split()[i], v) for i, v in v]
        for k, v in feat_importance.items()
    }
    feat_importance_pos = {
        k: [v for v in v if v[1] > 0]
        for k, v in feat_importance.items()
    }
    return feat_importance_pos


def main():
    # Load model
    model = pickle.load(open(LR_MODEL_MASKED, "rb"))

    # Predict probability
    to_pred = "oxygen coronary chest pain"
    res_df = predict_probability(model, [to_pred])
    print(res_df)

    # Get top symptoms
    top_symptoms_df = top_symptoms(model)
    print(top_symptoms_df)

    # features the model used for top predicted classes
    feat_importance = lime_explainer(model, to_pred)
    print(feat_importance)


if __name__ == "__main__":
    main()
