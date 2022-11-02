# -*- coding: utf-8 -*-

"""
Description:
    Predicting probability for medical labels from saved model 
    Showing top features model uses per class in general
    Showing features from sample the model uses to predict top classes 
    
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
from lime.lime_text import LimeTextExplainer


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

    coef = model.named_steps["classifier"].coef_
    vectorizer = model.named_steps["preprocessing"]
    feat = vectorizer.get_feature_names()
    coef_df = pd.DataFrame(coef, columns=feat, index=model.classes_)
    coef_df = coef_df.abs()
    top_symptoms = coef_df.apply(lambda x: x.nlargest(5).index.tolist(), axis=1)
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
        value, model.predict_proba, num_features=num_features, top_labels=3
    )
    feat_importance = exp.as_map()
    feat_importance = {model.classes_[k]: v for k, v in feat_importance.items()}
    feat_importance = {
        k: [(value.split()[i], v) for i, v in v] for k, v in feat_importance.items()
    }
    feat_importance_pos = {
        k: [v for v in v if v[1] > 0] for k, v in feat_importance.items()
    }
    return feat_importance_pos


def get_words(x, feat_importance):
    if x in feat_importance:
        value = feat_importance[x]
        words = [v[0] for v in value]
        return words


def main():
    # Load model
    file_path = os.path.join("models", "clf", "sklearn_logistic_regression_model.pkl")
    model = load_model(file_path)

    # Predict probability
    to_pred = "coronary nitroglycerin muscle heart breast oxygen valve artery"
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
