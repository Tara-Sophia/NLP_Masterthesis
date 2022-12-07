# -*- coding: utf-8 -*-
"""
Description:
    Implementation of the Streamlit app for the classification part of the project
"""
import pickle

import pandas as pd
from imblearn.pipeline import Pipeline
from lime.lime_text import LimeTextExplainer

import streamlit as st
from src.clf.constants import XGB_MIMIC_CLASSIFIED


def predict_probability(model: Pipeline, value: str) -> pd.DataFrame:
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
        pd.DataFrame(prob_array, index=["Probability"], columns=model.classes_)
        .transpose()
        .sort_values(by="Probability", ascending=False)
    )
    return prob_df


def top_symptoms(model: Pipeline) -> pd.Series:
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
    feat = vectorizer.get_feature_names_out()
    coef_df = pd.DataFrame(coef, columns=feat, index=model.classes_)
    top = coef_df.apply(lambda x: x.nlargest(5).index.tolist(), axis=1)
    return top


def lime_explainer(model: Pipeline, value: str):
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


def clf_main(text: str) -> None:
    """
    Main function for the classification part of the project
    """
    st.header("Classification")
    # Load model
    model = pickle.load(open(XGB_MIMIC_CLASSIFIED, "rb"))

    prediction = predict_probability(model, [text])

    # find the value of top symptoms from lime
    feat_importance = lime_explainer(model, text)
    top_symptoms_1_lime = get_words(prediction.index[0], feat_importance)
    top_symptoms_2_lime = get_words(prediction.index[1], feat_importance)
    top_symptoms_3_lime = get_words(prediction.index[2], feat_importance)

    lime_list = [top_symptoms_1_lime, top_symptoms_2_lime, top_symptoms_3_lime]

    st.subheader(
        "Based on our algorithm you should consider contacting these departments"
    )
    for i_expander, l_list in zip(range(3), lime_list):
        with st.expander(prediction.index[i_expander]):
            st.metric(
                label="Percentage of probability",
                value="{:.0%}".format(prediction["Probability"][i_expander]),
            )
            st.write("Decision was based on these symptoms from your description:")
            s = ""
            for i in l_list:
                s += "- " + i.title() + "\n"
            st.write(s)
    st.markdown(
        """
    <style>
    .streamlit-expanderHeader {
        font-size: x-large;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )
