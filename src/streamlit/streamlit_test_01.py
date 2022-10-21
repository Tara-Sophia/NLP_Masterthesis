# -*- coding: utf-8 -*-
import streamlit as st

from helper import *

# importing all the helper fxn from helper.py which we will create later

import os

import matplotlib.pyplot as plt

import seaborn as sns

# from helper file we are importing load_model and predict_probability
file_path = os.path.join("models", "sklearn_logistic_regression_model.pkl")
model = load_model(file_path)

st.title("Welcome to medical symptom checker")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Speech to text", "Transcription", "Classification", "Probabilities"]
)

with tab1:
    st.write("Coming soon")

with tab2:
    with st.form("inputfield", clear_on_submit=True):
        transcription = st.text_area("Please describe how you are feeling")

        submit = st.form_submit_button("Submit")
        if submit:
            st.write("Your text: ", transcription)
            keywords = [
                "cardiac",
                "ventricular",
                "mitral",
                "left",
                "valve",
            ]
            st.write("These are the most important keywords")
            st.write(transcription)


with tab3:
    prediction = predict_probability(model, [transcription])
    prob1 = prediction["Probability"][0]
    prob2 = prediction["Probability"][1]
    prob3 = prediction["Probability"][2]

    st.subheader(
        "Based on our algorithm you should consider contacting these departments"
    )
    with st.expander(prediction.index[0]):
        st.metric(label="Percentage of probability", value="{:.0%}".format(prob1))
        st.write("Decision was based on these symptoms from your description:")
        st.write("coming soon")
        st.write("Most relevant symptoms for this department in general:")
        st.write("coming soon")

    with st.expander(prediction.index[1]):
        st.metric(label="Percentage of probability", value="{:.0%}".format(prob2))
        st.write("Decision was based on these symptoms from your description:")
        st.write("coming soon")
        st.write("Most relevant symptoms for this department in general:")
        st.write("coming soon")

    with st.expander(prediction.index[2]):
        st.metric(label="Percentage of probability", value="{:.0%}".format(prob3))
        st.write("Decision was based on these symptoms from your description:")
        st.write("coming soon")
        st.write("Most relevant symptoms for this department in general:")
        st.write("coming soon")

with tab4:
    prediction = predict_probability(model, [transcription])
    st.subheader("Probability of each department")
    st.write(prediction)

# Transcipription
# DESCRIPTION:,1.  Normal cardiac chambers size.,2.  Normal left ventricular size.,3.  Normal LV systolic function.  Ejection fraction estimated around 60%.,4.  Aortic valve seen with good motion.,5.  Mitral valve seen with good motion.,6.  Tricuspid valve seen with good motion.,7.  No pericardial effusion or intracardiac masses.,DOPPLER:,1.  Trace mitral regurgitation.,2.  Trace tricuspid regurgitation.,IMPRESSION:,1.  Normal LV systolic function.,2.  Ejection fraction estimated around 60%.
