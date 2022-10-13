# -*- coding: utf-8 -*-
import streamlit as st

st.title("Welcome to medical symptom checker")

tab1, tab2, tab3 = st.tabs(
    ["Speech to text", "Transcription", "Classification"]
)

with tab1:
    st.write("Coming soon")

with tab2:
    with st.form("inputfield", clear_on_submit=True):
        transcription = st.text_area(
            "Please describe how you are feeling"
        )

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
            st.write(keywords)


with tab3:
    st.subheader(
        "Based on our algroithm you should consider contacting these departments"
    )
    with st.expander(
        "1. Cardiovascular / Pulmonary (Click for more information)"
    ):
        st.metric(label="Percentage of probability", value="70%")
        st.write("Decision was based on theses keywords:")
        st.write(["cardiac", "valve"])

    with st.expander("2. Surgery (Click for more information)"):
        st.metric(label="Percentage of probability", value="30%")
        st.write("Decision was based on theses keywords:")
        st.write(["ventricular", "mitral"])

# Transcipription
# DESCRIPTION:,1.  Normal cardiac chambers size.,2.  Normal left ventricular size.,3.  Normal LV systolic function.  Ejection fraction estimated around 60%.,4.  Aortic valve seen with good motion.,5.  Mitral valve seen with good motion.,6.  Tricuspid valve seen with good motion.,7.  No pericardial effusion or intracardiac masses.,DOPPLER:,1.  Trace mitral regurgitation.,2.  Trace tricuspid regurgitation.,IMPRESSION:,1.  Normal LV systolic function.,2.  Ejection fraction estimated around 60%.
