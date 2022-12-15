# -*- coding: utf-8 -*-
"""
Description:
    Implementation of the Streamlit app

Usage:
    $ streamlit run src/streamlit/app.py
"""

import streamlit as st
from src.streamlit.app_clf import clf_main
from src.streamlit.app_nlp import nlp_main
from src.streamlit.app_stt import stt_main

# Specify the title and logo for the web page.

st.set_page_config(
    page_title="Symptom Checker",
    page_icon="https://static.thenounproject.com/png/1630376-200.png",
    layout="wide",
)

st.title("Symptom Checker")

st.markdown("---")

# Sidebar Configuration
st.sidebar.image("https://static.thenounproject.com/png/1630376-200.png", width=100)
st.sidebar.markdown("# Medical Symptom Checker Demo")
st.sidebar.markdown("This is a demo of the masterthesis project.")
st.sidebar.markdown(
    "The project is about the development of a symptom checker \
    for the diagnosing medical departments."
)


st.sidebar.markdown("---")
st.sidebar.write(
    "Developed by Florentin von Haugwitz, Tara-Sophia Tumbraegel, Hannah Petry"
)
st.sidebar.write("Contact at 48458@novasbe.pt")


# Initialization
if "show_page" not in st.session_state:
    st.session_state["show_page"] = "stt"

if "res" not in st.session_state:
    st.session_state["res"] = None


# Speech-to-text
if st.session_state.get("show_page") == "stt":
    st.session_state["res"] = stt_main()
    if st.session_state.get("res"):
        st.session_state["res"] = nlp_main(st.session_state.get("res"))
        next = st.button("Show best departments", type="primary")
        if next:
            st.session_state["show_page"] = "clf"
            st.experimental_rerun()

# CLF
else:
    clf_main(st.session_state.get("res"))
    redo = st.button("Rerun with new input", type="primary")
    if redo:
        st.session_state["show_page"] = "stt"
        st.experimental_rerun()
