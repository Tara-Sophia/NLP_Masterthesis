# -*- coding: utf-8 -*-
"""
Description:
    Implementation of the Streamlit app

Usage:
    $ streamlit run src/streamlit/app.py
"""
import streamlit as st
from src.streamlit.app_clf import clf_main
from src.streamlit.app_nlp import show_nlp
from src.streamlit.app_stt import stt_main

# Settings
st.set_page_config(layout="wide")

# Title
st.title("Masterthesis Demo")
st.text("This is a demo of the masterthesis project.")

# Sidebar (Only for demo purposes)
st.sidebar.title("Debug options")
debug = st.sidebar.checkbox("Debug mode", value=False)

# Initialization
if "show_page" not in st.session_state:
    st.session_state["show_page"] = "stt"
else:
    st.session_state["show_page"] = st.session_state["show_page"]

if "res" not in st.session_state:
    st.session_state["res"] = None
else:
    st.session_state["res"] = st.session_state["res"]


# Speech-to-text
if st.session_state.get("show_page") == "stt":  # type: ignore
    st.session_state["res"] = stt_main()
    if st.session_state.get("res"):
        st.session_state["show_page"] = "nlp"
        st.experimental_rerun()

# NLP
elif st.session_state.get("show_page") == "nlp":
    st.session_state.get("res") = show_nlp(st.session_state.get("res"))
    next = st.button("Show next")
    if next:
        st.session_state["show_page"] = "clf"
        st.experimental_rerun()

# CLF
else:
    clf_main(st.session_state.get("res"))
    redo = st.button("Rerun with new input")
    if redo:
        st.session_state["show_page"] = "stt"
        st.experimental_rerun()
