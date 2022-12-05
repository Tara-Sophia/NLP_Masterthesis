# -*- coding: utf-8 -*-
import time

import streamlit as st

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
st.sidebar.markdown("# Master Thesis Demo")
st.sidebar.markdown("This is a demo of the masterthesis project.")
st.sidebar.markdown(
    "The project is about the development of a symptom checker for the diagnosing medical departments."
)


st.sidebar.markdown("---")
st.sidebar.write(
    "Developed by Florentin von Haugwitz, Tara-Sophia Tumbrägel, Hannah Petry"
)
st.sidebar.write("Contact at 48458@novasbe.pt")


with st.spinner("Wait for it..."):
    time.sleep(5)
st.success("Done!")
