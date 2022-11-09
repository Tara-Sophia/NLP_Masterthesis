# -*- coding: utf-8 -*-
import streamlit as st
from helper_02 import KeywordExtraction, clean_input

user_input = st.text_input("Enter your symptom text: ")
button = st.button("Analyze", key="1")


# clean input text with functions from predicting.py
def clean_text_input(text):
    text_clean = clean_input(text)
    return text_clean


# Get keywords from input text with functions from predicting.py
@st.cache(allow_output_mutation=True)
def get_keywords(text):
    keywords = KeywordExtraction(text)
    return keywords


# if user input and button key 1 is pressed

if button:
    print(user_input)
    # clean input text with functions from predicting.py
    # text_clean = clean_input(user_input)
    # predict keywords with functions from predicting.py
    keywords = get_keywords(user_input)
    # output field write keywords
    st.write("the important keywords are", keywords)
