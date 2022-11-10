# -*- coding: utf-8 -*-
import sys
import streamlit as st

sys.path.insert(0, "src/nlp")
from constants import MODEL_UNSUPERVISED_MODEL_DIR
from KeyBert_on_Mtsamples import KeywordExtraction
from utils import cleaning_input

user_input = st.text_input("Enter your symptom text: ")
button = st.button("Analyze", key="1")

# clean input text with functions from predicting.py
def clean_text_input(text):
    text_clean = cleaning_input(text)
    return text_clean


# Get keywords from input text with functions from predicting.py
@st.cache(allow_output_mutation=True)
def get_keywords(text, model):
    keywords = KeywordExtraction(text, model)
    return keywords


# if user input and button key 1 is pressed

if button:
    print(user_input)
    # clean input text with functions from predicting.py
    # text_clean = clean_input(user_input)
    # predict keywords with functions from predicting.py
    model = MODEL_UNSUPERVISED_MODEL_DIR
    keywords = get_keywords(user_input, model)
    # output field write keywords
    st.write("the important keywords are", keywords)
