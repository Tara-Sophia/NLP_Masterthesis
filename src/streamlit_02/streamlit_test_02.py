# -*- coding: utf-8 -*-
import streamlit as st
from src.nlp.constants import MODEL_MLM_DIR
from src.nlp.keybert_mtsamples import keyword_extraction
from src.nlp.utils import cleaning_input

user_input = st.text_input("Enter your symptom text: ")
button = st.button("Analyze", key="1")


def clean_text_input(text):
    text_clean = cleaning_input(text)
    return text_clean


# Get keywords from input text with functions from predicting.py
@st.cache(allow_output_mutation=True)
def get_keywords(text, model):
    keywords = keyword_extraction(text, model)
    return keywords


if button:
    print(user_input)
    # clean input text with functions from predicting.py
    # text_clean = clean_input(user_input)
    # predict keywords with functions from predicting.py
    model = MODEL_MLM_DIR
    keywords = get_keywords(user_input, model)
    # output field write keywords
    st.write("the important keywords are", keywords)
