# -*- coding: utf-8 -*-
from keybert import KeyBERT
from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline

import streamlit as st


@st.cache(allow_output_mutation=True)
def load_keybert() -> KeyBERT:
    """
    Load the keybert model

    Returns
    -------
    KeyBERT
        KeyBERT pipeline
    """
    tokenizer = AutoTokenizer.from_pretrained(
        "tarasophia/Bio_ClinicalBERT_medical", model_max_length=512
    )
    model = AutoModelForMaskedLM.from_pretrained("tarasophia/Bio_ClinicalBERT_medical")

    pipe = pipeline(
        "feature-extraction",
        model=model,
        tokenizer=tokenizer,
    )

    kw_model = KeyBERT(model=pipe)
    return kw_model


def keyword_extraction(
    text: str, kw_model: KeyBERT, nr_candidates: int, top_n: int
) -> list[tuple[str, float]]:
    """
    This function extracts keywords from the input text

    Parameters
    ----------
    text : str
        Input sentence
    kw_model : KeyBERT
        KeyBERT pipeline
    nr_candidates : int
        Number of candidates
    top_n : int
        Number of keywords to extract

    Returns
    -------
    list[tuple[str, float]]
        List of keywords and their weights
    """
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),
        stop_words="english",
        use_maxsum=True,
        nr_candidates=nr_candidates,
        top_n=top_n,
        use_mmr=True,
        diversity=0.5,
    )
    return keywords


def nlp_main(str_nlp: str) -> str:
    """
    Main function for the NLP part of the app

    Parameters
    ----------
    str_nlp : str
        Input text from the speech-to-text part

    Returns
    -------
    str
        Output text from the NLP part
    """

    kw_model = load_keybert()
    keywords_weights = keyword_extraction(str_nlp, kw_model, 20, 10)
    keywords = [keyword[0] for keyword in keywords_weights]
    return keywords
