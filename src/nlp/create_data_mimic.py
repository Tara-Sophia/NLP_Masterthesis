# -*- coding: utf-8 -*-
"""
Description:
    This script cleans the data from the mimic-iii dataset.
"""

import re

import pandas as pd
from nltk.corpus import stopwords

from src.nlp.constants import (
    MIMIC_DATA_CLEANED,
    MIMIC_DATA_DIR,
    MIMIC_PERSONALIZED_STOPWORDS_FILTERED,
)


def keep_only_text_from_choosen_headers(text: str, choosen_headers: list) -> str:
    """
    Keep only text from choosen headers

    Parameters
    ----------
    text : str
        Text to clean
    choosen_headers : list
        List of choosen headers

    Returns
    -------
    str
        Text with only choosen headers
    """
    new_text = ""
    # headers is a list of headers to keep

    text = str(text)
    # get all headers with regex (\n(.*?):\n)
    all_headers = re.findall(r"(\n(.*?):\n)", text)
    # get only first value in tuple
    all_headers = [x[1] for x in all_headers]

    # get the text after choosen headers until next header
    for chosen_header in choosen_headers:
        if chosen_header in text:
            if chosen_header in all_headers:
                # get index of choosen header
                index_chosen_header = all_headers.index(chosen_header)
                # get next header
                try:
                    next_header = all_headers[index_chosen_header + 1]
                    # get index of next header
                    index_next_header = text.index(next_header)
                    text_after_chosen_header = text[
                        text.index(chosen_header)
                        + len(chosen_header) : index_next_header
                    ]
                    # add text to new text
                    new_text += text_after_chosen_header
                except IndexError:
                    print("IndexError")
                    # raise NameError('IndexError')

    return new_text


def clean_text(text: str) -> str:
    """
    Clean text

    Parameters
    ----------
    text : str
        Text to clean

    Returns
    -------
    str
       Cleaned text
    """

    # replace \n with space
    text = text.replace("\n", " ")
    # Remove all the special characters
    document = re.sub(r"\W", " ", str(text))
    # Remove all single characters
    document = re.sub(r"\s+[a-zA-Z]\s+", " ", document)
    # Substituting multiple spaces with single space
    document = re.sub(r"\s+", " ", document, flags=re.I)
    # Removing prefixed 'b'
    document = re.sub(r"^b\s+", "", document)
    # Remove digits
    document = re.sub(r"\d+", " ", document)
    # Remove everything in brackets
    document = re.sub(r"\[[^]]*\]", "", document)
    # Filter out stop words
    document = " ".join(
        [word for word in document.split() if word not in stopwords.words("english")]
    )
    # Filter out most common words
    document = " ".join(
        [
            word
            for word in document.split()
            if word not in MIMIC_PERSONALIZED_STOPWORDS_FILTERED
        ]
    )
    document = " ".join([word for word in document.split() if len(word) > 1])
    return document


def main():
    """
    Main function
    """

    df = pd.read_csv(MIMIC_DATA_DIR)
    # Choosen headers
    df["TEXT_final"] = df["TEXT"].apply(
        lambda x: keep_only_text_from_choosen_headers(
            x,
            [
                "CHIEF COMPLAINT",
                "Chief Complaint",
                "Present Illness",
                "PRESENT ILLNESS",
                "History of Present Illness",
                "Impression",
                "ROS",
                "HISTORY OF PRESENT ILLNESS",
                "Respiratory",
            ],
        )
    )
    df["TEXT_final_cleaned"] = df["TEXT_final"].apply(clean_text)

    # Save to csv
    df.to_csv(MIMIC_DATA_CLEANED, index=False)
